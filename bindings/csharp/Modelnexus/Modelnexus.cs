using System.Text.Json;
using System.Text.Json.Serialization;

namespace Modelnexus;

/// <summary>A failure reported by the native core.</summary>
/// <remarks>
/// <see cref="Code"/> is stable and identical across every language binding;
/// the message is for humans.
/// </remarks>
public sealed class ModelException : Exception
{
    public string Code { get; }

    public ModelException(string code, string message)
        : base(string.IsNullOrEmpty(code) ? message : $"{code}: {message}")
        => Code = code;
}

/// <summary>One turn of a conversation.</summary>
public sealed record Message(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content);

/// <summary>A call the model proposes. modelnexus never executes it.</summary>
public sealed record ToolCall(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("arguments")] string Arguments);

/// <summary>Token accounting for one call.</summary>
public sealed record Usage(
    [property: JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: JsonPropertyName("completion_tokens")] int CompletionTokens,
    [property: JsonPropertyName("total_tokens")] int TotalTokens);

/// <summary>The result of one inference call.</summary>
public sealed record Response(
    [property: JsonPropertyName("type")] string Type,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("tool_calls")] IReadOnlyList<ToolCall>? ToolCalls,
    [property: JsonPropertyName("finish_reason")] string? FinishReason,
    [property: JsonPropertyName("usage")] Usage? Usage);

/// <summary>One LoRA adapter applied to a chat context.</summary>
public sealed record Adapter(
    [property: JsonPropertyName("id")] int Id,
    [property: JsonPropertyName("path")] string Path,
    [property: JsonPropertyName("scale")] double Scale);

/// <summary>One scored document. <see cref="Index"/> is its position in the ORIGINAL list.</summary>
public sealed record RerankHit(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("score")] double Score);

/// <summary>How token vectors are reduced to one vector per input.</summary>
public enum Pooling
{
    /// <summary>The model's own default.</summary>
    Default,
    /// <summary>Average token vectors -- the usual choice for sentence embeddings.</summary>
    Mean,
    /// <summary>The first token's vector, for BERT-style encoders.</summary>
    Cls,
    /// <summary>The final token's vector, for decoder-style embedders.</summary>
    Last,
    /// <summary>Attaches the classification head. Required for reranking, useless otherwise.</summary>
    Rank,
    /// <summary>No pooled vector at all.</summary>
    None
}

internal static class Json
{
    internal static readonly JsonSerializerOptions Options = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNameCaseInsensitive = true
    };

    /// <summary>Throw if the core returned error JSON; otherwise hand back the document.</summary>
    internal static JsonElement Check(string raw)
    {
        if (string.IsNullOrEmpty(raw))
            throw new ModelException("EMPTY_RESPONSE", "the core returned nothing");

        var doc = JsonDocument.Parse(raw).RootElement;
        if (doc.TryGetProperty("type", out var type) && type.GetString() == "error")
        {
            var code = "UNKNOWN";
            var message = "";
            if (doc.TryGetProperty("error", out var err))
            {
                if (err.TryGetProperty("code", out var c)) code = c.GetString() ?? code;
                if (err.TryGetProperty("message", out var m)) message = m.GetString() ?? "";
            }
            throw new ModelException(code, message);
        }
        return doc;
    }
}

/// <summary>Entry points that need no loaded model.</summary>
public static class Modelnexus
{
    /// <summary>Bridge version and the llama.cpp tag it was linked against.</summary>
    public static string Version() => Native.BorrowString(Native.Version());

    /// <summary>The os-arch key used for the staged native directory layout.</summary>
    public static string PlatformKey() => Native.PlatformKey();

    /// <summary>Inspect a GGUF's tool-calling capability without loading an engine.</summary>
    public static JsonElement ModelInfo(string ggufPath) =>
        JsonDocument.Parse(Native.TakeString(Native.ModelInfo(ggufPath))).RootElement.Clone();
}

/// <summary>A loaded model and its inference context.</summary>
public sealed class Chat : IDisposable
{
    private IntPtr _handle;
    private readonly List<string> _events = new();

    // The core STORES this delegate and calls it for the whole life of the handle.
    // If it is collected while native code still holds the pointer, the process
    // crashes -- so it is a field, never a local. Verified the hard way in Python.
    private readonly Native.StringCallback _eventCallback;

    /// <summary>Load a GGUF model and create an inference engine.</summary>
    /// <remarks>
    /// Models whose chat template cannot do tool calling are rejected here rather
    /// than silently degraded.
    /// </remarks>
    public Chat(string ggufPath, Action<string>? onEvent = null)
    {
        _eventCallback = (ptr, _) =>
        {
            var text = Native.BorrowString(ptr);
            _events.Add(text);
            onEvent?.Invoke(text);
        };

        _handle = Native.ChatCreate(ggufPath, _eventCallback, IntPtr.Zero);
        if (_handle == IntPtr.Zero)
        {
            // NULL is the one place the core signals failure with a null pointer;
            // the reason arrives through the event callback instead.
            if (_events.Any(e => e.Contains("tools_unsupported")))
                throw new ModelException("MODEL_NOT_TOOL_CAPABLE",
                    $"{ggufPath} has no tool-calling chat template");
            var detail = _events.Count > 0 ? string.Join("; ", _events) : "unknown reason";
            throw new ModelException("MODEL_LOAD_FAILED", $"could not load {ggufPath} ({detail})");
        }
    }

    private void EnsureOpen()
    {
        if (_handle == IntPtr.Zero)
            throw new ModelException("ENGINE_CLOSED", "this Chat has already been closed");
    }

    /// <summary>Run one turn.</summary>
    /// <param name="request">
    /// The request object. Generation parameters travel inside it, so new ones need
    /// no change to this method.
    /// </param>
    /// <param name="onToken">Pass a handler to stream; the full response still returns.</param>
    public Response Infer(object request, Action<string>? onToken = null)
    {
        EnsureOpen();
        var payload = JsonSerializer.Serialize(request, Json.Options);

        string raw;
        if (onToken is null)
        {
            raw = Native.TakeString(Native.ChatInfer(_handle, payload));
        }
        else
        {
            // Only needs to survive the call, but keeping a local reference is what
            // guarantees that -- the GC does not know native code holds it.
            Native.StringCallback cb = (ptr, _) => onToken(Native.BorrowString(ptr));
            raw = Native.TakeString(Native.ChatInferStream(_handle, payload, cb, IntPtr.Zero));
            GC.KeepAlive(cb);
        }

        var doc = Json.Check(raw);
        return doc.Deserialize<Response>(Json.Options)
               ?? throw new ModelException("BAD_RESPONSE", "could not deserialize the response");
    }

    /// <summary>Convenience overload for a plain message list.</summary>
    public Response Infer(IEnumerable<Message> messages, int? maxTokens = null, uint? seed = null,
                          Action<string>? onToken = null)
    {
        var request = new Dictionary<string, object?>
        {
            ["messages"] = messages,
            ["max_tokens"] = maxTokens,
            ["seed"] = seed
        };
        return Infer(request, onToken);
    }

    // ---- LoRA ----

    private JsonElement Lora(object op)
    {
        EnsureOpen();
        var payload = JsonSerializer.Serialize(op, Json.Options);
        return Json.Check(Native.TakeString(Native.ChatLora(_handle, payload)));
    }

    /// <summary>Load a LoRA adapter and apply it. Returns its id.</summary>
    /// <remarks>
    /// Adapters change <i>behaviour</i> -- format, tone, tool-call reliability --
    /// not knowledge. For facts, retrieve.
    /// </remarks>
    public int LoadLora(string path, double scale = 1.0) =>
        Lora(new { op = "load", path, scale }).GetProperty("id").GetInt32();

    /// <summary>Change an adapter's scale. Takes effect on the next inference.</summary>
    public void SetLoraScale(int id, double scale) => Lora(new { op = "set", id, scale });

    /// <summary>Unload one adapter and reapply the rest.</summary>
    public void RemoveLora(int id) => Lora(new { op = "remove", id });

    /// <summary>Unload every adapter, returning the model to its base behaviour.</summary>
    public void ClearLoras() => Lora(new { op = "clear" });

    /// <summary>The adapters currently applied, in order.</summary>
    public IReadOnlyList<Adapter> Loras() =>
        Lora(new { op = "list" }).GetProperty("adapters")
            .Deserialize<List<Adapter>>(Json.Options) ?? new List<Adapter>();

    /// <summary>Release the model and its context. Idempotent.</summary>
    public void Dispose()
    {
        if (_handle == IntPtr.Zero) return;
        Native.ChatDestroy(_handle);
        _handle = IntPtr.Zero;
        GC.SuppressFinalize(this);
    }

    ~Chat() => Dispose();
}

/// <summary>A model loaded for embedding or reranking.</summary>
/// <remarks>
/// Separate from <see cref="Chat"/> because embedding needs a context built with
/// embeddings enabled and a pooling strategy fixed at creation, and reranking needs
/// <see cref="Pooling.Rank"/> specifically -- neither can be switched afterwards.
/// </remarks>
public sealed class Embedder : IDisposable
{
    private IntPtr _handle;
    private readonly List<string> _events = new();
    private readonly Native.StringCallback _eventCallback;

    public Embedder(string ggufPath, Pooling pooling = Pooling.Default,
                    int nCtx = 0, int nBatch = 512, Action<string>? onEvent = null)
    {
        _eventCallback = (ptr, _) =>
        {
            var text = Native.BorrowString(ptr);
            _events.Add(text);
            onEvent?.Invoke(text);
        };

        var config = new Dictionary<string, object?> { ["n_batch"] = nBatch };
        if (pooling != Pooling.Default) config["pooling"] = pooling.ToString().ToLowerInvariant();
        if (nCtx > 0) config["n_ctx"] = nCtx;

        _handle = Native.EmbedCreate(ggufPath, JsonSerializer.Serialize(config, Json.Options),
                                     _eventCallback, IntPtr.Zero);
        if (_handle == IntPtr.Zero)
        {
            var detail = _events.Count > 0 ? string.Join("; ", _events) : "unknown reason";
            throw new ModelException("MODEL_LOAD_FAILED", $"could not load {ggufPath} ({detail})");
        }
    }

    private void EnsureOpen()
    {
        if (_handle == IntPtr.Zero)
            throw new ModelException("ENGINE_CLOSED", "this Embedder has already been closed");
    }

    /// <summary>Embed one or more texts. One vector per input, in input order.</summary>
    /// <param name="normalize">
    /// L2-normalize, so a dot product is a cosine similarity. On by default.
    /// </param>
    public IReadOnlyList<float[]> Embed(IEnumerable<string> texts, bool normalize = true)
    {
        EnsureOpen();
        var payload = JsonSerializer.Serialize(new { input = texts, normalize }, Json.Options);
        var doc = Json.Check(Native.TakeString(Native.Embed(_handle, payload)));
        return doc.GetProperty("embeddings").Deserialize<List<float[]>>(Json.Options)
               ?? new List<float[]>();
    }

    /// <summary>Score documents against a query, best first.</summary>
    /// <remarks>
    /// Each hit carries the document's ORIGINAL index, because results come back
    /// reordered. Scores are raw model logits: comparable within one call, not across
    /// models, and not probabilities. Requires <see cref="Pooling.Rank"/>.
    /// </remarks>
    public IReadOnlyList<RerankHit> Rerank(string query, IReadOnlyList<string> documents, int topN = 0)
    {
        EnsureOpen();
        var request = new Dictionary<string, object?> { ["query"] = query, ["documents"] = documents };
        if (topN > 0) request["top_n"] = topN;
        var doc = Json.Check(Native.TakeString(
            Native.Rerank(_handle, JsonSerializer.Serialize(request, Json.Options))));
        return doc.GetProperty("results").Deserialize<List<RerankHit>>(Json.Options)
               ?? new List<RerankHit>();
    }

    /// <summary>Release the model and its context. Idempotent.</summary>
    public void Dispose()
    {
        if (_handle == IntPtr.Zero) return;
        Native.EmbedDestroy(_handle);
        _handle = IntPtr.Zero;
        GC.SuppressFinalize(this);
    }

    ~Embedder() => Dispose();
}
