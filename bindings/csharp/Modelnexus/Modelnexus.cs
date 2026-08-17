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
    [property: JsonPropertyName("usage")] Usage? Usage)
{
    /// <summary>True when generation was stopped early rather than finishing.</summary>
    /// <remarks>
    /// A cancelled generation is a normal, complete response carrying the text produced
    /// so far and honest usage counts -- never an exception. This property exists so
    /// that contract is checkable without a magic string at every call site.
    /// </remarks>
    [JsonIgnore]
    public bool IsCancelled => FinishReason == "cancelled";
}

/// <summary>How many tokens a message list occupies, and the window it must fit in.</summary>
public sealed record TokenCount(
    [property: JsonPropertyName("tokens")] int Tokens,
    [property: JsonPropertyName("n_ctx")] int NCtx);

/// <summary>What the engine's KV cache holds, and the window it holds it in.</summary>
/// <remarks>
/// <c>Tokens</c> is the state AFTER the operation that reported it, so a clear always
/// reports zero and a caller can assert rather than assume.
/// </remarks>
public sealed record CacheState(
    [property: JsonPropertyName("tokens")] int Tokens,
    [property: JsonPropertyName("n_ctx")] int NCtx);

/// <summary>One inference request, with every generation parameter the core accepts.</summary>
/// <remarks>
/// Every property is optional and unset properties are omitted from the JSON entirely,
/// so the core's own defaults are what apply -- a binding that materialised them here
/// would freeze today's values into five languages.
/// <see cref="Chat.Infer(object, Func{string, bool}?, CancellationToken)"/> also accepts
/// any other object, so a parameter the core gains before this type does is still reachable.
/// </remarks>
public sealed record InferRequest
{
    [JsonPropertyName("messages")] public required IEnumerable<Message> Messages { get; init; }

    /// <summary>OpenAI-shaped function declarations. modelnexus never executes them.</summary>
    [JsonPropertyName("tools")] public IEnumerable<object>? Tools { get; init; }

    /// <summary>"auto", "none" or "required".</summary>
    [JsonPropertyName("tool_choice")] public string? ToolChoice { get; init; }

    [JsonPropertyName("temperature")] public double? Temperature { get; init; }
    [JsonPropertyName("top_k")] public int? TopK { get; init; }
    [JsonPropertyName("top_p")] public double? TopP { get; init; }
    [JsonPropertyName("min_p")] public double? MinP { get; init; }
    [JsonPropertyName("max_tokens")] public int? MaxTokens { get; init; }
    [JsonPropertyName("repeat_penalty")] public double? RepeatPenalty { get; init; }
    [JsonPropertyName("seed")] public uint? Seed { get; init; }
    [JsonPropertyName("stop")] public IEnumerable<string>? Stop { get; init; }

    /// <summary>
    /// A JSON Schema the output must satisfy -- an anonymous object, a
    /// <see cref="JsonElement"/>, or anything else serializable to a JSON object.
    /// </summary>
    /// <remarks>
    /// The returned text is guaranteed to parse: upstream's generated grammar permits a
    /// ```json markdown fence, and the core strips it before returning. Setting this
    /// together with <see cref="Grammar"/> is an INVALID_REQUEST error, not a precedence
    /// rule -- a silent winner between two output constraints is a debugging session
    /// nobody should have.
    /// </remarks>
    [JsonPropertyName("json_schema")] public object? JsonSchema { get; init; }

    /// <summary>Raw GBNF the output must satisfy. Mutually exclusive with <see cref="JsonSchema"/>.</summary>
    [JsonPropertyName("grammar")] public string? Grammar { get; init; }

    /// <summary>
    /// Whether the engine may reuse the KV cache prefix it already holds. Unset means the
    /// key is not sent at all, so the core's default (reuse) applies.
    /// </summary>
    /// <remarks>
    /// Reuse is purely a latency property -- output is identical either way -- and it turns
    /// an appending conversation's total prefill from quadratic into linear. Set it false
    /// only when each call must be provably independent: a determinism harness, or tenants
    /// sharing one handle.
    /// </remarks>
    [JsonPropertyName("reuse_cache")] public bool? ReuseCache { get; init; }
}

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

/// <summary>How much the inference engine is allowed to say.</summary>
public enum LogLevel
{
    /// <summary>Silence the engine entirely.</summary>
    None = 0,
    Debug = 1,
    Info = 2,
    /// <summary>The default. A library should be quiet unless asked.</summary>
    Warn = 3,
    Error = 4
}

/// <summary>Entry points that need no loaded model.</summary>
public static class Modelnexus
{
    // The core retains the log delegate until it is replaced, so it is held in a
    // static field. A local would be collected while native code still holds the
    // pointer -- the same rule as the engine event callback.
    private static Native.LogCallback? _logCallback;

    /// <summary>
    /// Set how much the engine logs. Defaults to <see cref="LogLevel.Warn"/>.
    /// </summary>
    /// <remarks>
    /// Call before loading a model: llama.cpp starts logging during load, so after
    /// is too late to silence it.
    /// </remarks>
    public static void SetLogLevel(LogLevel level) => Native.SetLogLevel((int)level);

    /// <summary>Route engine log output to a handler instead of stderr. Null restores stderr.</summary>
    public static void SetLogHandler(Action<LogLevel, string>? handler)
    {
        if (handler is null)
        {
            Native.SetLogCallback(null, IntPtr.Zero);
            _logCallback = null;
            return;
        }
        _logCallback = (level, text, _) => handler((LogLevel)level, Native.BorrowString(text));
        Native.SetLogCallback(_logCallback, IntPtr.Zero);
    }

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
    /// <para>
    /// <paramref name="nCtx"/>, <paramref name="nBatch"/> and <paramref name="nSeqMax"/>
    /// are fixed when the context is built and cannot be changed per request, which is why
    /// they live here while every generation parameter lives on <see cref="Infer(object, Func{string, bool}?, CancellationToken)"/>.
    /// Leaving them unset sends no config at all, so the core's defaults are reached by
    /// exactly the path they were before this parameter existed.
    /// </para>
    /// <para>
    /// <paramref name="nSeqMax"/> is reserved and has no observable effect today. It is
    /// accepted now because create-time parameters are the only ones that cannot be added
    /// later through request JSON without breaking the ABI.
    /// </para>
    /// <para>
    /// <paramref name="nGpuLayers"/> is how many model layers are offloaded to the GPU.
    /// Unset means ALL of them, which is the core's default and almost always what you
    /// want. Pass 0 for CPU only — a real setting rather than "unset", for a measurement
    /// that must be reproducible across machines, or to leave the GPU for something else.
    /// </para>
    /// </remarks>
    public Chat(string ggufPath, int? nCtx = null, int? nBatch = null, int? nSeqMax = null,
                int? nGpuLayers = null, Action<string>? onEvent = null)
    {
        _eventCallback = (ptr, _) =>
        {
            var text = Native.BorrowString(ptr);
            _events.Add(text);
            onEvent?.Invoke(text);
        };

        var config = new Dictionary<string, object?>();
        if (nCtx is not null) config["n_ctx"] = nCtx;
        if (nBatch is not null) config["n_batch"] = nBatch;
        if (nSeqMax is not null) config["n_seq_max"] = nSeqMax;
        // `is not null`, not `> 0`: 0 is a legitimate nGpuLayers meaning "CPU only",
        // and folding it into "unset" would silently hand the caller the GPU instead.
        if (nGpuLayers is not null) config["n_gpu_layers"] = nGpuLayers;
        var configJson = config.Count > 0 ? JsonSerializer.Serialize(config, Json.Options) : null;

        _handle = Native.ChatCreate(ggufPath, configJson, _eventCallback, IntPtr.Zero);
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
    /// <param name="onToken">
    /// Pass a handler to stream; the full response still returns. Return <c>true</c> to keep
    /// generating, <c>false</c> to stop after the piece just delivered.
    /// </param>
    /// <param name="cancellationToken">
    /// Signalling it stops generation at the next token boundary, exactly as returning
    /// <c>false</c> from <paramref name="onToken"/> does.
    /// </param>
    /// <returns>
    /// The response. A stopped generation returns NORMALLY with
    /// <see cref="Response.IsCancelled"/> true and the text produced so far -- it does NOT
    /// throw <see cref="OperationCanceledException"/>.
    /// </returns>
    /// <remarks>
    /// That return-don't-throw choice is deliberate, and it is the one place this binding
    /// departs from .NET convention. The core treats cancellation as a <i>result</i>: a
    /// complete response with honest usage counts, because the tokens were really generated
    /// and really cost something. Throwing would discard both the partial text and the bill
    /// for it, and would make a <see cref="CancellationToken"/> behave differently from an
    /// <paramref name="onToken"/> returning false, which is the same mechanism underneath.
    /// Callers who want the exception can still write
    /// <c>cancellationToken.ThrowIfCancellationRequested()</c> on the result.
    /// </remarks>
    public Response Infer(object request, Func<string, bool>? onToken = null,
                          CancellationToken cancellationToken = default)
    {
        EnsureOpen();
        var payload = JsonSerializer.Serialize(request, Json.Options);

        string raw;
        if (onToken is null && !cancellationToken.CanBeCanceled)
        {
            raw = Native.TakeString(Native.ChatInfer(_handle, payload));
        }
        else
        {
            // An exception thrown out of a managed callback and through a native frame is
            // undefined behaviour, so it is caught, turned into a stop, and rethrown once
            // the native call has unwound normally.
            Exception? escaped = null;

            // Only needs to survive the call, but keeping a local reference is what
            // guarantees that -- the GC does not know native code holds it.
            Native.TokenCallback cb = (ptr, _) =>
            {
                if (cancellationToken.IsCancellationRequested) return 1;
                if (onToken is null) return 0;
                try { return onToken(Native.BorrowString(ptr)) ? 0 : 1; }
                catch (Exception e) { escaped = e; return 1; }
            };
            raw = Native.TakeString(Native.ChatInferStream(_handle, payload, cb, IntPtr.Zero));
            GC.KeepAlive(cb);

            if (escaped is not null) throw escaped;
        }

        var doc = Json.Check(raw);
        return doc.Deserialize<Response>(Json.Options)
               ?? throw new ModelException("BAD_RESPONSE", "could not deserialize the response");
    }

    /// <summary>Convenience overload for a plain message list.</summary>
    public Response Infer(IEnumerable<Message> messages, int? maxTokens = null, uint? seed = null,
                          Func<string, bool>? onToken = null,
                          CancellationToken cancellationToken = default) =>
        Infer(new InferRequest { Messages = messages, MaxTokens = maxTokens, Seed = seed },
              onToken, cancellationToken);

    /// <summary>Count the tokens a request occupies, without generating anything.</summary>
    /// <remarks>
    /// Applies the chat template and tokenizes; it creates no context, decodes nothing, and
    /// does not touch the KV cache, so calling it between two inferences cannot disturb
    /// prefix reuse. It lives in the ABI rather than here because counting needs BOTH the
    /// model's vocabulary and its parsed chat template, and a binding holds neither.
    /// </remarks>
    public TokenCount CountTokens(object request)
    {
        EnsureOpen();
        var payload = JsonSerializer.Serialize(request, Json.Options);
        var doc = Json.Check(Native.TakeString(Native.CountTokens(_handle, payload)));
        return doc.Deserialize<TokenCount>(Json.Options)
               ?? throw new ModelException("BAD_RESPONSE", "could not deserialize the token count");
    }

    /// <summary>Convenience overload for a plain message list.</summary>
    public TokenCount CountTokens(IEnumerable<Message> messages) =>
        CountTokens(new InferRequest { Messages = messages });

    // ---- KV cache ----

    private CacheState Cache(string op)
    {
        EnsureOpen();
        var payload = JsonSerializer.Serialize(new { op }, Json.Options);
        var doc = Json.Check(Native.TakeString(Native.ChatCache(_handle, payload)));
        return doc.Deserialize<CacheState>(Json.Options)
               ?? throw new ModelException("BAD_RESPONSE", "could not deserialize the cache state");
    }

    /// <summary>What the engine's KV cache currently holds. Changes nothing.</summary>
    public CacheState CacheStatus() => Cache("status");

    /// <summary>
    /// Drop the KV cache, freeing its memory and forgetting the sequence. Returns the state
    /// afterwards -- always zero tokens, so a caller can assert the release happened.
    /// </summary>
    /// <remarks>
    /// Prefix reuse is right for a conversation that appends and wrong when a chat moves to
    /// unrelated work: the old conversation keeps occupying context memory, and two tenants
    /// sharing a handle would share a cache. Setting <c>ReuseCache = false</c> on the next
    /// inference also clears, but only as a side effect of doing work -- no help when the
    /// point is to release memory now, or to prove the cache is empty before handing the
    /// handle on.
    /// </remarks>
    public CacheState ClearCache() => Cache("clear");

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

    /// <summary>Load a GGUF model for embedding or reranking.</summary>
    /// <param name="ggufPath">Path to the GGUF model.</param>
    /// <param name="pooling">
    /// How token vectors are reduced to one vector per input. <see cref="Pooling.Rank"/>
    /// is required for <see cref="Rerank"/>; <see cref="Pooling.Default"/> leaves the
    /// choice to the model.
    /// </param>
    /// <param name="nCtx">Context size; 0 leaves it to the core's default.</param>
    /// <param name="nBatch">
    /// Caps how many tokens one input may have; 0 leaves it to the core's default.
    /// The binding states no number of its own -- a default restated here is a second
    /// place it can drift from.
    /// </param>
    /// <param name="nGpuLayers">
    /// How many model layers are offloaded to the GPU. Unset means ALL of them, the
    /// core's default. 0 is CPU only — a deliberate setting, not "unset", which is why
    /// this is nullable while <paramref name="nCtx"/> is not.
    /// </param>
    /// <param name="onEvent">Receives the core's lifecycle events during load.</param>
    public Embedder(string ggufPath, Pooling pooling = Pooling.Default,
                    int nCtx = 0, int nBatch = 0, int? nGpuLayers = null,
                    Action<string>? onEvent = null)
    {
        _eventCallback = (ptr, _) =>
        {
            var text = Native.BorrowString(ptr);
            _events.Add(text);
            onEvent?.Invoke(text);
        };

        var config = new Dictionary<string, object?>();
        if (pooling != Pooling.Default) config["pooling"] = pooling.ToString().ToLowerInvariant();
        if (nCtx > 0) config["n_ctx"] = nCtx;
        if (nBatch > 0) config["n_batch"] = nBatch;
        // `is not null` for the same reason as Chat: 0 means CPU only, not unset.
        if (nGpuLayers is not null) config["n_gpu_layers"] = nGpuLayers;
        // Nothing set means NULL, not "{}" -- same rule as Chat above.
        var configJson = config.Count > 0 ? JsonSerializer.Serialize(config, Json.Options) : null;

        _handle = Native.EmbedCreate(ggufPath, configJson, _eventCallback, IntPtr.Zero);
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
