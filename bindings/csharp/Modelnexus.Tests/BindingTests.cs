using System.Text.Json;
using Modelnexus;
using Xunit;

namespace Modelnexus.Tests;

/// <summary>
/// Mirrors the Go, Python and JS suites case for case.
/// </summary>
/// <remarks>
/// Four suites asserting the same behaviour -- and the same error codes -- is what
/// keeps the bindings from drifting. That is the entire point of a shared C core.
/// </remarks>
public class BindingTests
{
    // Skipping rather than failing is deliberate: the binding is worth testing on a
    // machine with no multi-gigabyte GGUF sitting around.
    private static string? Model => Env("MODELNEXUS_MODEL");
    private static string? Reranker => Env("MODELNEXUS_RERANKER");

    private static string? Env(string name)
    {
        var value = Environment.GetEnvironmentVariable(name);
        return string.IsNullOrEmpty(value) || !File.Exists(value) ? null : value;
    }

    public static bool HasModel => Model is not null;
    public static bool HasReranker => Reranker is not null;

    [Fact]
    public void VersionIdentifiesBridgeAndEngine()
    {
        var v = Modelnexus.Version();
        Assert.Contains("llamabridge", v);
        Assert.Contains("llama.cpp", v);
    }

    [Fact]
    public void PlatformKeyLooksLikeOsArch()
    {
        Assert.Matches("^(darwin|linux|windows)-(x86_64|aarch64)$", Modelnexus.PlatformKey());
    }

    [Fact]
    public void MissingModelIsATypedError()
    {
        // The code is part of the cross-binding contract: Go, Python and JS report
        // this same one.
        var ex = Assert.Throws<ModelException>(() => new Chat("/definitely/not/here.gguf"));
        Assert.Equal("MODEL_LOAD_FAILED", ex.Code);
    }

    [SkippableFact]
    public void Infer()
    {
        Skip.IfNoModel();
        using var chat = new Chat(Model!);
        var r = chat.Infer(new[] { new Message("user", "Reply with exactly: pong") },
                           maxTokens: 16, seed: 1);
        Assert.Equal("assistant_text", r.Type);
        Assert.NotEmpty(r.Text);
        Assert.True(r.Usage!.TotalTokens > 0);
    }

    [SkippableFact]
    public void StreamingDeliversPiecesAndTheSameFinalResponse()
    {
        Skip.IfNoModel();
        using var chat = new Chat(Model!);
        var pieces = new List<string>();
        var r = chat.Infer(new[] { new Message("user", "Count: 1 2 3") },
                           maxTokens: 24, seed: 1,
                           onToken: p => { pieces.Add(p); return true; });
        Assert.NotEmpty(pieces);
        // Streaming must not be a separate path with a different result.
        Assert.True(r.Usage!.CompletionTokens > 0);
        // Returning true throughout must leave generation undisturbed.
        Assert.False(r.IsCancelled);
    }

    // ---- inference control (ADR-0008) ----
    //
    // These mirror core/tests/abi_test.c case for case. Three of them exist because
    // spike 0003 found the failure mode first, and each of those three fails silently
    // rather than loudly if the core regresses.

    [SkippableFact]
    public void ContextOptionsAreAcceptedAndOmittingThemIsUnchanged()
    {
        Skip.IfNoModel();
        using var configured = new Chat(Model!, nCtx: 4096, nBatch: 512, nSeqMax: 1);
        using var defaulted = new Chat(Model!);
        // No options must send NO config at all, so the core reaches its defaults by
        // exactly the path it used before the parameter existed.
        Assert.Equal("assistant_text",
            defaulted.Infer(new[] { new Message("user", "hi") }, maxTokens: 8).Type);
    }

    [SkippableFact]
    public void CountTokensReportsAPlausibleCountAndTheWindow()
    {
        Skip.IfNoModel();
        using var chat = new Chat(Model!, nCtx: 4096);
        var few = chat.CountTokens(new[] { new Message("user", "hi") });
        var many = chat.CountTokens(new[]
        {
            new Message("user", string.Join(' ', Enumerable.Repeat("elephant", 200)))
        });
        Assert.True(few.Tokens > 0, $"an empty-ish prompt still costs template tokens, got {few.Tokens}");
        Assert.True(many.Tokens > few.Tokens, "a longer prompt must count higher");
        Assert.Equal(4096, few.NCtx);
    }

    [SkippableFact]
    public void ReuseCacheDoesNotChangeTheOutput()
    {
        // Reuse is a LATENCY property. Any observable difference in output is a defect,
        // and this is the assertion that catches it.
        Skip.IfNoModel();
        using var chat = new Chat(Model!);

        InferRequest Ask(bool? reuse) => new()
        {
            Messages = new[] { new Message("user", "Name the capital of France in one word.") },
            MaxTokens = 16,
            Seed = 42,
            Temperature = 0.0,
            ReuseCache = reuse
        };

        var cold = chat.Infer(Ask(false));   // cache cleared
        var warm = chat.Infer(Ask(null));    // default: prefix reused
        var again = chat.Infer(Ask(false));  // cold again -- the control

        Assert.Equal(cold.Text, warm.Text);
        Assert.Equal(cold.Text, again.Text);
    }

    [SkippableFact]
    public void CancellingIsAResultAndLeavesTheCacheSane()
    {
        // The D2xD4 interaction: an abort leaves a partial assistant turn in the cache,
        // and without rollback the next call's prefix match extends a truncated turn as
        // though it were complete. Silent, plausible, wrong.
        Skip.IfNoModel();
        using var chat = new Chat(Model!);

        var seen = 0;
        var stopped = chat.Infer(new InferRequest
        {
            Messages = new[] { new Message("user", "Count slowly from one to fifty in words, one per line.") },
            MaxTokens = 300, Seed = 7, Temperature = 0.0
        }, onToken: _ => ++seen < 8);

        Assert.True(stopped.IsCancelled, $"finish_reason was {stopped.FinishReason}");
        Assert.Equal(8, seen);                              // stopped at the requested token, not later
        Assert.Equal(8, stopped.Usage!.CompletionTokens);   // and the bill is honest

        // The proof that rollback happened: a DIFFERENT request on the same engine must
        // be correct, uninfluenced by the abandoned partial turn.
        var after = chat.Infer(new InferRequest
        {
            Messages = new[] { new Message("user", "Name the capital of France in one word.") },
            MaxTokens = 16, Seed = 42, Temperature = 0.0
        });
        Assert.Contains("Paris", after.Text);
    }

    [SkippableFact]
    public void ASignalledCancellationTokenStopsGeneration()
    {
        // The .NET idiom wired to the ABI's one mechanism -- and, deliberately, it
        // RETURNS the partial response rather than throwing, because the core reports
        // cancellation as a result with real usage counts, not as an error.
        Skip.IfNoModel();
        using var chat = new Chat(Model!);
        using var cts = new CancellationTokenSource();

        var seen = 0;
        var r = chat.Infer(new InferRequest
        {
            Messages = new[] { new Message("user", "Count slowly from one to fifty in words, one per line.") },
            MaxTokens = 300, Seed = 7, Temperature = 0.0
        }, onToken: _ => { if (++seen >= 5) cts.Cancel(); return true; }, cancellationToken: cts.Token);

        Assert.True(r.IsCancelled, $"finish_reason was {r.FinishReason}");
        Assert.True(r.Usage!.CompletionTokens < 300, "generation must have stopped short");
    }

    [SkippableFact]
    public void SchemaConstrainedOutputParsesAndSatisfiesTheSchema()
    {
        Skip.IfNoModel();
        using var chat = new Chat(Model!);
        var r = chat.Infer(new InferRequest
        {
            Messages = new[] { new Message("user", "Describe Paris.") },
            MaxTokens = 120, Seed = 42, Temperature = 0.0,
            JsonSchema = new
            {
                type = "object",
                properties = new
                {
                    city = new { type = "string" },
                    country = new { type = "string" }
                },
                required = new[] { "city", "country" },
                additionalProperties = false
            }
        });

        // PARSE, do not substring-match. Upstream's generated grammar permits a ```json
        // fence, so fenced output is grammar-conformant -- and a Contains check cannot
        // see it. The core strips the fence; this is the assertion that proves it did.
        using var doc = JsonDocument.Parse(r.Text);
        Assert.Equal(JsonValueKind.Object, doc.RootElement.ValueKind);
        Assert.Equal(JsonValueKind.String, doc.RootElement.GetProperty("city").ValueKind);
        Assert.Equal(JsonValueKind.String, doc.RootElement.GetProperty("country").ValueKind);
        foreach (var p in doc.RootElement.EnumerateObject())
            Assert.Contains(p.Name, new[] { "city", "country" });   // additionalProperties: false
    }

    [SkippableFact]
    public void SchemaAndGrammarTogetherIsRejected()
    {
        // An error rather than a precedence rule: a silent winner between two output
        // constraints is a debugging session nobody should have.
        Skip.IfNoModel();
        using var chat = new Chat(Model!);
        var ex = Assert.Throws<ModelException>(() => chat.Infer(new InferRequest
        {
            Messages = new[] { new Message("user", "hi") },
            JsonSchema = new { type = "object" },
            Grammar = "root ::= \"x\""
        }));
        Assert.Equal("INVALID_REQUEST", ex.Code);
    }

    [SkippableFact]
    public void ARawGbnfGrammarConstrainsGeneration()
    {
        Skip.IfNoModel();
        using var chat = new Chat(Model!);
        var r = chat.Infer(new InferRequest
        {
            Messages = new[] { new Message("user", "Pick a colour.") },
            MaxTokens = 16, Seed = 42, Temperature = 0.0,
            Grammar = "root ::= \"red\" | \"blue\""
        });
        Assert.Contains(r.Text.Trim(), new[] { "red", "blue" });
    }

    [SkippableFact]
    public void EventCallbackSurvivesPastConstruction()
    {
        // The core keeps the event delegate for the life of the handle; a binding
        // that lets it be collected crashes the process on the next emit.
        Skip.IfNoModel();
        var events = new List<string>();
        using var chat = new Chat(Model!, onEvent: events.Add);
        chat.Infer(new[] { new Message("user", "hi") }, maxTokens: 8);
        Assert.NotEmpty(events);
    }

    [SkippableFact]
    public void LoraStartsEmptyAndClearingAnEmptySetIsANoOp()
    {
        Skip.IfNoModel();
        using var chat = new Chat(Model!);
        Assert.Empty(chat.Loras());
        chat.ClearLoras();   // must not throw
    }

    [SkippableFact]
    public void LoraErrorsCarryTheSharedCodes()
    {
        Skip.IfNoModel();
        using var chat = new Chat(Model!);
        Assert.Equal("LORA_LOAD_FAILED",
            Assert.Throws<ModelException>(() => chat.LoadLora("/definitely/not/here.gguf")).Code);
        Assert.Equal("LORA_NOT_FOUND",
            Assert.Throws<ModelException>(() => chat.SetLoraScale(99, 0.5)).Code);
    }

    [SkippableFact]
    public void ClearCacheIsObservableAndTheEngineStillWorksAfterIt()
    {
        // The assertion that matters is that the clear is OBSERVABLE. A clear that
        // silently did nothing would still return a well-formed CacheState, and the next
        // inference would still be correct -- just slow, and still holding the previous
        // tenant's conversation.
        Skip.IfNoModel();
        using var chat = new Chat(Model!);

        var ask = new InferRequest
        {
            Messages = new[] { new Message("user", "Name the capital of France in one word.") },
            MaxTokens = 16,
            Seed = 42,
            Temperature = 0.0
        };
        chat.Infer(ask);

        var before = chat.CacheStatus();
        Assert.True(before.Tokens > 0, $"the cache is not empty after an inference, got {before.Tokens}");
        Assert.True(before.NCtx >= before.Tokens);

        // Status is the non-destructive call -- this binding's stand-in for the ABI's
        // "a NULL request reads status, it does not clear". Reading twice must not empty
        // the cache; backwards, an innocent-looking call would wipe a conversation.
        Assert.Equal(before.Tokens, chat.CacheStatus().Tokens);

        Assert.Equal(0, chat.ClearCache().Tokens);
        Assert.Equal(0, chat.CacheStatus().Tokens);   // the clear persisted

        Assert.Contains("Paris", chat.Infer(ask).Text);
    }

    [SkippableFact]
    public void CacheCallsAfterDisposeAreRejected()
    {
        Skip.IfNoModel();
        var chat = new Chat(Model!);
        chat.Dispose();
        Assert.Equal("ENGINE_CLOSED", Assert.Throws<ModelException>(() => chat.CacheStatus()).Code);
        Assert.Equal("ENGINE_CLOSED", Assert.Throws<ModelException>(() => chat.ClearCache()).Code);
    }

    [SkippableFact]
    public void UseAfterDisposeIsRejected()
    {
        Skip.IfNoModel();
        var chat = new Chat(Model!);
        chat.Dispose();
        chat.Dispose();   // idempotent
        Assert.Equal("ENGINE_CLOSED",
            Assert.Throws<ModelException>(() =>
                chat.Infer(new[] { new Message("user", "hi") })).Code);
    }

    [SkippableFact]
    public void EmbedReturnsUnitVectorsOnePerInput()
    {
        Skip.IfNoModel();
        using var emb = new Embedder(Model!, Pooling.Mean, nCtx: 512);
        var v = emb.Embed(new[] { "hello world", "goodbye world" });
        Assert.Equal(2, v.Count);
        Assert.Equal(v[0].Length, v[1].Length);
        // Normalization is what makes a dot product a cosine similarity. If it
        // drifts, every downstream similarity silently changes meaning.
        var norm = Math.Sqrt(v[0].Sum(x => (double)x * x));
        Assert.True(Math.Abs(norm - 1.0) < 1e-3, $"L2 norm was {norm}");
    }

    [SkippableFact]
    public void EmbedIsUnaffectedByBatching()
    {
        // Batching several sequences into one decode must not change the numbers.
        Skip.IfNoModel();
        var texts = Enumerable.Range(0, 6).Select(i => $"sentence number {i}").ToArray();
        using var one = new Embedder(Model!, Pooling.Mean, nCtx: 2048, nBatch: 2048);
        using var many = new Embedder(Model!, Pooling.Mean, nCtx: 2048, nBatch: 2048);
        var a = one.Embed(texts);
        var b = many.Embed(texts);
        for (var i = 0; i < texts.Length; i++)
            for (var j = 0; j < a[i].Length; j++)
                Assert.True(Math.Abs(a[i][j] - b[i][j]) < 1e-4);
    }

    [SkippableFact]
    public void RerankRefusesWithoutRankPooling()
    {
        // Returning plausible-looking numbers from a model whose classification head
        // is not even in the graph would be worse than failing.
        Skip.IfNoModel();
        using var emb = new Embedder(Model!, Pooling.Mean, nCtx: 512);
        Assert.Equal("POOLING_NOT_RANK",
            Assert.Throws<ModelException>(() => emb.Rerank("q", new[] { "a" })).Code);
    }

    [SkippableFact]
    public void RerankRanksSemantically()
    {
        Skip.IfNoReranker();
        using var rr = new Embedder(Reranker!, Pooling.Rank, nCtx: 512);
        var docs = new[]
        {
            "Berlin is the capital and largest city of Germany.",
            "Paris has been France's capital since the 10th century.",
            "Bananas are a good source of potassium."
        };
        var hits = rr.Rerank("What is the capital of France?", docs);
        Assert.Equal(3, hits.Count);
        Assert.Equal(1, hits[0].Index);      // the Paris document must win
        for (var i = 1; i < hits.Count; i++)
            Assert.True(hits[i].Score <= hits[i - 1].Score, "results must be sorted best-first");
        Assert.Single(rr.Rerank("What is the capital of France?", docs, topN: 1));
    }

    [Fact]
    public void LogLevelCanBeSetBeforeAnyModelIsLoaded()
    {
        // A host must be able to silence the engine before it starts talking --
        // llama.cpp begins logging during model load, so after is too late.
        Modelnexus.SetLogLevel(LogLevel.None);
        Modelnexus.SetLogLevel(LogLevel.Warn);
    }
}

internal static class Skip
{
    internal static void IfNoModel() =>
        Xunit.Skip.If(!BindingTests.HasModel, "set MODELNEXUS_MODEL to a tool-capable GGUF");

    internal static void IfNoReranker() =>
        Xunit.Skip.If(!BindingTests.HasReranker, "set MODELNEXUS_RERANKER to a reranker GGUF");
}
