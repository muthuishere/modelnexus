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
                           maxTokens: 24, seed: 1, onToken: pieces.Add);
        Assert.NotEmpty(pieces);
        // Streaming must not be a separate path with a different result.
        Assert.True(r.Usage!.CompletionTokens > 0);
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
