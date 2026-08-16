using Modelnexus;

var model = Environment.GetEnvironmentVariable("MODELNEXUS_MODEL")!;
var reranker = Environment.GetEnvironmentVariable("MODELNEXUS_RERANKER")!;

Console.Error.WriteLine($"version : {Modelnexus.Modelnexus.Version()}");
Console.Error.WriteLine($"platform: {Modelnexus.Modelnexus.PlatformKey()}");

using (var chat = new Chat(model))
{
    var r = chat.Infer(new[] { new Message("user", "Reply with exactly: pong") }, maxTokens: 16, seed: 1);
    Console.Error.WriteLine($"infer   : {r.Type} \"{r.Text}\" tokens={r.Usage?.TotalTokens}");

    var pieces = new List<string>();
    chat.Infer(new[] { new Message("user", "Count: 1 2 3") }, maxTokens: 24, seed: 1,
               onToken: p => { pieces.Add(p); return true; });
    Console.Error.WriteLine($"stream  : {pieces.Count} pieces");

    var count = chat.CountTokens(new[] { new Message("user", "Reply with exactly: pong") });
    Console.Error.WriteLine($"count   : {count.Tokens} tokens of {count.NCtx}");

    // Stopping is a result, not an error: a complete response comes back.
    var stopped = chat.Infer(new InferRequest
    {
        Messages = new[] { new Message("user", "Count slowly from one to fifty in words.") },
        MaxTokens = 300, Seed = 7, Temperature = 0.0
    }, onToken: _ => false);
    Console.Error.WriteLine($"cancel  : {stopped.FinishReason} after {stopped.Usage?.CompletionTokens} tokens");

    Console.Error.WriteLine($"loras   : {chat.Loras().Count}");
    try { chat.LoadLora("/nope.gguf"); }
    catch (ModelException e) { Console.Error.WriteLine($"lora err: {e.Code}"); }
}

try { new Chat("/definitely/not/here.gguf"); }
catch (ModelException e) { Console.Error.WriteLine($"load err: {e.Code}"); }

using (var emb = new Embedder(model, Pooling.Mean, nCtx: 512))
{
    var v = emb.Embed(new[] { "hello world", "goodbye world" });
    var norm = Math.Sqrt(v[0].Sum(x => (double)x * x));
    Console.Error.WriteLine($"embed   : {v.Count} x {v[0].Length}, L2={norm:F4}");
    try { emb.Rerank("q", new[] { "a" }); }
    catch (ModelException e) { Console.Error.WriteLine($"rerank g: {e.Code}"); }
}

using (var rr = new Embedder(reranker, Pooling.Rank, nCtx: 512))
{
    var docs = new[] {
        "Berlin is the capital and largest city of Germany.",
        "Paris has been France's capital since the 10th century.",
        "Bananas are a good source of potassium." };
    foreach (var hit in rr.Rerank("What is the capital of France?", docs))
        Console.Error.WriteLine($"  {hit.Score,+8:F3}  [{hit.Index}] {docs[hit.Index][..Math.Min(40, docs[hit.Index].Length)]}");
}
Console.Error.WriteLine("ALL OK");
