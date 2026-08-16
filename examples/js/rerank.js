// rerank -- score documents against a query with a reranker model, best first.
//
// A reranker reads the query and the document TOGETHER and emits one relevance logit,
// which is why it beats comparing two independently-computed embeddings — and why it
// costs a forward pass per document. The usual shape is: retrieve 50 by embedding,
// rerank them, keep 5.
//
// This needs a reranker GGUF, not a chat model, opened with pooling 'rank': that is
// what attaches the model's classification head to the graph. Without it the call fails
// with POOLING_NOT_RANK rather than returning numbers that look like scores.
//
//   MODELNEXUS_RERANKER=/path/to/bge-reranker.gguf node rerank.js

import { Embedder, setLogLevel, LogLevel } from '@muthuishere/modelnexus';

const path = process.env.MODELNEXUS_RERANKER;
if (!path) {
  console.error('MODELNEXUS_RERANKER is not set — point it at a reranker GGUF');
  process.exit(1);
}

setLogLevel(LogLevel.ERROR);

const query = 'How do I stop my sourdough loaf from being dense?';

const documents = [
  'Sourdough stays dense when the starter is underactive: feed it twice daily until it doubles in four hours.',
  'The Rialto Bridge in Venice was completed in 1591 and spans the Grand Canal.',
  'Under-proofed dough has not built enough gas; extend the bulk ferment until the dough rises by half.',
  'Store flour in an airtight container away from sunlight to keep weevils out.',
  'Over-handling during shaping degasses the crumb, giving a tight, heavy loaf.',
];

const pad = (s, n) => String(s).padStart(n);

const rr = new Embedder(path, { pooling: 'rank' });
let hits;
try {
  // Omitting topN returns every document. Results arrive sorted best-first.
  hits = rr.rerank(query, documents);
} finally {
  rr.close();
}

console.log(`query: ${query}\n`);
console.log('rank   score   original   document');
hits.forEach((hit, i) => {
  // `index` is the document's position in the ORIGINAL array. The list came back
  // reordered, so this is the only way to map a score onto your own data.
  console.log(
    `${pad(i + 1, 4)} ${pad(hit.score.toFixed(3), 8)} ${pad(hit.index, 10)}   ${documents[hit.index]}`,
  );
});

console.log();
console.log('Scores are raw model logits: comparable inside this one call, not across');
console.log('models, and not probabilities.');
