// embeddings -- turn sentences into vectors, then compare them.
//
// Vectors come back L2-normalised, so a dot product IS the cosine similarity. There is
// no norm to divide by and no library to install for it: the one-liner below is the
// whole of the maths.
//
// An Embedder is a separate handle from a Chat on purpose — embedding needs a context
// built with embeddings enabled and a pooling strategy fixed at creation, neither of
// which can be switched on a generation context afterwards.
//
//   MODELNEXUS_MODEL=/path/to/model.gguf node embeddings.js

import { Embedder, setLogLevel, LogLevel } from '@muthuishere/modelnexus';

const path = process.env.MODELNEXUS_MODEL;
if (!path) {
  console.error('MODELNEXUS_MODEL is not set — point it at a GGUF');
  process.exit(1);
}

setLogLevel(LogLevel.ERROR);

const sentences = [
  'The cat sat on the mat.',
  'A kitten rested on the rug.',
  'Interest rates rose again this quarter.',
  'The central bank raised borrowing costs.',
];

/** A plain dot product, valid only because embed() normalises. Pass
 *  { normalize: false } and you owe yourself the division. */
const cosine = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);

const pad = (s, n) => String(s).padStart(n);

// Mean pooling averages the token vectors — the usual choice for sentence similarity.
// A dedicated embedding model would be smaller and better; a chat model works and keeps
// this example to one model file.
const emb = new Embedder(path, { pooling: 'mean' });
let vectors;
try {
  // One call, several inputs: they are batched into as few decodes as nBatch allows,
  // and come back in input order.
  vectors = emb.embed(sentences);
} finally {
  emb.close();
}

console.log(`${vectors.length} vectors of ${vectors[0].length} dimensions\n`);
sentences.forEach((s, i) => console.log(`  [${i}] ${s}`));
console.log();

console.log('cosine   ' + sentences.map((_, i) => pad(i, 8)).join(''));
sentences.forEach((_, i) => {
  const row = sentences.map((__, j) => pad(cosine(vectors[i], vectors[j]).toFixed(3), 8)).join('');
  console.log(`${pad(i, 6)}   ${row}`);
});

console.log();
console.log(`cats  [0]x[1] = ${cosine(vectors[0], vectors[1]).toFixed(3)}`);
console.log(`rates [2]x[3] = ${cosine(vectors[2], vectors[3]).toFixed(3)}`);
console.log(`across topics [0]x[2] = ${cosine(vectors[0], vectors[2]).toFixed(3)}`);
