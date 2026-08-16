// conversation -- a multi-turn loop, with per-turn wall clock printed.
//
// Every turn resends the whole conversation, so the prompt grows without bound. The
// engine keeps what is already in its KV cache and re-decodes only the part that
// differs, which for an appending conversation is just the new turn. The cost of
// re-reading the prefix therefore stops growing.
//
// This example does not assert a speedup. It runs the SAME turns twice — once with
// reuse on (the default) and once with reuseCache: false — and prints both clocks so
// the reader sees whatever this machine actually does.
//
//   MODELNEXUS_MODEL=/path/to/model.gguf node conversation.js

import { Chat, setLogLevel, LogLevel } from '@muthuishere/modelnexus';

const path = process.env.MODELNEXUS_MODEL;
if (!path) {
  console.error('MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF');
  process.exit(1);
}

setLogLevel(LogLevel.ERROR);

// Scripted turns, so both runs do identical work and the only variable is the cache.
const turns = [
  'I am planning a week in Lisbon. Give me one neighbourhood to stay in.',
  'What is one dish I should eat there?',
  'Name one day trip within two hours.',
  'What is the weather like in October?',
  'One phrase of Portuguese I should learn?',
  'Is the metro worth using?',
  'One museum worth an afternoon?',
  'Sum up the trip in one sentence.',
];

// A system prompt long enough that the reused prefix is worth something from turn one.
const system =
  'You are a concise travel assistant. Answer in at most two short sentences. ' +
  'Never list more than one option. Do not repeat the question back. Do not add caveats ' +
  'about checking current information. Assume the traveller is an experienced adult who ' +
  'has been to Europe before and wants opinions, not disclaimers.';

/** One full conversation. Returns per-turn milliseconds and prompt sizes. */
function run(reuse) {
  const elapsed = [];
  const prompts = [];
  const chat = new Chat(path);
  try {
    const messages = [{ role: 'system', content: system }];
    for (const question of turns) {
      messages.push({ role: 'user', content: question });

      const start = process.hrtime.bigint();
      const reply = chat.infer({
        messages,
        // reuseCache is only sent when set, because the core's default is true and
        // writing `false` for an absent option would opt every caller out. This example
        // states it explicitly on both sides.
        reuseCache: reuse,
        temperature: 0.0,
        seed: 11,
        max_tokens: 40,
      });
      elapsed.push(Number(process.hrtime.bigint() - start) / 1e6);
      prompts.push(reply.usage.prompt_tokens);

      // Appending the reply is what makes the next prompt a strict extension of this
      // one — exactly the shape prefix reuse can exploit.
      messages.push({ role: 'assistant', content: reply.text });
    }
  } finally {
    chat.close();
  }
  return { elapsed, prompts };
}

// Each run gets a fresh engine so neither inherits the other's cache.
const withReuse = run(true);
const withoutReuse = run(false);

const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const pad = (s, n) => String(s).padStart(n);

console.log();
console.log('turn  prompt tokens   reuse on   reuse off');
for (let i = 0; i < turns.length; i++) {
  console.log(
    `${pad(i + 1, 3)} ${pad(withReuse.prompts[i], 12)} ` +
      `${pad(withReuse.elapsed[i].toFixed(0), 8)} ms ${pad(withoutReuse.elapsed[i].toFixed(0), 8)} ms`,
  );
}
console.log(
  `total ${pad(sum(withReuse.elapsed).toFixed(0), 19)} ms ${pad(sum(withoutReuse.elapsed).toFixed(0), 8)} ms`,
);
console.log();

// Wall clock, not a claim: these numbers are whatever this machine produced just now.
// Each turn generates the same 40 tokens, so the difference between the columns is the
// prefill the reuse run did not have to redo.
console.log(
  `prompt grew from ${withReuse.prompts[0]} to ${withReuse.prompts[turns.length - 1]} ` +
    `tokens across ${turns.length} turns`,
);
