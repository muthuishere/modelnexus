// counting -- how big is this conversation, before you commit to sending it?
//
// countTokens applies the model's chat template and tokenizes. It creates no context,
// decodes nothing and does not touch the KV cache, so it is safe between inferences. It
// lives in the ABI because counting needs the model's vocabulary AND its parsed chat
// template, and no binding holds either — a tokenizer bolted on in JS would be a
// different tokenizer.
//
//   MODELNEXUS_MODEL=/path/to/model.gguf node counting.js

import { Chat, setLogLevel, LogLevel } from '@muthuishere/modelnexus';

const path = process.env.MODELNEXUS_MODEL;
if (!path) {
  console.error('MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF');
  process.exit(1);
}

setLogLevel(LogLevel.ERROR);

// Tools are part of the prompt too — the chat template renders them — so counting
// without them under-reports a tool-calling request.
const tools = [
  {
    type: 'function',
    function: {
      name: 'lookup_part',
      description: 'Find a spare part by model and component name',
      parameters: {
        type: 'object',
        properties: { model: { type: 'string' }, component: { type: 'string' } },
        required: ['model', 'component'],
      },
    },
  },
];

const pad = (s, n) => String(s).padStart(n);

// A deliberately small window, so the budget is something you can watch fill up.
const chat = new Chat(path, { nCtx: 2048 });
try {
  const messages = [
    { role: 'system', content: 'You are a support agent for a bicycle shop.' },
    { role: 'user', content: 'My rear derailleur skips under load in the two lowest gears.' },
  ];

  // Grow the conversation and watch the count against the window. This is the loop a
  // real agent runs before every call, to decide whether to trim history.
  console.log('messages   tokens   n_ctx   used');
  for (let i = 0; i < 5; i++) {
    const { tokens, nCtx } = chat.countTokens({ messages });
    console.log(
      `${pad(messages.length, 8)} ${pad(tokens, 8)} ${pad(nCtx, 7)} ` +
        `${pad(((100 * tokens) / nCtx).toFixed(1), 5)}%`,
    );
    messages.push(
      {
        role: 'assistant',
        content:
          'Check the cable tension at the barrel adjuster and index the shifter again. '.repeat(6),
      },
      { role: 'user', content: 'That did not fix it. What else?' },
    );
  }

  const bare = chat.countTokens({ messages: messages.slice(0, 2) });
  const withTools = chat.countTokens({ messages: messages.slice(0, 2), tools });

  console.log();
  console.log(
    `the same two messages cost ${bare.tokens} tokens, or ${withTools.tokens} once one ` +
      `tool declaration is attached (+${withTools.tokens - bare.tokens})`,
  );
} finally {
  chat.close();
}
