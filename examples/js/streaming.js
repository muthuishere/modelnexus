// streaming -- print tokens as they arrive, and stop early from the callback.
//
// Stopping is the point. Before 0.2.0 a consumer who walked away — a closed stream, a
// user pressing stop — could not tell the model, so it generated to completion and you
// paid for all of it. Returning false from onToken ends the turn now.
//
//   MODELNEXUS_MODEL=/path/to/model.gguf node streaming.js

import { Chat, setLogLevel, LogLevel } from '@muthuishere/modelnexus';

const path = process.env.MODELNEXUS_MODEL;
if (!path) {
  console.error('MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF');
  process.exit(1);
}

// Quieten the engine so the streamed tokens are the only thing on the terminal.
setLogLevel(LogLevel.ERROR);

const BUDGET = 20;
const maxTokens = 200; // deliberately more than we intend to read

const chat = new Chat(path);
let seen = 0;
let reply;
try {
  process.stdout.write('streaming: ');
  reply = chat.infer(
    {
      messages: [{ role: 'user', content: 'List the planets of the solar system, one per line.' }],
      temperature: 0.0,
      seed: 42,
      maxTokens: maxTokens,
    },
    (piece) => {
      process.stdout.write(piece);
      seen += 1;
      // Only an explicit false stops. The usual callback — one that writes a piece and
      // returns nothing — yields undefined, and treating that falsy value as "stop"
      // would cancel every stream after its first token.
      return seen < BUDGET;
    },
  );
  process.stdout.write('\n');
} finally {
  chat.close();
}

// A cancelled generation is a RESULT, not an error: the response is complete, the text
// is what was really produced, and the usage counts are the tokens you were really
// charged for. Nothing threw above, precisely because nothing went wrong.
console.log();
console.log('finishReason:', reply.finishReason);
console.log('cancelled:    ', reply.finishReason === 'cancelled');
console.log(
  `usage:         ${reply.usage.promptTokens} prompt + ${reply.usage.completionTokens} ` +
    `completion (asked for up to ${maxTokens})`,
);
console.log(`pieces seen:   ${seen}, response text length: ${reply.text.length} bytes`);

if (reply.finishReason !== 'cancelled') {
  console.error('expected the callback to have stopped generation');
  process.exit(1);
}
