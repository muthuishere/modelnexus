// structured -- pass a JSON Schema, get output that is guaranteed to parse.
//
// A schema is compiled into a grammar that constrains decoding, so the model cannot
// emit a token that would break the shape. The usual small-model failure — your JSON
// plus an apology, or a truncated object — becomes impossible rather than unlikely.
//
//   MODELNEXUS_MODEL=/path/to/model.gguf node structured.js

import { Chat, setLogLevel, LogLevel } from '@muthuishere/modelnexus';

const path = process.env.MODELNEXUS_MODEL;
if (!path) {
  console.error('MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF');
  process.exit(1);
}

setLogLevel(LogLevel.ERROR);

// `required` and `enum` are worth setting: they are constraints the grammar can
// enforce, which makes them free — unlike a prompt instruction the model may ignore.
const schema = {
  type: 'object',
  properties: {
    sentiment: { type: 'string', enum: ['positive', 'negative', 'mixed'] },
    rating: { type: 'integer', minimum: 1, maximum: 5 },
    topics: { type: 'array', items: { type: 'string' } },
  },
  required: ['sentiment', 'rating', 'topics'],
  additionalProperties: false,
};

const review =
  'Classify this review: "The battery lasts two days and the screen is gorgeous, ' +
  'but the camera is mediocre and it costs too much."';

const chat = new Chat(path);
let reply;
try {
  reply = chat.infer({
    messages: [{ role: 'user', content: review }],
    // camelCase here, json_schema on the wire: the binding renames the fields it names
    // and passes everything else through untouched.
    jsonSchema: schema,
    temperature: 0.0,
    seed: 7,
    max_tokens: 120,
  });
} finally {
  chat.close();
}

// No repair pass, no fence stripping, no retry loop. The core already removed the
// ```json fence llama.cpp's generated grammar permits, so what arrives is JSON.
let parsed;
try {
  parsed = JSON.parse(reply.text);
} catch (err) {
  console.error('the schema did not hold:', err.message);
  console.error('raw:', reply.text);
  process.exit(1);
}

console.log('sentiment:', parsed.sentiment);
console.log('rating:   ', parsed.rating);
console.log('topics:   ', parsed.topics);
console.log(
  `(parsed from ${reply.text.length} bytes of model output, finish_reason=${reply.finish_reason})`,
);
