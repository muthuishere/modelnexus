// hello -- load a GGUF, run one inference, print the answer.
//
// The smallest thing that works. There is no server to start, no port to pick and no
// subprocess: the model is mapped into this Node process and decodes in it.
//
//   MODELNEXUS_MODEL=/path/to/model.gguf node hello.js

import { Chat, setLogLevel, LogLevel } from '@muthuishere/modelnexus';

const path = process.env.MODELNEXUS_MODEL;
if (!path) {
  console.error('MODELNEXUS_MODEL is not set — point it at a tool-capable GGUF');
  process.exit(1);
}

// llama.cpp narrates the load. The bridge already owns that sink and defaults to WARN;
// dropping to ERROR leaves this program's own output as the only output. Set it before
// loading — logging starts during the load, so afterwards is too late.
setLogLevel(LogLevel.ERROR);

// The constructor loads the weights and builds the inference context. It rejects a
// model whose chat template cannot do tool calling, rather than loading it and
// degrading tool calls to prose later — the failure arrives here, where it is cheap to
// read.
const chat = new Chat(path);
try {
  // Generation parameters travel inside the request object under the core's own names,
  // so a new one in the core needs no change in this binding.
  const reply = chat.infer({
    messages: [{ role: 'user', content: 'Name the capital of France. Answer in one word.' }],
    temperature: 0.0,
    seed: 42,
    max_tokens: 64,
  });

  console.log('answer:', reply.text);
  console.log(
    `tokens: ${reply.usage.prompt_tokens} prompt + ${reply.usage.completion_tokens} completion, ` +
      `finish_reason=${reply.finish_reason}`,
  );
} finally {
  // The handle owns native memory. Nothing here is reclaimed by the garbage collector.
  chat.close();
}
