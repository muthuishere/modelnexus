// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightLlmsTxt from 'starlight-llms-txt';

// Project GitHub Pages: https://muthuishere.github.io/modelnexus
export default defineConfig({
	site: 'https://muthuishere.github.io',
	base: '/modelnexus',
	integrations: [
		starlight({
			title: 'modelnexus',
			// No right-hand table of contents. The pages are short enough that a
			// second navigation column is furniture, not help.
			tableOfContents: false,
			description:
				'Run a local LLM inside your process. One C ABI, one thin binding per language — Go, Python, JavaScript. Chat, structured output, embeddings, reranking and LoRA, with no server and no daemon.',
			plugins: [
				starlightLlmsTxt({
					projectName: 'modelnexus',
					description:
						'Local LLM inference exposed as a C ABI, with one thin FFI binding per language (Go/purego, Python/ctypes, JavaScript/koffi). llama.cpp underneath, GGUF models, in-process, no daemon.',
					details:
						'modelnexus gives an application chat inference, JSON-Schema-constrained output, embeddings, reranking and runtime LoRA adapters from a single native library it loads directly — no HTTP, no subprocess, no model server. The KV cache is reused across calls by longest common prefix, so an agent conversation pays a flat cost per turn instead of re-reading itself. Every capability lives in the C ABI and appears in every binding, or it does not ship.',
				}),
			],
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/muthuishere/modelnexus' },
			],
			customCss: ['@fontsource-variable/inter', './src/styles/deemwar.css'],
			sidebar: [
				{
					label: 'Start here',
					items: [
						{ label: 'What it is', slug: 'index' },
						{ label: 'Install', slug: 'install' },
						{ label: 'Quickstart', slug: 'quickstart' },
					],
				},
				{
					label: 'Capabilities',
					items: [
						{ label: 'Structured output', slug: 'structured-output' },
						{ label: 'Streaming and cancellation', slug: 'streaming' },
						{ label: 'The KV cache', slug: 'cache' },
						{ label: 'Counting tokens', slug: 'counting' },
						{ label: 'Embeddings', slug: 'embeddings' },
						{ label: 'Reranking', slug: 'reranking' },
						{ label: 'LoRA adapters', slug: 'lora' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'The C ABI', slug: 'abi' },
						{ label: 'Go', slug: 'api/go' },
						{ label: 'Python', slug: 'api/python' },
						{ label: 'JavaScript', slug: 'api/js' },
					],
				},
				{
					label: 'Understanding it',
					items: [
						{ label: 'Why a C ABI', slug: 'why-c-abi' },
						{ label: 'Choosing a model', slug: 'models' },
						{ label: 'What it is not', slug: 'not' },
					],
				},
			],
		}),
	],
});
