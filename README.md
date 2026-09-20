> ## This is a fork of [ollama/ollama](https://github.com/ollama/ollama)
>
> **Everything below this banner is upstream's README, describing upstream's
> work.** Ollama is theirs; this fork only adds to it, and tries to add as
> little as possible.
>
> It tracks upstream releases and carries a narrow set of changes:
>
> - **Vision-model correctness.** Image token budgets and native-aspect
>   dynamic resolution for gemma4 and nemotron, and an fp32-accumulation gate
>   for qwen2.5-vl that closes an fp16 overflow in the vision tower — a few
>   elements in millions reach `inf` on CUDA and the caption collapses into a
>   repeated glyph. Offered upstream as
>   [ollama#18070](https://github.com/ollama/ollama/pull/18070).
> - **A vision regression suite.** A preflight harness with recorded
>   per-model expectations, generated (public, no-download) trigger images,
>   and env-gated node-level instrumentation, run against every build before
>   it deploys.
> - **Fixes carried until upstream takes them**, each tracked against an
>   upstream issue or PR, and deleted from here when it lands there — the
>   [retirement register](docs/maxusai/retirement-register.md) lists every
>   carried item, what retires it, and the test that gates its deletion.
> - An experimental MLX runtime for Apple Silicon and CUDA — see the caveats
>   below before using it for anything that matters.
>
> **Current fold:** [`v0.34.1-dynres`](https://github.com/MaxusAI/ollama/releases/tag/v0.34.1-dynres)
> — upstream v0.34.1, llama.cpp `b10864`, MLX `d9add9d1`. `main` moves ahead of this between
> folds; the tag is the fixed point to build and roll back to.
> **Deployed:** `main` at `16649e8`, stamped `0.34.1-dynres-16-g16649e8`, on the CUDA host since
> 2026-09-18 22:30 — the tag's native payload with a Go-only rebuild that adds ADR 0036 (a gemma4
> vision runner starts from the batch rung holding its image ceiling), with
> `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` on the container (ADR 0033; see the fold's memory finding in
> [`docs/maxusai/tasks/upstream-sync-0.34.1.md`](docs/maxusai/tasks/upstream-sync-0.34.1.md)).
> The matrix below is the tag's full preflight run; the rebuild changes no native input, and its
> verification on production is in the task doc's deploy section.
> On the Apple Silicon host the same commit has served the mlx-metal surface on `:11435` since
> 2026-09-18, stamped `0.34.0-maxusai-8a7ba949` — the same build as `0.34.1-dynres-0-g8a7ba94` (ADR 0032,
> 2026-09-19 amendment) — and was promoted on 2026-09-19 with the MLX #3912 kernel fix kept
> ([ADR 0037](docs/maxusai/adr/0037-keep-the-mlx-3912-kernel-fix.md)), with
> `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` in its launchd environment as on the CUDA container.
> **The AMD/gfx1151 host joined on 2026-09-19 18:55**, stamped `0.34.1-dynres-16649e8c` from a
> full `FLAVOR=rocm` build of the same commit, when the
> [upgrade gate](docs/maxusai/amd-upgrade-gate.md#decision-2026-09-19--the-gate-lifts-on-evidence)
> lifted — it had held that host on 0.32.1 since 2026-07-31, and all three platforms now serve
> one commit. That build **must** carry `llama/compat/906-revert-hip-integrated-flag.patch`:
> llama.cpp b10864 misses upstream's HIP revert by 78 minutes, and without it vision output on
> gfx1151 is silently wrong — no crash, no warning, unchanged token counts.

> Fork builds are stamped `<upstream-version>-dynres-<n>-g<sha>`; `dynres`
> names the change that started the fork, not the company that runs it.
> Fork-specific documentation, ADRs and measurements live in
> [`docs/maxusai/`](docs/maxusai/).

### What tested green for this fold

<!-- GENERATED — do not hand-edit. Regenerate on each fold from the release's
     full preflight run:
       python3 docs/maxusai/vision-suite/preflight/release_matrix.py \
           --version <fold-version> docs/maxusai/vision-suite/preflight/runs/<full-run>.json
     Feed it the FULL run only: the generator takes the newest run per surface,
     so a later smoke (which deliberately skips probes) would overwrite green
     cells with "skipped". The release notes carry the same generated matrix.

     For v0.34.1 the two surfaces carry equivalent stamps of one build (ADR 0032,
     2026-09-19 amendment), so name both:
       release_matrix.py --version 0.34.1-dynres \
           --version 0.34.0-maxusai-8a7ba949 runs/*.json
     The mlx-metal run is committed (runs/preflight-mlx-metal-0340-8a7ba949.json);
     the CUDA host's post-deploy run is not, so today the two rows are each
     generated on their own host and pasted verbatim. Committing the CUDA run
     makes the command above regenerate both.

     The table below is the verbatim generator output for the v0.34.1 fold and
     is NOT hand-edited to match a later generator. release_matrix.py has since
     gained an "M5 tensor path" column (the three metal_tensor_* checks); both
     runs here predate those checks, so it regenerates as "not run" on every
     row, and the column appears at the next fold's regeneration. -->

| surface | Build identity | Image size ladder | Pinned image budget | thinking on/off | Output quality | fp16 overflow canary | Runner isolation | measured on |
|---|---|---|---|---|---|---|---|---|
| **cuda** | green | green | green | green | skipped | green | green | `0.34.1-dynres-0-g8a7ba94` |
| **mlx-cuda** | not run | not run | not run | not run | not run | not run | not run | — |
| **mlx-metal** | skipped | green | skipped | green | not run | skipped | green | `0.34.0-maxusai-8a7ba949` |
| **apple-silicon-mlx** | not run | not run | not run | not run | not run | not run | not run | — |
| **rocm** | not run | not run | not run | not run | not run | not run | not run | — |
| **cpu** | not run | not run | not run | not run | not run | not run | not run | — |

Generated by `release_matrix.py` from recorded preflight runs. A surface with no run for this release reads *not run* — absence is shown, never assumed green. A group is reported at its weakest check, so one skipped probe does not read as a pass.

`cuda` here is the deployed serving surface. `apple-silicon-mlx` is the
deprecated alias for `mlx-metal` and will disappear from the generator with it.


### What differs from upstream, concretely

Measured against upstream ollama v0.34.1 at llama.cpp `b10864`. Every row is a
capability the fork has and upstream does not; the record column is where the
decision and its measurements live (`docs/maxusai/`).

**Vision correctness on the llama.cpp path — the deployed engine**

| | upstream ollama | this fork | record |
|---|---|---|---|
| **nemotron-3 vision** | fixed 512×512 canvas — **256 tokens per image**, whatever the aspect ratio | native-aspect dynamic resolution, **256–3,328 tokens**, position embeddings interpolated to the patch grid in-graph | patch `002`, ADR 0001 |
| **gemma4 image budget** | default limits **70–1,120 tokens** (40–280 before b10864); an under-budget image keeps its natural rounded grid and is letterbox-padded | every image scaled to *fill* the requested budget and snapped to gemma4's supported ladder (70/140/280/560/1120), never padded — off-ladder grids measurably break `box_2d` vertical grounding. The budget is a per-request option (`image_min_tokens`/`image_max_tokens`, defaults 70/1120) and the scheduler reloads when the resolved flags change. Upstream has since adopted the same default limits; the fill is still fork-only | patch `004`, ADR 0003/0008/0016 |
| **qwen2.5-vl on CUDA** | f16 vision matmuls accumulate in fp16; on some ordinary images a few elements of millions reach `inf` at `v.blk.31.ffn_down` and the caption collapses into one repeated glyph | fp32 accumulation forced for every `qwen25vl` runner, keyed on the GGUF architecture. Offered upstream as [ollama#18070](https://github.com/ollama/ollama/pull/18070) | `llm/llama_server.go` |
| **MoE + MMQ on CUDA** | ids-path tail padding sized from `ne11`; under broadcast `ne11 == 1`, so the buffer gets no padding and the kernel overruns by up to a 512-row tile | padding sized from the flattened row count. Reported as [llama.cpp#27044](https://github.com/ggml-org/llama.cpp/issues/27044) | patch `903` |
| **transparent images** | pixels as decoded | composited over white before the resize, matching the mlx-vlm reference | ADR 0015 |

**Structured output and generation control**

| | upstream ollama | this fork | record |
|---|---|---|---|
| **`think` + `format` in one request** | defers the grammar until the thinking→content transition and folds pass-one metrics into the final response | the same, plus: a model with a known think-close marker stops pass one exactly there and continues textually, so runaway thinking cannot burn the budget; pass-one metrics are reconstructed when a runner does not report them; the second pass is pinned to pass one's truncation window | ADR 0002/0004/0010 |
| **drafting under a grammar (MLX)** | always on | on by default to match upstream; `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` restores the gate | ADR 0033 |
| **stop sequences (MLX)** | not honoured by the MLX runner | honoured, with a possible stop prefix held back until it matches or the stream ends | `mlxrunner/stopper.go` |
| **KV cache type** | one global `OLLAMA_KV_CACHE_TYPE` | per model, with K/V pair syntax and a policy for reasoning models | ADR 0005 |

**Serving and scheduling**

| | upstream ollama | this fork | record |
|---|---|---|---|
| **MLX admission** | weights against free device memory (and, since v0.34.1, a system-memory bound on integrated GPUs) | weights + KV priced at the requested `num_ctx` + a per-architecture headroom; an explicit rung that does not fit is refused, an automatic one is clamped | ADR 0034 |
| **MLX memory ceiling** | none | `OLLAMA_MLX_MEMORY_LIMIT` and a cache limit, set per runner from the admitted budget | runner knobs |
| **nvfp4 global scales** | — | held in MLX's `m × 2688` representation and divided back out by every wrapper that applies the scale itself, which is not the identity in float32 (17 of 31b's 191 vision scales move one ulp) | ADR 0039 (proposed) |
| **model identity in a record** | a tag | the manifest digest: the library re-published `gemma4:*-nvfp4` with bf16 vision towers under unchanged tags and config blobs | ADR 0038 (proposed) |
| **gemma4 image chunk vs. generation batch (GGUF)** | the batch follows `num_ctx` (1024 above 4096), so a top-rung gemma4 image (up to 1120 tokens) is decoded in two pieces, bidirectional only within each | a gemma4 vision runner starts from the batch rung that holds its image ceiling (2048 at 1120) and steps down only when it does not fit | ADR 0036 |
| **gemma4 on MLX** | upstream's own vision and audio tower with a fixed per-checkpoint soft-token set, no per-request budget | vision through upstream's `MediaModel` with a per-request budget seam; audio not shipped | ADR 0021 |
| **media prompts on MLX** | — | prefill chunks span-aligned around image blocks; a late image is refused | ADR 0014 |
| **scheduler** | — | log sites never drop fields under contention; head-of-line and evict-all-wait fixes; attached media charged against capabilities before the load; capability advertising corrected for MLX architectures | `server/sched.go`, `images.go` |
| **panic hygiene (MLX)** | — | a cleanup that fails while a request is unwinding never replaces the panic that caused it | `mlxrunner/unwind.go` |

**Measurement — nothing comparable upstream**

| | this fork | record |
|---|---|---|
| **vision regression suite** | preflight with versioned per-model expectations, generated (public, no-download) trigger images, an env-gated node-level meter, and a generated release matrix — run before every deploy | ADR 0011/0012, patch `801` |
| **campaigns** | five report templates rendered only by generators; per-request memory and drafting analysis; bbox conformance scoped to image geometry | ADR 0012/0028/0030 |
| **bounding-box protocol** | requests pin **norm-1000** and carry a self-calibrating anchor, so the model's internal resize cannot contaminate coordinates: **111 of 112** cells convert cleanly across 14 geometries × 4 models × 2 think modes. A protocol and its measurements, not a runtime change | ADR 0027/0030 |
| **fork identity** | builds stamped `<upstream>-dynres-<n>-g<sha>`, a tag per fold, release notes carrying the generated matrix | ADR 0032 |

### MLX runtime — experimental, and slower on CUDA

It works. Models load, stay resident and generate correct output on both
Metal and CUDA. But it is not the path to reach for by default:

- **On CUDA it runs at 34–75% of the `cuda` path's decode throughput, median
  46%.** Measured across four matched model pairs on one host, one server
  process — [full report](docs/maxusai/vision-benchmark-mlx-cuda-vs-cuda-2026-08-30.md).
  "Roughly half" is a fair central estimate and a poor description of any
  single case: the spread is 2.2× and it is not architectural (the two dense
  pairs sit at 75% and 53%, the two MoE at 34% and 39%). Two of the four
  `mlx-cuda` arms had no stable throughput to quote at all.
- **Its bigger cost is variance, not speed.** `mlx-cuda`'s per-request spread
  is ~5× the `cuda` path's and reaches 46% within a single arm — same host,
  same prompt, back to back — while every `cuda` arm held inside ±1.5% first
  time. For anything that sets a timeout or compares two builds, that matters
  more than the ratio. It is per-model: `gemma4:31b-nvfp4` reproduced to 1.1%
  across four independent measurements.
- **On Metal it is the other way round.** A matched campaign measured MLX
  ~2.4× faster than llama-server (gemma4 12b: 121 vs 50 tok/s decode). So the
  CUDA gap is CUDA-specific, not an MLX property — do not generalise either
  number to the other platform.
- **Engine and quantization move together in every figure above** — nvfp4 on
  MLX against q4_K_M on GGUF, because those are the artefacts that exist. So
  the throughput numbers describe the two stacks **as shipped**, not the
  engine in isolation, and the same confound makes MLX-vs-GGUF *quality* an
  uncontrolled comparison: a quality difference cannot be attributed to the
  engine either. Nobody has separated them; until someone does, treat "which
  is better" as open.
- It is **converging with upstream's own MLX work** and is expected to be
  superseded by it; the fork has already retired its constrained-sampling
  layer in favour of upstream's engine (ADR 0033).

Use GGML/llama-server for anything where throughput or comparability matters.

Every row above is a delta we would rather not have. Each is offered upstream where
it is upstream's to take, and deleted from here once it lands there — the
`qwen25vl` gate and the MMQ padding fix are both filed and pending.

---

<p align="center">
  <a href="https://ollama.com">
    <img src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" alt="ollama" width="200"/>
  </a>
</p>

# Ollama

Start building with open models.

## Download

### macOS

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

or [download manually](https://ollama.com/download/Ollama.dmg)

### Windows

```shell
irm https://ollama.com/install.ps1 | iex
```

or [download manually](https://ollama.com/download/OllamaSetup.exe)

### Linux

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

[Manual install instructions](https://docs.ollama.com/linux#manual-install)

### Docker

The official [Ollama Docker image](https://hub.docker.com/r/ollama/ollama) `ollama/ollama` is available on Docker Hub.

### Libraries

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

### Community

- [Discord](https://discord.gg/ollama)
- [𝕏 (Twitter)](https://x.com/ollama)
- [Reddit](https://reddit.com/r/ollama)

## Get started

```
ollama
```

You'll be prompted to run a model or connect Ollama to your existing agents or applications such as `Claude Code`, `OpenClaw`, `OpenCode` , `Codex`, `Copilot`,  and more.

### Coding

To launch a specific integration:

```
ollama launch claude
```

Supported integrations include [Claude Code](https://docs.ollama.com/integrations/claude-code), [Codex](https://docs.ollama.com/integrations/codex), [Copilot CLI](https://docs.ollama.com/integrations/copilot-cli), [DeepSeek Harness](https://docs.ollama.com/integrations/deepseek-harness), [Droid](https://docs.ollama.com/integrations/droid), and [OpenCode](https://docs.ollama.com/integrations/opencode).

### AI assistant

Use [OpenClaw](https://docs.ollama.com/integrations/openclaw) to turn Ollama into a personal AI assistant across WhatsApp, Telegram, Slack, Discord, and more:

```
ollama launch openclaw
```

### Chat with a model

Run and chat with [Gemma 4](https://ollama.com/library/gemma4):

```
ollama run gemma4
```

See [ollama.com/library](https://ollama.com/library) for the full list.

See the [quickstart guide](https://docs.ollama.com/quickstart) for more details.

## REST API

Ollama has a REST API for running and managing models.

```
curl http://localhost:11434/api/chat -d '{
  "model": "gemma4",
  "messages": [{
    "role": "user",
    "content": "Why is the sky blue?"
  }],
  "stream": false
}'
```

See the [API documentation](https://docs.ollama.com/api) for all endpoints.

### Python

```
pip install ollama
```

```python
from ollama import chat

response = chat(model='gemma4', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response.message.content)
```

### JavaScript

```
npm i ollama
```

```javascript
import ollama from "ollama";

const response = await ollama.chat({
  model: "gemma4",
  messages: [{ role: "user", content: "Why is the sky blue?" }],
});
console.log(response.message.content);
```

## Supported backends

- [llama.cpp](https://github.com/ggml-org/llama.cpp) project founded by Georgi Gerganov.

## Documentation

- [CLI reference](https://docs.ollama.com/cli)
- [REST API reference](https://docs.ollama.com/api)
- [Importing models](https://docs.ollama.com/import)
- [Modelfile reference](https://docs.ollama.com/modelfile)
- [Building from source](https://github.com/ollama/ollama/blob/main/docs/development.md)

## Community Integrations

> Want to add your project? Open a pull request.

### Chat Interfaces

#### Web

- [Open WebUI](https://github.com/open-webui/open-webui) - Extensible, self-hosted AI interface
- [Onyx](https://github.com/onyx-dot-app/onyx) - Connected AI workspace
- [LibreChat](https://github.com/danny-avila/LibreChat) - Enhanced ChatGPT clone with multi-provider support
- [Lobe Chat](https://github.com/lobehub/lobe-chat) - Modern chat framework with plugin ecosystem ([docs](https://lobehub.com/docs/self-hosting/examples/ollama))
- [NextChat](https://github.com/ChatGPTNextWeb/ChatGPT-Next-Web) - Cross-platform ChatGPT UI ([docs](https://docs.nextchat.dev/models/ollama))
- [Perplexica](https://github.com/ItzCrazyKns/Perplexica) - AI-powered search engine, open-source Perplexity alternative
- [big-AGI](https://github.com/enricoros/big-AGI) - AI suite for professionals
- [Lollms WebUI](https://github.com/ParisNeo/lollms-webui) - Multi-model web interface
- [ChatOllama](https://github.com/sugarforever/chat-ollama) - Chatbot with knowledge bases
- [Bionic GPT](https://github.com/bionic-gpt/bionic-gpt) - On-premise AI platform
- [Chatbot UI](https://github.com/ivanfioravanti/chatbot-ollama) - ChatGPT-style web interface
- [Hollama](https://github.com/fmaclen/hollama) - Minimal web interface
- [Chatbox](https://github.com/Bin-Huang/Chatbox) - Desktop and web AI client
- [chat](https://github.com/swuecho/chat) - Chat web app for teams
- [Ollama RAG Chatbot](https://github.com/datvodinh/rag-chatbot.git) - Chat with multiple PDFs using RAG
- [Tkinter-based client](https://github.com/chyok/ollama-gui) - Python desktop client

#### Desktop

- [Dify.AI](https://github.com/langgenius/dify) - LLM app development platform
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) - All-in-one AI app for Mac, Windows, and Linux
- [Maid](https://github.com/Mobile-Artificial-Intelligence/maid) - Cross-platform mobile and desktop client
- [Witsy](https://github.com/nbonamy/witsy) - AI desktop app for Mac, Windows, and Linux
- [Cherry Studio](https://github.com/kangfenmao/cherry-studio) - Multi-provider desktop client
- [Ollama App](https://github.com/JHubi1/ollama-app) - Multi-platform client for desktop and mobile
- [PyGPT](https://github.com/szczyglis-dev/py-gpt) - AI desktop assistant for Linux, Windows, and Mac
- [Alpaca](https://github.com/Jeffser/Alpaca) - GTK4 client for Linux and macOS
- [SwiftChat](https://github.com/aws-samples/swift-chat) - Cross-platform including iOS, Android, and Apple Vision Pro
- [Enchanted](https://github.com/AugustDev/enchanted) - Native macOS and iOS client
- [RWKV-Runner](https://github.com/josStorer/RWKV-Runner) - Multi-model desktop runner
- [Ollama Grid Search](https://github.com/dezoito/ollama-grid-search) - Evaluate and compare models
- [macai](https://github.com/Renset/macai) - macOS client for Ollama and ChatGPT
- [AI Studio](https://github.com/MindWorkAI/AI-Studio) - Multi-provider desktop IDE
- [Reins](https://github.com/ibrahimcetin/reins) - Parameter tuning and reasoning model support
- [ConfiChat](https://github.com/1runeberg/confichat) - Privacy-focused with optional encryption
- [LLocal.in](https://github.com/kartikm7/llocal) - Electron desktop client
- [MindMac](https://mindmac.app) - AI chat client for Mac
- [Msty](https://msty.app) - Multi-model desktop client
- [BoltAI for Mac](https://boltai.com) - AI chat client for Mac
- [IntelliBar](https://intellibar.app/) - AI-powered assistant for macOS
- [Kerlig AI](https://www.kerlig.com/) - AI writing assistant for macOS
- [Hillnote](https://hillnote.com) - Markdown-first AI workspace
- [Perfect Memory AI](https://www.perfectmemory.ai/) - Productivity AI personalized by screen and meeting history

#### Mobile

- [Ollama Android Chat](https://github.com/sunshine0523/OllamaServer) - One-click Ollama on Android

> SwiftChat, Enchanted, Maid, Ollama App, Reins, and ConfiChat listed above also support mobile platforms.

### Code Editors & Development

- [Cline](https://github.com/cline/cline) - VS Code extension for multi-file/whole-repo coding
- [Continue](https://github.com/continuedev/continue) - Open-source AI code assistant for any IDE
- [Void](https://github.com/voideditor/void) - Open source AI code editor, Cursor alternative
- [Copilot for Obsidian](https://github.com/logancyang/obsidian-copilot) - AI assistant for Obsidian
- [twinny](https://github.com/rjmacarthy/twinny) - Copilot and Copilot chat alternative
- [gptel Emacs client](https://github.com/karthink/gptel) - LLM client for Emacs
- [Ollama Copilot](https://github.com/bernardo-bruning/ollama-copilot) - Use Ollama as GitHub Copilot
- [Obsidian Local GPT](https://github.com/pfrankov/obsidian-local-gpt) - Local AI for Obsidian
- [Ellama Emacs client](https://github.com/s-kostyaev/ellama) - LLM tool for Emacs
- [orbiton](https://github.com/xyproto/orbiton) - Config-free text editor with Ollama tab completion
- [AI ST Completion](https://github.com/yaroslavyaroslav/OpenAI-sublime-text) - Sublime Text 4 AI assistant
- [VT Code](https://github.com/vinhnx/vtcode) - Rust-based terminal coding agent with Tree-sitter
- [QodeAssist](https://github.com/Palm1r/QodeAssist) - AI coding assistant for Qt Creator
- [AI Toolkit for VS Code](https://aka.ms/ai-tooklit/ollama-docs) - Microsoft-official VS Code extension
- [Open Interpreter](https://docs.openinterpreter.com/language-model-setup/local-models/ollama) - Natural language interface for computers

### Libraries & SDKs

- [LiteLLM](https://github.com/BerriAI/litellm) - Unified API for 100+ LLM providers
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel/tree/main/python/semantic_kernel/connectors/ai/ollama) - Microsoft AI orchestration SDK
- [LangChain4j](https://github.com/langchain4j/langchain4j) - Java LangChain ([example](https://github.com/langchain4j/langchain4j-examples/tree/main/ollama-examples/src/main/java))
- [LangChainGo](https://github.com/tmc/langchaingo/) - Go LangChain ([example](https://github.com/tmc/langchaingo/tree/main/examples/ollama-completion-example))
- [Spring AI](https://github.com/spring-projects/spring-ai) - Spring framework AI support ([docs](https://docs.spring.io/spring-ai/reference/api/chat/ollama-chat.html))
- [LangChain](https://python.langchain.com/docs/integrations/chat/ollama/) and [LangChain.js](https://js.langchain.com/docs/integrations/chat/ollama/) with [example](https://js.langchain.com/docs/tutorials/local_rag/)
- [Ollama for Ruby](https://github.com/crmne/ruby_llm) - Ruby LLM library
- [any-llm](https://github.com/mozilla-ai/any-llm) - Unified LLM interface by Mozilla
- [OllamaSharp for .NET](https://github.com/awaescher/OllamaSharp) - .NET SDK
- [LangChainRust](https://github.com/Abraxas-365/langchain-rust) - Rust LangChain ([example](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_ollama.rs))
- [Agents-Flex for Java](https://github.com/agents-flex/agents-flex) - Java agent framework ([example](https://github.com/agents-flex/agents-flex/tree/main/agents-flex-llm/agents-flex-llm-ollama/src/test/java/com/agentsflex/llm/ollama))
- [Elixir LangChain](https://github.com/brainlid/langchain) - Elixir LangChain
- [Ollama-rs for Rust](https://github.com/pepperoni21/ollama-rs) - Rust SDK
- [LangChain for .NET](https://github.com/tryAGI/LangChain) - .NET LangChain ([example](https://github.com/tryAGI/LangChain/blob/main/examples/LangChain.Samples.OpenAI/Program.cs))
- [chromem-go](https://github.com/philippgille/chromem-go) - Go vector database with Ollama embeddings ([example](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama))
- [LangChainDart](https://github.com/davidmigloz/langchain_dart) - Dart LangChain
- [LlmTornado](https://github.com/lofcz/llmtornado) - Unified C# interface for multiple inference APIs
- [Ollama4j for Java](https://github.com/ollama4j/ollama4j) - Java SDK
- [Ollama for Laravel](https://github.com/cloudstudio/ollama-laravel) - Laravel integration
- [Ollama for Swift](https://github.com/mattt/ollama-swift) - Swift SDK
- [LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/llm/ollama/) and [LlamaIndexTS](https://ts.llamaindex.ai/modules/llms/available_llms/ollama) - Data framework for LLM apps
- [Haystack](https://github.com/deepset-ai/haystack-integrations/blob/main/integrations/ollama.md) - AI pipeline framework
- [Firebase Genkit](https://firebase.google.com/docs/genkit/plugins/ollama) - Google AI framework
- [Ollama-hpp for C++](https://github.com/jmont-dev/ollama-hpp) - C++ SDK
- [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) - Julia LLM toolkit ([example](https://svilupp.github.io/PromptingTools.jl/dev/examples/working_with_ollama))
- [Ollama for R - rollama](https://github.com/JBGruber/rollama) - R SDK
- [Portkey](https://portkey.ai/docs/welcome/integration-guides/ollama) - AI gateway
- [Testcontainers](https://testcontainers.com/modules/ollama/) - Container-based testing
- [LLPhant](https://github.com/theodo-group/LLPhant?tab=readme-ov-file#ollama) - PHP AI framework

### Frameworks & Agents

- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT/blob/master/docs/content/platform/ollama.md) - Autonomous AI agent platform
- [crewAI](https://github.com/crewAIInc/crewAI) - Multi-agent orchestration framework
- [Strands Agents](https://github.com/strands-agents/sdk-python) - Model-driven agent building by AWS
- [Cheshire Cat](https://github.com/cheshire-cat-ai/core) - AI assistant framework
- [any-agent](https://github.com/mozilla-ai/any-agent) - Unified agent framework interface by Mozilla
- [Stakpak](https://github.com/stakpak/agent) - Open source DevOps agent
- [Hexabot](https://github.com/hexastack/hexabot) - Conversational AI builder
- [Neuro SAN](https://github.com/cognizant-ai-lab/neuro-san-studio) - Multi-agent orchestration ([docs](https://github.com/cognizant-ai-lab/neuro-san-studio/blob/main/docs/user_guide.md#ollama))

### RAG & Knowledge Bases

- [RAGFlow](https://github.com/infiniflow/ragflow) - RAG engine based on deep document understanding
- [R2R](https://github.com/SciPhi-AI/R2R) - Open-source RAG engine
- [MaxKB](https://github.com/1Panel-dev/MaxKB/) - Ready-to-use RAG chatbot
- [Minima](https://github.com/dmayboroda/minima) - On-premises or fully local RAG
- [Chipper](https://github.com/TilmanGriesel/chipper) - AI interface with Haystack RAG
- [ARGO](https://github.com/xark-argo/argo) - RAG and deep research on Mac/Windows/Linux
- [Archyve](https://github.com/nickthecook/archyve) - RAG-enabling document library
- [Casibase](https://casibase.org) - AI knowledge base with RAG and SSO
- [BrainSoup](https://www.nurgo-software.com/products/brainsoup) - Native client with RAG and multi-agent automation

### Bots & Messaging

- [LangBot](https://github.com/RockChinQ/LangBot) - Multi-platform messaging bots with agents and RAG
- [AstrBot](https://github.com/Soulter/AstrBot/) - Multi-platform chatbot with RAG and plugins
- [Discord-Ollama Chat Bot](https://github.com/kevinthedang/discord-ollama) - TypeScript Discord bot
- [Ollama Telegram Bot](https://github.com/ruecat/ollama-telegram) - Telegram bot
- [LLM Telegram Bot](https://github.com/innightwolfsleep/llm_telegram_bot) - Telegram bot for roleplay

### Terminal & CLI

- [aichat](https://github.com/sigoden/aichat) - All-in-one LLM CLI with Shell Assistant, RAG, and AI tools
- [oterm](https://github.com/ggozad/oterm) - Terminal client for Ollama
- [gollama](https://github.com/sammcj/gollama) - Go-based model manager for Ollama
- [tlm](https://github.com/yusufcanb/tlm) - Local shell copilot
- [tenere](https://github.com/pythops/tenere) - TUI for LLMs
- [ParLlama](https://github.com/paulrobello/parllama) - TUI for Ollama
- [llm-ollama](https://github.com/taketwo/llm-ollama) - Plugin for [Datasette's LLM CLI](https://llm.datasette.io/en/stable/)
- [ShellOracle](https://github.com/djcopley/ShellOracle) - Shell command suggestions
- [LLM-X](https://github.com/mrdjohnson/llm-x) - Progressive web app for LLMs
- [cmdh](https://github.com/pgibler/cmdh) - Natural language to shell commands
- [VT](https://github.com/vinhnx/vt.ai) - Minimal multimodal AI chat app

### Productivity & Apps

- [AppFlowy](https://github.com/AppFlowy-IO/AppFlowy) - AI collaborative workspace, self-hostable Notion alternative
- [Screenpipe](https://github.com/mediar-ai/screenpipe) - 24/7 screen and mic recording with AI-powered search
- [Vibe](https://github.com/thewh1teagle/vibe) - Transcribe and analyze meetings
- [Page Assist](https://github.com/n4ze3m/page-assist) - Chrome extension for AI-powered browsing
- [NativeMind](https://github.com/NativeMindBrowser/NativeMindExtension) - Private, on-device browser AI assistant
- [Ollama Fortress](https://github.com/ParisNeo/ollama_proxy_server) - Security proxy for Ollama
- [1Panel](https://github.com/1Panel-dev/1Panel/) - Web-based Linux server management
- [Writeopia](https://github.com/Writeopia/Writeopia) - Text editor with Ollama integration
- [QA-Pilot](https://github.com/reid41/QA-Pilot) - GitHub code repository understanding
- [Raycast extension](https://github.com/MassimilianoPasquini97/raycast_ollama) - Ollama in Raycast
- [Painting Droid](https://github.com/mateuszmigas/painting-droid) - Painting app with AI integrations
- [Serene Pub](https://github.com/doolijb/serene-pub) - AI roleplaying app
- [Mayan EDMS](https://gitlab.com/mayan-edms/mayan-edms) - Document management with Ollama workflows
- [TagSpaces](https://www.tagspaces.org) - File management with [AI tagging](https://docs.tagspaces.org/ai/)

### Observability & Monitoring

- [Opik](https://www.comet.com/docs/opik/cookbook/ollama) - Debug, evaluate, and monitor LLM applications
- [OpenLIT](https://github.com/openlit/openlit) - OpenTelemetry-native monitoring for Ollama and GPUs
- [Lunary](https://lunary.ai/docs/integrations/ollama) - LLM observability with analytics and PII masking
- [Langfuse](https://langfuse.com/docs/integrations/ollama) - Open source LLM observability
- [HoneyHive](https://docs.honeyhive.ai/integrations/ollama) - AI observability and evaluation for agents
- [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html#automatic-tracing) - Open source LLM observability

### Database & Embeddings

- [pgai](https://github.com/timescale/pgai) - PostgreSQL as a vector database ([guide](https://github.com/timescale/pgai/blob/main/docs/vectorizer-quick-start.md))
- [MindsDB](https://github.com/mindsdb/mindsdb/blob/staging/mindsdb/integrations/handlers/ollama_handler/README.md) - Connect Ollama with 200+ data platforms
- [chromem-go](https://github.com/philippgille/chromem-go/blob/v0.5.0/embed_ollama.go) - Embeddable vector database for Go ([example](https://github.com/philippgille/chromem-go/tree/v0.5.0/examples/rag-wikipedia-ollama))
- [Kangaroo](https://github.com/dbkangaroo/kangaroo) - AI-powered SQL client

### Infrastructure & Deployment

#### Cloud

- [Google Cloud](https://cloud.google.com/run/docs/tutorials/gpu-gemma2-with-ollama)
- [Fly.io](https://fly.io/docs/python/do-more/add-ollama/)
- [Koyeb](https://www.koyeb.com/deploy/ollama)
- [Harbor](https://github.com/av/harbor) - Containerized LLM toolkit with Ollama as default backend

#### Package Managers

- [Pacman](https://archlinux.org/packages/extra/x86_64/ollama/)
- [Homebrew](https://formulae.brew.sh/formula/ollama)
- [Nix package](https://search.nixos.org/packages?show=ollama&from=0&size=50&sort=relevance&type=packages&query=ollama)
- [Helm Chart](https://artifacthub.io/packages/helm/ollama-helm/ollama)
- [Gentoo](https://github.com/gentoo/guru/tree/master/app-misc/ollama)
- [Flox](https://flox.dev/blog/ollama-part-one)
- [Guix channel](https://codeberg.org/tusharhero/ollama-guix)
