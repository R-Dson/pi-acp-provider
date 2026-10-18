# pi-acp-provider

A [Pi](https://github.com/earendil-works/pi) provider extension that drives any
[Agent Client Protocol](https://agentclientprotocol.com) (ACP) agent, such as
GitHub Copilot CLI, Gemini CLI, or any other command that speaks ACP over stdio.

```
┌────────────────────────────────────────────────────────┐
│                        Pi TUI                          │
│         (Input bar, Stream Renderer, Status line)      │
└───────────────────────────▲────────────────────────────┘
                            │ Pi AssistantMessageEventStream
┌───────────────────────────▼────────────────────────────┐
│                  ACP Provider Extension                │
│        streamSimple() → ACP client (NDJSON RPC)        │
└───────────────────────────┼────────────────────────────┘
                            │ stdio (stdin / stdout), one process per agent
┌───────────────────────────▼────────────────────────────┐
│          ACP agent (e.g. `copilot --acp`, `gemini …`)  │
│       (Autonomous agent, file tools, bash, LSP)        │
└────────────────────────────────────────────────────────┘
```

Instead of Pi running its own LLM tool loop, the ACP agent does the work, the
same way Zed uses ACP agents: it plans, edits files, runs commands, and streams
progress back. Pi's own tools are bypassed. The agent runs everything itself
in the session working directory.

## Install

```bash
pi install npm:pi-acp-provider
```

Or from git: `pi install git:github.com/R-Dson/pi-acp-provider`.

To try it for one session without installing: `pi -e npm:pi-acp-provider`.

## Configuration

Declare agents under `acp.agents` in `~/.pi/agent/settings.json`. Each key
becomes a model: `/model acp/<key>`.

```json
{
  "acp": {
    "autoApproveTools": true,
    "agents": {
      "copilot": { "name": "GitHub Copilot", "command": "copilot", "args": ["--acp"] },
      "gemini": { "name": "Gemini", "command": "gemini", "args": ["--experimental-acp"] }
    }
  }
}
```

The commands above are examples. Check each agent's documentation for the
flag that starts its ACP mode. With no agents configured, Pi shows a warning at
startup and the provider has no models.

### Models and thinking level

Models are discovered automatically. When Pi refreshes its model list (at
interactive startup and whenever `/model` opens), each agent is started and an
ACP session is opened to read the models it offers. They show up as
`acp/<key>/<model id>`, e.g. `/model acp/copilot/gpt-5`. `acp/<key>` keeps the
agent's own default. The list is saved, so `pi -p` and offline startups use the
last one found.

Both the ACP `configOptions` model selector and the older `models` /
`session/set_model` API are supported. If the agent exposes a `thought_level`
option, Pi's thinking level (`shift+tab`, `--thinking`) is applied to it; levels
the agent doesn't offer are hidden. Discovery never triggers `authMethod`
sign-in; that waits for the first prompt.

Per-agent keys:

| Key | Required | Meaning |
|---|---|---|
| `command` | yes | Executable, spawned in the session cwd. |
| `args` | no | Arguments, e.g. the agent's ACP flag. Default `[]`. |
| `name` | no | Display name in the model picker and messages. Default: the key. |
| `env` | no | Extra environment variables for the agent process. |
| `authMethod` | no | ACP `authenticate` method id to use when the agent reports `auth_required` (see Auth below). |

Global keys:

| Key | Env override | Default | Meaning |
|---|---|---|---|
| `autoApproveTools` | `ACP_AUTO_APPROVE` | `true` | When `false`, each permission request opens a select dialog (TUI/RPC) or is denied (print/JSON mode). |
| `handshakeTimeoutMs` | | `30000` | Milliseconds to wait for `initialize` and `session/new`. |
| | `ACP_DEBUG=1` | off | Wire diagnostics on stderr. |

## Behavior

- **Lifecycle**: each agent's process spawns lazily on its first prompt,
  stays alive across prompts, and is killed on session shutdown (`/new`,
  `/resume`, quit, reload). Each agent runs in its own process group, so the
  whole tree (including tool subprocesses) is killed. Parent signal hooks
  cover exits without a session shutdown. A crashed agent is respawned with a
  fresh session on the next prompt.
- **Session mapping**: one ACP session per Pi session per agent. Each Pi model
  turn sends only the new user message, because the agent keeps conversation
  history on its side. Image attachments are noted but not forwarded.
- **Streaming**: ACP v1 `session/update` notifications. `agent_thought_chunk`
  maps to Pi thinking blocks and `agent_message_chunk` to text blocks. Tool
  calls render as `⚙️ Executing [title]...` and, on completion,
  `✓ title: <first line of output>` (or `✗` on failure). Other update kinds
  (plans, mode changes) are ignored.
- **Abort**: Pi's abort signal (Ctrl+C) sends `session/cancel` and finalizes
  the message as aborted immediately. It answers any open permission prompt
  with `cancelled` and leaves the session usable. The next prompt waits (up to
  5 s) for the agent to confirm the cancellation, so late updates can't leak
  into it.
- **Permissions**: `session/request_permission` is auto-approved by default
  (`allow_once`, never `allow_always`). Set `autoApproveTools: false` to get an
  interactive select dialog that names the asking agent.
- **Auth**: when an agent answers `session/new` with ACP `auth_required`
  (code -32000), the provider calls `authenticate` with the agent's configured
  `authMethod` and retries once. Without one, the error lists the auth methods
  the agent advertised in `initialize`, so you can sign in with the agent's own
  CLI or pick a method. A missing command reports a not-found hint naming the
  agent. The provider registers as ambient-auth so `/model` selection never
  blocks.
- **Usage**: ACP does not report per-turn token usage, so the provider records
  zeros.

### Known limitations

- No image input (prompts are sent as text only).
- `/resume` and `/fork` start a fresh ACP session. Prior Pi history is not
  replayed to the agent, and switching models mid-conversation does not carry
  history across agents.
- No ACP client capabilities (`fs`, `terminal`) are offered, so agents use
  their own file and terminal tools. Other agent→client requests get
  method-not-found.
- Of Pi's stream hooks, `onPayload` and `onProviderStreamEvent` are honored.
  `onResponse` is HTTP-specific and not called.

## Development

```bash
npm run typecheck   # strict tsc, no emit
npm test            # node:test suite against a scripted fake ACP agent
```

The fake agent (`test/fake-agent.mjs`) speaks ACP v1 wire shapes and covers
streaming, permissions (string request ids), cancellation, crashes, auth
(`auth_required` plus `authenticate`), refusals, and split NDJSON frames. No
real agent or network is needed.

## Layout

- `src/acp.ts`: ACP v1 / JSON-RPC wire types and boundary parsers
- `src/acp-client.ts`: child process, NDJSON transport, handshake, sessions, auth
- `src/provider.ts`: the Pi `Provider` (one model per agent) and event translation
- `src/config.ts`: settings and env resolution
- `src/index.ts`: extension factory (`pi.registerProvider`, lifecycle wiring)
