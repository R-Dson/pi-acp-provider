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

The ACP agent runs the loop, the same way Zed uses ACP agents: it plans, edits
files, runs commands, and streams progress back. Pi still provides what it
provides to any model: its tools (served to the agent over MCP and run by Pi,
with Pi's approvals, hooks and tool rows), its system prompt (context files,
skills, extension sections), its message queue (steering and follow-ups), and
its history (resumed and forked sessions, model switches).

## Install

```bash
pi install npm:pi-acp-provider
```

Or from git: `pi install git:github.com/R-Dson/pi-acp-provider`.

To try it for one session without installing: `pi -e npm:pi-acp-provider`.

## Configuration

Declare agents under `acp.agents` in `~/.pi/agent/settings.json` (or
`$PI_CODING_AGENT_DIR/settings.json`). Each key becomes a model:
`/model acp/<key>`.

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

Each model's context window starts at 128K. Once the agent reports its real
window (`size` in an ACP `usage_update`) during a turn, the next model list
refresh publishes that value for the model that ran the turn, and it is saved
with the list. A `contextWindow` in a `~/.pi/agent/models.json`
`modelOverrides` entry (for example
`{ "providers": { "acp": { "modelOverrides": { "claude": { "contextWindow": 1000000 } } } } }`
for `acp/claude`) always wins over both, since Pi applies it on top of the
provider's list.

### Agent slash commands

Each agent registers a Pi command named after its key. `/<key> <command>`
runs a slash command the agent advertises over ACP (`available_commands_update`),
e.g. `/copilot context` or `/copilot session info` for Copilot CLI. Arguments
autocomplete from the agent's current list. `/<key>` alone lists the commands.
Running one switches to that agent if another model is active, then sends
`/<command>` as the prompt, which is how ACP agents expect commands.

Per-agent keys:

| Key | Required | Meaning |
|---|---|---|
| `command` | yes | Executable, spawned in the session cwd. |
| `args` | no | Arguments, e.g. the agent's ACP flag. Default `[]`. |
| `name` | no | Display name in the model picker and messages. Default: the key. |
| `env` | no | Extra environment variables for the agent process. |
| `authMethod` | no | ACP `authenticate` method id to use when the agent reports `auth_required` (see Auth below). |
| `piTools` | no | `both` (Pi's tools next to the agent's own), `pi` (Pi's tools only; needs `ownToolsOffMeta`), or `off`. |
| `mcpTransport` | no | `stdio` or `http` for Pi's MCP server. `http` falls back to stdio when the agent does not advertise it. |
| `systemPrompt` | no | `prompt` (in front of the first prompt), `session` (ACP `systemPrompt` on `session/new`; falls back to `prompt`), or `off`. |
| `sessionMeta` | no | Extra `_meta` for `session/new`, merged over the preset. |
| `ownToolsOffMeta` | no | `_meta` that turns off the agent's own tools. Live sessions merge it in only with `piTools: "pi"`; isolated requests (summaries) always send it, merged over `sessionMeta`, whatever `piTools` says. |

#### Defaults and presets

Agents start from what every ACP agent supports (the level of Antigravity
and Mistral Vibe): `piTools: "both"`, `mcpTransport: "stdio"`,
`systemPrompt: "prompt"`. Two commands get a preset; any key in settings wins
(`env`, `sessionMeta` and `ownToolsOffMeta` merge key by key):

| Command | Preset |
|---|---|
| `claude-agent-acp` | `piTools: "pi"` (`tools: []` and `strictMcpConfig` via `_meta.claudeCode.options`), `mcpTransport: "http"`, `systemPrompt: "session"`, `allowedTools: ["mcp__pi"]`, env `ENABLE_CLAUDEAI_MCP_SERVERS=0` (no claude.ai cloud connectors) and `DISABLE_AUTO_COMPACT=1` (Pi owns compaction; also stops the adapter's idle compaction) |
| `codex-acp` | `piTools: "both"`, `mcpTransport: "http"`, `systemPrompt: "session"` |

Global keys:

| Key | Env override | Default | Meaning |
|---|---|---|---|
| `autoApproveTools` | `ACP_AUTO_APPROVE` | `true` | When `false`, each permission request opens a select dialog (TUI/RPC) or is denied (print/JSON mode). |
| `handshakeTimeoutMs` | | `30000` | Milliseconds to wait for `initialize`, `session/new`, and model or thinking-level changes. |
| `nativeTools` | `ACP_NATIVE_TOOLS` | `true` | Show agent tool calls as native Pi tool rows. `false` restores the inline text notices. |
| | `ACP_DEBUG=1` | off | Wire diagnostics on stderr. |

## Behavior

- **Lifecycle**: each agent's process spawns lazily on first use: model
  discovery (see Models and thinking level) or, when discovery has not run,
  the first prompt. It stays alive across prompts, and is killed on session shutdown (`/new`,
  `/resume`, quit, reload). Each agent runs in its own process group, so the
  whole tree (including tool subprocesses) is killed. Parent signal hooks,
  shared by all agents, cover exits without a session shutdown. A crashed
  agent is respawned with a fresh session on the next prompt; a request that
  was still being prepared when its agent exited fails with a request to send
  the message again, rather than reaching the respawned agent's sessions.
- **Session mapping**: one ACP session per Pi session per agent. Pi sessions
  that stream at the same time in one Pi process (for example an extension's
  side conversation) run in separate ACP sessions of the same agent process,
  each with its own Pi tools, and neither cancels the other's turn; streams
  to different agents never cancel each other either. A new request still
  replaces the running turn of the same agent in the same Pi session. The
  agent keeps the conversation on its side, so each prompt carries only what it has
  not seen: every queued user message (with images, when the agent accepts
  them), plus a `<conversation-history>` transcript of any part of Pi's
  history the session missed (a resumed or forked Pi session, another
  model's turns, a branch the session did not see). When Pi's history no
  longer holds the session's latest reply, including an aborted one (tree
  navigation or a fork to an earlier point, even one that keeps an earlier
  reply), or a message the session received that a later request still held,
  or its first message changed (compaction), the provider starts a new ACP
  session from Pi's history. Pi does not say which messages it saved and
  which an extension added to one request only, so a message the session
  received counts only once a later request still holds it, and only if the
  session actually saw it (a transcript leaves out empty replies and cuts the
  oldest part when it is too long). An edit that removes a message before a
  later request has held it is not noticed.
- **Pi's tools**: Pi's current tools are served by an MCP server inside Pi,
  as the server `pi` (`mcp__pi__<tool>` in Claude Code). A call ends the Pi
  message with a real Pi tool call; Pi runs it and the result answers the
  agent's MCP request, and the ACP turn continues. A changed loadout is served
  on the next `tools/list` (stdio connections also get `list_changed`; HTTP
  connections do not, see Known limitations).
  The HTTP transport listens on 127.0.0.1 and the stdio relay uses a socket in
  a private temp directory. Each ACP session gets its own nonce (in the HTTP
  path, or `PI_MCP_SESSION` for the relay) and its own token, derived from
  the agent id and nonce with a per-process key and compared in constant
  time, so a token works only for the agent and session it was issued for
  (HTTP answers 401 otherwise). Request bodies and relay request lines are
  capped at 8 MiB, and the relay's handshake line, which arrives before the
  connection is authenticated, at 4 KiB. Past a cap, HTTP answers 413 and
  closes the connection without reading the rest, and the relay connection is
  closed before the line is parsed. A call is accepted only from the session
  of the running turn, and an ended session's nonce is revoked: HTTP answers
  403, its calls still running are cancelled, and its relay connections are
  closed. Over HTTP, `initialize` issues an `Mcp-Session-Id`; a cancellation
  sent in a later POST of that MCP session reaches its call, and `DELETE`
  cancels the session's calls and ends it.
  A permission request for a Pi tool is approved
  without asking, since Pi applies its own approval when it runs the tool,
  but only when its structure shows it is one: it comes from the running
  turn's session, which holds a live MCP nonce, it names a tool call that
  the agent reported and has not reported finished, and every structured
  tool name present (ACP `name` and Claude Code's `_meta.claudeCode.toolName`,
  on the request and on each report of the call) names one and the same tool
  Pi serves, such as `mcp__pi__read`. Titles, and how the call is shown in
  Pi, are never trusted. Every other request goes to the normal permission
  policy.
  If Pi's MCP server fails to start, an agent with `piTools: "pi"` (the
  `claude-agent-acp` preset) ends the turn with an error naming Pi's tool
  server instead of running with its built-in tools; a `piTools: "both"`
  agent goes on without Pi's tools. The next turn tries to start the server
  again, and a session opened without Pi's tools is then replaced by a new
  one that has them and first gets the conversation so far as a transcript.
- **System prompt**: Pi's system prompt goes to each ACP session once (see
  `systemPrompt`), and a changed prompt is sent with the next prompt.
- **Queue and steering**: a message Pi queues while the agent works reaches
  the running turn with `_session/steering` when the agent advertises it
  (Claude Code, Codex). Otherwise it rides on the Pi tool result the agent is
  waiting for (its images as MCP image items), or, when none is pending, is sent as the next prompt once the
  running one ends. The running turn is not cancelled. A queued message
  counts as delivered only once the agent accepts the prompt that carries it.
  If that turn ends early (`max_tokens`, an abort, an error, a replaced
  session) or the agent rejects the prompt, the message goes with the next
  prompt instead, after a one-line note that it arrives late, and never as
  part of a `<conversation-history>` transcript. A message the agent rejected
  twice is dropped, and the error says so. A message no longer on Pi's
  active branch (after a fork or tree navigation) is not sent. The same holds
  for the rest of a prompt: a prompt the agent rejected, or one an `onPayload`
  hook failed on, is sent in full again with the next request, its user
  messages as requests even when Pi keeps the failed reply. Content an
  `onPayload` hook removes or rewrites counts as not sent, so it goes with
  the next prompt: a user message as a late user block, like a steer, and a
  removed `<conversation-history>` seed as a seed again. When a
  hook rebuilds blocks and two of them are identical, neither counts as sent.
- **Isolated requests**: compaction and branch summaries (`cacheRetention:
  "none"`) run in a separate, throwaway agent process, so the live session
  is not touched. They get no Pi tools, and when the agent has
  `ownToolsOffMeta` (the `claude-agent-acp` preset) it is sent on that
  session, merged over `sessionMeta`, whatever `piTools` says, so the summary
  cannot run the agent's own tools either. Agents without it get no `_meta`
  on that session, not even `sessionMeta`. Every permission request from an
  isolated session is answered with a reject option (`reject_once`, else
  `reject_always`), or `cancelled` when it offers none, without asking the
  permission policy. Cache-warming requests are answered without contacting the
  agent.
- **Streaming**: ACP v1 `session/update` notifications. `agent_thought_chunk`
  maps to Pi thinking blocks and `agent_message_chunk` to text blocks.
  `tool_call` and `tool_call_update` drive tool rows (see Tool calls),
  `usage_update` records the context window, `config_option_update` tracks
  the session's model and thinking-level settings, and
  `available_commands_update` feeds the agent slash commands. Plans and mode
  changes are ignored.
- **Tool calls** (`nativeTools`, the default): each agent tool call is a native
  Pi tool row. The provider ends the Pi message at the call with a `toolCall`
  block for the `acp_tool` mirror tool. Pi runs that tool, which waits for the
  agent's final report and returns its output or file diffs, and then calls
  the provider again, which continues the same ACP turn from buffered
  updates. One ACP turn therefore spans several Pi messages. The row shows the
  ACP kind, title and locations; the result shows diffs (Pi's diff renderer),
  text output with live partial updates, and failures. Expand the row to see
  the raw input and full output. `acp_tool` hides its own declaration, so no
  model is offered it; if it is not active (for example with a `--tools`
  allowlist), or with `nativeTools: false`, tool calls render inline as
  `⚙️ Executing [title]...` and `✓ title: <first line of output>`.
- **Abort**: Pi's abort signal (Ctrl+C) sends `session/cancel` and finalizes
  the message as aborted immediately. It answers any open permission prompt
  with `cancelled` and leaves the session usable. The next prompt waits (up to
  5 s) for the agent to confirm the cancellation; until then the cancelled
  prompt's updates are dropped and its permission requests answered
  `cancelled`, so they can't leak into the next prompt. If the agent does not
  confirm in time, the provider gives up on that session: it ends it, revokes
  its MCP nonce, and sends the next prompt on a new session, which first gets
  the conversation so far as a transcript. The old session's later updates,
  permission requests and Pi tool calls are refused. If this happens twice in
  a row, or the new session fails to open, the agent process is restarted
  instead.
- **Permissions**: `session/request_permission` is auto-approved by default
  (`allow_once`, never `allow_always`). Set `autoApproveTools: false` to get an
  interactive select dialog that names the asking agent. Isolated summary
  requests never get a permission granted (see Isolated requests).
- **Auth**: when an agent answers `session/new` with ACP `auth_required`
  (code -32000), the provider calls `authenticate` with the agent's configured
  `authMethod` and retries once. Without one, the error lists the auth methods
  the agent advertised in `initialize`, so you can sign in with the agent's own
  CLI or pick a method. A missing command reports a not-found hint naming the
  agent. The provider registers as ambient-auth so `/model` selection never
  blocks.
- **Usage**: token counts come from the `usage` of the `session/prompt`
  result when the agent reports it (unstable ACP field), summed over the
  ACP turn and recorded on its last Pi message. Cost is zero.

### Known limitations

- History the agent missed is sent as a plain-text transcript, not as native
  records: thinking, images and full tool outputs are left out or clipped.
- How an agent titles calls to Pi's MCP server is matched by pattern (the four
  agents above were checked). An unrecognized title is shown as an extra
  `acp_tool` row next to Pi's own row for the same call.
- With `autoApproveTools: false`, an agent that asks permission for a Pi tool
  without a structured tool name, or before it reports the tool call, gets a
  second prompt from the policy on top of Pi's own approval.
- Agents may time out long MCP calls on their side (for example while a Pi
  tool waits on the user). Claude Code's limit is effectively unbounded;
  other agents were not measured.
- No ACP client capabilities (`fs`, `terminal`) are offered, so agents use
  their own file and terminal tools. Other agent→client requests get
  method-not-found.
- Agents connected to Pi's MCP server over HTTP (`mcpTransport: "http"`, the
  `claude-agent-acp` and `codex-acp` presets) are not told when Pi's tool
  loadout changes (for example when an extension adds or removes a
  tool). The server offers no server-to-client stream, so only stdio
  connections get `notifications/tools/list_changed`. An HTTP agent sees the
  new tools only when it lists them again, which an agent may do only once per
  session. Start a new session with `/new` after changing the loadout. Calls
  to a tool that is no longer served fail with "Pi has no tool named …".
- Agents without an `ownToolsOffMeta` switch keep their built-in tools in
  isolated summary requests. This includes Codex (`codex-acp`), which has no
  documented ACP setting that turns them off. Their permission requests there
  are rejected, but a tool the agent runs without asking permission is not
  stopped.
- A context window the agent reports does not change the running Pi
  session: Pi reads the window from the selected model, which it replaces
  only when a model is selected or the provider is registered. The reported
  window applies after the next model list refresh (opening `/model`, or the
  next startup), once the model is selected again or Pi reloads.
- Of Pi's stream hooks, `onPayload` and `onProviderStreamEvent` are honored.
  `onResponse` is HTTP-specific and not called.

## Development

```bash
npm run typecheck   # strict tsc, no emit
npm test            # node:test suite against a scripted fake ACP agent (Node 22.6+)
```

The fake agent (`test/fake-agent.mjs`) speaks ACP v1 wire shapes and covers
streaming, permissions (string request ids), cancellation, crashes, auth
(`auth_required` plus `authenticate`), refusals, and split NDJSON frames. No
real agent or network is needed.

## Layout

- `src/acp.ts`: ACP v1 / JSON-RPC wire types and boundary parsers
- `src/acp-client.ts`: child process, NDJSON transport, handshake, sessions, auth
- `src/provider.ts`: the Pi `Provider` (agents and their discovered models), model and thinking-level selection, and event translation
- `src/config.ts`: settings, env resolution, and agent presets
- `src/mcp-server.ts`, `src/mcp-relay.mjs`: Pi's MCP server (HTTP and stdio relay)
- `src/transcript.ts`: Pi messages to ACP prompt blocks and history transcripts
- `src/index.ts`: extension factory (`pi.registerProvider`, lifecycle wiring)
