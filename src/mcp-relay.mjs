#!/usr/bin/env node
/**
 * stdio MCP server process for ACP agents: pipes stdin and stdout to Pi's
 * MCP socket. The agent spawns it from the `session/new` entry; the socket
 * path is argv[2], and PI_MCP_TOKEN / PI_MCP_AGENT / PI_MCP_SESSION
 * authenticate the connection and route it to one agent's ACP session. Exits when either side closes.
 */

import { connect } from "node:net";

const socketPath = process.argv[2];
const token = process.env.PI_MCP_TOKEN;
const agent = process.env.PI_MCP_AGENT;
const session = process.env.PI_MCP_SESSION;
if (!socketPath || !token || !agent || !session) {
	process.stderr.write("mcp-relay: missing socket path, PI_MCP_TOKEN, PI_MCP_AGENT or PI_MCP_SESSION\n");
	process.exit(2);
}

const socket = connect(socketPath, () => {
	socket.write(`${JSON.stringify({ token, agent, session })}\n`);
	process.stdin.pipe(socket);
	socket.pipe(process.stdout);
});
socket.on("error", (error) => {
	process.stderr.write(`mcp-relay: ${error.message}\n`);
	process.exit(1);
});
socket.on("close", () => process.exit(0));
process.stdin.on("end", () => socket.end());
