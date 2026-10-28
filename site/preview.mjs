// A local preview of the page: serves public/ with the headers Cloudflare will send (public/_headers), and proxies the
// record's files under /v1/, because the record does not send CORS headers to other origins yet. The page is pointed
// at the proxy by rewriting its laplace:record meta tag; nothing else differs from what is deployed.
//
//   node preview.mjs [--host 127.0.0.1] [--port 8787]

import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const RECORD = "https://data.laplaceprotocol.com/v1/";
const PUBLIC = fileURLToPath(new URL("public/", import.meta.url));
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".json": "application/json",
};

/** The headers a Cloudflare `_headers` file sets for every path (the only rule this site uses). */
export function parseHeaders(text) {
  const headers = {};
  for (const line of text.split("\n")) {
    const match = /^\s+([A-Za-z-]+):\s*(.+)$/.exec(line);
    if (match) headers[match[1]] = match[2].trim();
  }
  return headers;
}

/** index.html pointed at a proxy of the record instead of the record itself. */
export function pointAtProxy(html, proxy) {
  const tag = `<meta name="laplace:record" content="${RECORD}" />`;
  if (!html.includes(tag)) throw new Error("index.html has no laplace:record meta tag to rewrite");
  return html.replace(tag, `<meta name="laplace:record" content="${proxy}" />`);
}

// The record's files the page reads, and nothing else, are proxied.
const RECORD_FILE =
  /^\/v1\/(latest\.json|scoreboard\.json|issues\/\d+\/[a-z0-9-]+\.json|releases\/r[0-9.]+\/programs\/[a-z0-9][a-z0-9.-]*)$/;

/** The record's URL for a proxied /v1/ path, or null for anything that is not one of the record's files. */
export function recordUrl(pathname) {
  const file = RECORD_FILE.exec(pathname)?.[1];
  return file === undefined ? null : "https://data.laplaceprotocol.com/v1/" + file;
}

async function serve(request, response, headers) {
  const url = new URL(request.url, "http://preview");
  if (request.method !== "GET" && request.method !== "HEAD") {
    response.writeHead(405).end();
    return;
  }
  if (url.pathname.startsWith("/v1/")) {
    const target = recordUrl(url.pathname);
    if (target === null) {
      response.writeHead(404).end();
      return;
    }
    const upstream = await fetch(target);
    response.writeHead(upstream.status, {
      "content-type": upstream.headers.get("content-type") ?? "application/octet-stream",
      "cache-control": upstream.headers.get("cache-control") ?? "no-store",
    });
    response.end(Buffer.from(await upstream.arrayBuffer()));
    return;
  }
  const path = normalize(join(PUBLIC, url.pathname === "/" ? "index.html" : url.pathname));
  if (!path.startsWith(PUBLIC) || path.endsWith("_headers")) {
    response.writeHead(404).end();
    return;
  }
  let body;
  try {
    body = await readFile(path);
  } catch {
    response.writeHead(404).end();
    return;
  }
  if (path.endsWith("index.html")) body = Buffer.from(pointAtProxy(body.toString(), "/v1/"));
  response.writeHead(200, { ...headers, "content-type": TYPES[extname(path)] ?? "application/octet-stream" });
  response.end(request.method === "HEAD" ? undefined : body);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: { host: { type: "string", default: "127.0.0.1" }, port: { type: "string", default: "8787" } } });
  const headers = parseHeaders(await readFile(join(PUBLIC, "_headers"), "utf8"));
  createServer((request, response) =>
    serve(request, response, headers).catch((error) => {
      console.error(error);
      if (!response.headersSent) response.writeHead(502);
      response.end();
    }),
  ).listen(Number(values.port), values.host, () => {
    console.log(`preview on http://${values.host}:${values.port}/ (record proxied from ${RECORD})`);
  });
}
