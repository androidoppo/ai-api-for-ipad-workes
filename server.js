/**
 * ipad-llm-api-server
 *
 * OpenAI-compatible API server that forwards inference to browser workers
 * (iPad + Safari + llama.cpp/wllama WebGPU) connected over WebSocket.
 *
 * Endpoints:
 *   POST /v1/chat/completions   OpenAI-compatible (stream & non-stream)
 *   GET  /v1/models             OpenAI-compatible model list
 *   GET  /healthz               health check
 *   GET  /admin/status          worker/job inspection (JSON)
 *   WS   /worker                worker registration channel
 *
 * Env:
 *   PORT              default 8080
 *   API_KEY           if set, clients must send "Authorization: Bearer <key>"
 *   WORKER_TOKEN      if set, workers must connect to /worker?token=<key>
 *   DEFAULT_MODEL     default model reference used when request omits "model"
 *   MAX_TOKENS_CAP    hard cap of max_tokens per request (default 2048)
 *   MAX_QUEUE         max queued jobs (default 32)
 */

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';
import { resolveModel, resolveLoraList } from './lib/hf-resolve.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC_DIR = path.join(__dirname, 'public');

const PORT = Number(process.env.PORT || 8080);
const API_KEY = process.env.API_KEY || '';
const WORKER_TOKEN = process.env.WORKER_TOKEN || '';
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || '';
const MAX_TOKENS_CAP = Number(process.env.MAX_TOKENS_CAP || 2048);
const MAX_QUEUE = Number(process.env.MAX_QUEUE || 32);

const JOB_QUEUE_TIMEOUT_MS = 60_000; // max wait in queue
const JOB_WATCHDOG_MS = 180_000; // no worker activity for this long -> job fails
const JOB_MAX_MS = 600_000; // absolute job cap
const HEARTBEAT_MS = 15_000;
const WORKER_STALE_MS = 60_000;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** @type {Map<string, WorkerConn>} */
const workers = new Map();
/** @type {Map<string, Job>} */
const jobs = new Map();
const queue = [];

let workerSeq = 0;
let jobSeq = 0;

class WorkerConn {
  constructor(ws, req) {
    this.ws = ws;
    this.id = 'w' + ++workerSeq;
    this.state = 'connecting'; // connecting | ready | loading | busy
    this.caps = {};
    this.loadedModel = null;
    this.loadedLoraKey = '';
    this.lastSeen = Date.now();
    this.busyJobId = null;
    this.assignedJobId = null; // dispatch sent, not yet accepted
  }
  send(obj) {
    if (this.ws.readyState === 1) this.ws.send(JSON.stringify(obj));
  }
  get isUsable() {
    return this.ws.readyState === 1 && (this.state === 'ready' || this.state === 'loading') && !this.assignedJobId && !this.busyJobId;
  }
}

class Job {
  constructor(reqBody, model, lora, isStream) {
    this.id = 'job-' + crypto.randomBytes(6).toString('hex');
    this.seq = ++jobSeq;
    this.createdAt = Date.now();
    this.lastActivity = Date.now();
    this.req = reqBody;
    this.modelUrl = model.url;
    this.modelRef = reqBody.model;
    this.lora = lora;
    this.isStream = isStream;
    this.status = 'queued'; // queued | dispatched | running | done | error
    this.worker = null;
    this.content = '';
    this.finishReason = null;
    this.usage = null;
    this.error = null;
    this.firstChunkSent = false;
    this.cancelled = false;
  }
  touch() {
    this.lastActivity = Date.now();
  }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

function json(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'content-type': 'application/json; charset=utf-8' });
  res.end(body);
}

function checkAuth(req) {
  if (!API_KEY) return true;
  const h = req.headers.authorization || '';
  const x = req.headers['x-api-key'] || '';
  return h === `Bearer ${API_KEY}` || x === API_KEY;
}

function serveStatic(req, res, pathname) {
  let rel = pathname === '/' ? '/index.html' : pathname;
  if (rel === '/worker' || rel === '/worker/') rel = '/worker/index.html';
  const filePath = path.normalize(path.join(PUBLIC_DIR, rel));
  if (!filePath.startsWith(PUBLIC_DIR)) {
    res.writeHead(403);
    return res.end('Forbidden');
  }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'content-type': 'text/plain' });
      return res.end('Not found');
    }
    res.writeHead(200, { 'content-type': MIME[path.extname(filePath)] || 'application/octet-stream' });
    res.end(data);
  });
}

function readBody(req, limit = 2 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks = [];
    req.on('data', (c) => {
      size += c.length;
      if (size > limit) {
        reject(new Error('Request body too large'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

// ---------------------------------------------------------------------------
// Job management
// ---------------------------------------------------------------------------

function loraKey(loraList) {
  return loraList.map((l) => `${l.url}@${l.scale}`).sort().join('|');
}

function pickWorker(modelUrl) {
  const usable = [...workers.values()].filter((w) => w.isUsable);
  if (usable.length === 0) return null;
  // prefer a worker that already has the exact model+lora state loaded
  const wantKey = modelUrl;
  const withModel = usable.filter((w) => w.state === 'ready' && w.loadedModel === wantKey);
  if (withModel.length > 0) {
    // least recently used
    withModel.sort((a, b) => a.lastSeen - b.lastSeen);
    return withModel[0];
  }
  const idle = usable.filter((w) => w.state === 'ready');
  if (idle.length > 0) {
    idle.sort((a, b) => a.lastSeen - b.lastSeen);
    return idle[0];
  }
  // only loading workers available (they are mid-unload/load; don't stack jobs)
  return null;
}

function tryDispatchQueued() {
  while (queue.length > 0) {
    const job = queue[0];
    const worker = pickWorker(job.modelUrl);
    if (!worker) return;
    queue.shift();
    dispatchJob(job, worker);
  }
}

function dispatchJob(job, worker) {
  job.status = 'dispatched';
  job.worker = worker;
  job.touch();
  worker.assignedJobId = job.id;
  worker.lastSeen = Date.now();
  worker.send({
    type: 'dispatch',
    job_id: job.id,
    model_url: job.modelUrl,
    lora: job.lora, // [{url, scale}]
    params: {
      messages: job.req.messages,
      max_tokens: Math.min(Number(job.req.max_tokens) || 512, MAX_TOKENS_CAP),
      temperature: typeof job.req.temperature === 'number' ? job.req.temperature : undefined,
      top_p: typeof job.req.top_p === 'number' ? job.req.top_p : undefined,
      top_k: typeof job.req.top_k === 'number' ? job.req.top_k : undefined,
      stop: job.req.stop,
      seed: typeof job.req.seed === 'number' ? job.req.seed : undefined,
      n_ctx: Math.min(Number(job.req.n_ctx) || Number(process.env.N_CTX || 2048), 8192),
      chat_template: typeof job.req.chat_template === 'string' ? job.req.chat_template : undefined,
    },
  });
}

function enqueue(job) {
  if (queue.length >= MAX_QUEUE) {
    job.status = 'error';
    job.error = { code: 'server_overloaded', message: 'Job queue is full. Try again later.' };
    respondError(job);
    return;
  }
  queue.push(job);
  tryDispatchQueued();
}

function finishJob(job) {
  job.status = 'done';
  cleanupJob(job);
  if (job.isStream) {
    if (job.res.writableEnded) return;
    const id = job.id;
    // final chunk with finish_reason (+usage)
    const chunk = {
      id,
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: job.modelRef,
      choices: [{ index: 0, delta: {}, finish_reason: job.finishReason || 'stop', logprobs: null }],
    };
    if (job.usage) chunk.usage = job.usage;
    job.res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    job.res.write('data: [DONE]\n\n');
    job.res.end();
  } else {
    if (job.res.writableEnded) return;
    json(job.res, 200, {
      id: job.id,
      object: 'chat.completion',
      created: Math.floor(job.createdAt / 1000),
      model: job.modelRef,
      choices: [
        {
          index: 0,
          message: { role: 'assistant', content: job.content },
          finish_reason: job.finishReason || 'stop',
          logprobs: null,
        },
      ],
      usage: job.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
      },
    });
  }
}

function respondError(job) {
  cleanupJob(job);
  const codeMap = {
    no_worker: 503,
    server_overloaded: 503,
    worker_gone: 502,
    worker_error: 502,
    load_error: 502,
    timeout: 504,
    bad_request: 400,
  };
  const code = codeMap[job.error?.code] || 500;
  if (job.res.writableEnded) return;
  json(job.res, code, {
    error: {
      message: job.error?.message || 'Internal error',
      type: job.error?.code || 'internal_error',
      code: job.error?.code || null,
    },
  });
}

function failJob(job, code, message) {
  if (job.status === 'done' || job.status === 'error') return;
  job.status = 'error';
  job.error = { code, message };
  // notify worker if it was running this job
  if (job.worker && job.worker.ws.readyState === 1 && job.worker.busyJobId === job.id) {
    job.worker.send({ type: 'cancel', job_id: job.id });
    job.worker.busyJobId = null;
    job.worker.state = 'ready';
  }
  if (job.worker) job.worker.assignedJobId = null;
  respondError(job);
}

function cleanupJob(job) {
  jobs.delete(job.id);
  const qi = queue.indexOf(job);
  if (qi >= 0) queue.splice(qi, 1);
  if (job.worker) {
    if (job.worker.assignedJobId === job.id) job.worker.assignedJobId = null;
    if (job.worker.busyJobId === job.id) {
      job.worker.busyJobId = null;
      job.worker.state = 'ready';
    }
  }
  tryDispatchQueued();
}

function cancelJob(job, notifyClient = true) {
  if (job.status === 'done' || job.status === 'error') return;
  job.cancelled = true;
  if (job.worker && job.worker.ws.readyState === 1 && (job.worker.busyJobId === job.id || job.worker.assignedJobId === job.id)) {
    job.worker.send({ type: 'cancel', job_id: job.id });
    if (job.worker.busyJobId === job.id) {
      job.worker.busyJobId = null;
      job.worker.state = 'ready';
    }
    job.worker.assignedJobId = null;
  }
  job.status = 'done'; // client is gone; nothing to write
  cleanupJob(job);
}

// ---------------------------------------------------------------------------
// Watchdogs
// ---------------------------------------------------------------------------

setInterval(() => {
  const now = Date.now();
  // queued timeouts
  for (const job of queue) {
    if (now - job.createdAt > JOB_QUEUE_TIMEOUT_MS) {
      job.status = 'error';
      job.error = { code: 'no_worker', message: 'No worker available for this model (timeout while queued).' };
      respondError(job);
    }
  }
  // running job watchdogs
  for (const job of jobs.values()) {
    if (job.status === 'dispatched' || job.status === 'running') {
      if (now - job.lastActivity > JOB_WATCHDOG_MS || now - job.createdAt > JOB_MAX_MS) {
        // recycle worker
        if (job.worker) {
          job.worker.state = 'ready';
          job.worker.assignedJobId = null;
          job.worker.busyJobId = null;
          try {
            job.worker.ws.close(4001, 'job watchdog');
          } catch {}
        }
        failJob(job, 'timeout', 'Worker did not respond in time (or job exceeded max duration).');
      }
    }
  }
  // stale workers
  for (const [id, w] of workers) {
    if (now - w.lastSeen > WORKER_STALE_MS) {
      try {
        w.ws.terminate();
      } catch {}
      dropWorker(w);
    }
  }
  // ws protocol-level pings
  for (const w of workers.values()) {
    if (w.ws.readyState === 1) {
      w.send({ type: 'ping', t: Date.now() });
    }
  }
}, 5000).unref();

function dropWorker(w) {
  if (workers.get(w.id) === w) workers.delete(w.id);
  const job = [...jobs.values()].find((j) => j.worker === w && (j.status === 'running' || j.status === 'dispatched'));
  if (job) failJob(job, 'worker_gone', 'Worker disconnected while processing the job.');
  tryDispatchQueued();
}

// ---------------------------------------------------------------------------
// HTTP server
// ---------------------------------------------------------------------------

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const p = url.pathname;

  // CORS (open: API is meant to be callable from anywhere)
  res.setHeader('access-control-allow-origin', '*');
  res.setHeader('access-control-allow-headers', 'authorization, content-type, x-api-key');
  res.setHeader('access-control-allow-methods', 'GET, POST, OPTIONS');
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  if (p === '/healthz') return json(res, 200, { ok: true, workers: workers.size, queued: queue.length });
  if (p === '/worker' && req.headers.upgrade) return; // handled by upgrade
  if (p.startsWith('/v1/') || p === '/admin/status') {
    if (!checkAuth(req)) {
      res.setHeader('www-authenticate', 'Bearer realm="api"');
      return json(res, 401, { error: { message: 'Invalid or missing API key', type: 'auth_error' } });
    }
  }

  if (p === '/v1/models' && req.method === 'GET') {
    const seen = new Set();
    const data = [];
    if (DEFAULT_MODEL) seen.add(DEFAULT_MODEL);
    for (const w of workers.values()) {
      if (w.loadedModel) seen.add(w.loadedModel);
    }
    for (const m of seen) {
      data.push({ id: m, object: 'model', created: 0, owned_by: 'ipad-worker' });
    }
    return json(res, 200, { object: 'list', data });
  }

  if (p === '/admin/status' && req.method === 'GET') {
    return json(res, 200, {
      uptime_s: Math.floor(process.uptime()),
      workers: [...workers.values()].map((w) => ({
        id: w.id,
        state: w.state,
        caps: w.caps,
        loaded_model: w.loadedModel,
        loaded_lora: w.loadedLoraKey || null,
        last_seen_s_ago: Math.floor((Date.now() - w.lastSeen) / 1000),
        busy_job: w.busyJobId,
      })),
      queued_jobs: queue.length,
      active_jobs: [...jobs.values()].filter((j) => j.status === 'running' || j.status === 'dispatched').length,
    });
  }

  if (p === '/v1/chat/completions' && req.method === 'POST') {
    return handleChatCompletions(req, res);
  }

  if (req.method === 'GET') {
    if (p === '/worker') {
      // keep query string (token etc.) across the redirect
      res.writeHead(301, { location: '/worker/' + url.search });
      return res.end();
    }
    return serveStatic(req, res, p);
  }

  json(res, 404, { error: { message: 'Not found', type: 'invalid_request_error' } });
});

async function handleChatCompletions(req, res) {
  let body;
  try {
    body = JSON.parse((await readBody(req)) || '{}');
  } catch (e) {
    return json(res, 400, { error: { message: 'Invalid JSON body: ' + e.message, type: 'invalid_request_error' } });
  }

  const isStream = body.stream === true;
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return json(res, 400, { error: { message: '"messages" must be a non-empty array', type: 'invalid_request_error' } });
  }

  const modelRef = typeof body.model === 'string' && body.model.trim() ? body.model.trim() : DEFAULT_MODEL;
  if (!modelRef) {
    return json(res, 400, {
      error: { message: '"model" is required (URL, "org/repo/file.gguf", or set DEFAULT_MODEL env)', type: 'invalid_request_error' },
    });
  }

  let model, loraList;
  try {
    model = await resolveModel(modelRef);
    loraList = await resolveLoraList(body.lora);
  } catch (e) {
    return json(res, 400, { error: { message: e.message, type: 'invalid_request_error' } });
  }

  const job = new Job(body, model, loraList, isStream);
  jobs.set(job.id, job);
  job.res = res;

  // client disconnect -> cancel
  req.on('close', () => {
    if (job.status !== 'done' && job.status !== 'error') cancelJob(job);
  });

  if (isStream) {
    res.writeHead(200, {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    });
    res.write(': connected\n\n'); // comment as keepalive / flush
  }

  enqueue(job);
}

server.listen(PORT, () => {
  console.log(`[server] listening on :${PORT}  (workers=${workers.size}, api_key=${API_KEY ? 'set' : 'off'}, worker_token=${WORKER_TOKEN ? 'set' : 'off'})`);
});

// ---------------------------------------------------------------------------
// WebSocket worker channel
// ---------------------------------------------------------------------------

const wss = new WebSocketServer({ noServer: true });

server.on('upgrade', (req, socket, head) => {
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname !== '/worker') {
    socket.write('HTTP/1.1 404 Not Found\r\n\r\n');
    return socket.destroy();
  }
  if (WORKER_TOKEN && url.searchParams.get('token') !== WORKER_TOKEN) {
    socket.write('HTTP/1.1 401 Unauthorized\r\n\r\n');
    return socket.destroy();
  }
  wss.handleUpgrade(req, socket, head, (ws) => {
    const w = new WorkerConn(ws, req);
    workers.set(w.id, w);
    console.log(`[server] worker connected: ${w.id}`);
    ws.on('message', (buf) => onWorkerMessage(w, buf));
    ws.on('close', () => {
      console.log(`[server] worker disconnected: ${w.id}`);
      dropWorker(w);
    });
    ws.on('error', () => dropWorker(w));
  });
});

function onWorkerMessage(w, buf) {
  w.lastSeen = Date.now();
  let msg;
  try {
    msg = JSON.parse(buf.toString());
  } catch {
    return;
  }
  switch (msg.type) {
    case 'hello': {
      w.caps = msg.caps || {};
      if (msg.loaded) {
        w.loadedModel = msg.loaded.model_url || null;
        w.loadedLoraKey = msg.loaded.lora_key || '';
        w.state = 'ready';
      } else {
        w.state = 'ready';
      }
      tryDispatchQueued();
      break;
    }
    case 'status': {
      // {state: 'ready'|'loading', loaded?: {model_url, lora_key}}
      if (msg.state) w.state = msg.state;
      if (msg.loaded !== undefined) {
        w.loadedModel = msg.loaded?.model_url || null;
        w.loadedLoraKey = msg.loaded?.lora_key || '';
      }
      break;
    }
    case 'job_accepted': {
      const job = jobs.get(msg.job_id);
      if (job && job.worker === w) {
        job.status = 'running';
        job.touch();
        w.assignedJobId = null;
        w.busyJobId = job.id;
        w.state = 'busy';
      }
      break;
    }
    case 'progress': {
      const job = jobs.get(msg.job_id);
      if (job && job.worker === w) job.touch();
      break;
    }
    case 'chunk': {
      const job = jobs.get(msg.job_id);
      if (!job || job.worker !== w) return;
      job.touch();
      job.status = 'running';
      const delta = typeof msg.delta === 'string' ? msg.delta : '';
      if (!delta) return;
      if (!job.firstChunkSent) {
        job.firstChunkSent = true;
        if (job.isStream && !job.res.writableEnded) {
          const first = {
            id: job.id,
            object: 'chat.completion.chunk',
            created: Math.floor(job.createdAt / 1000),
            model: job.modelRef,
            choices: [{ index: 0, delta: { role: 'assistant', content: delta }, finish_reason: null, logprobs: null }],
          };
          job.res.write(`data: ${JSON.stringify(first)}\n\n`);
        }
      } else if (job.isStream && !job.res.writableEnded) {
        const chunk = {
          id: job.id,
          object: 'chat.completion.chunk',
          created: Math.floor(job.createdAt / 1000),
          model: job.modelRef,
          choices: [{ index: 0, delta: { content: delta }, finish_reason: null, logprobs: null }],
        };
        job.res.write(`data: ${JSON.stringify(chunk)}\n\n`);
      }
      job.content += delta;
      break;
    }
    case 'job_done': {
      const job = jobs.get(msg.job_id);
      if (!job || job.worker !== w) return;
      job.finishReason = msg.finish_reason || 'stop';
      job.usage = msg.usage || null;
      job.timings = msg.timings || null;
      finishJob(job);
      break;
    }
    case 'job_cancelled': {
      const job = jobs.get(msg.job_id);
      if (!job || job.worker !== w) return;
      job.status = 'done';
      cleanupJob(job);
      break;
    }
    case 'job_error': {
      const job = jobs.get(msg.job_id);
      if (!job || job.worker !== w) return;
      failJob(job, msg.code || 'worker_error', msg.message || 'Worker failed to run the job.');
      break;
    }
    case 'heartbeat': {
      if (msg.loaded !== undefined) {
        w.loadedModel = msg.loaded?.model_url || null;
        w.loadedLoraKey = msg.loaded?.lora_key || '';
      }
      if (msg.state) w.state = msg.state;
      break;
    }
    case 'pong': {
      break;
    }
    default:
      break;
  }
}

process.on('SIGTERM', () => {
  console.log('[server] SIGTERM, shutting down');
  for (const w of workers.values()) {
    try {
      w.ws.close(1001, 'server shutdown');
    } catch {}
  }
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 3000).unref();
});
