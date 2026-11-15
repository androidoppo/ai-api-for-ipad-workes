/**
 * iPad browser worker.
 *
 * Connects to the API server via WebSocket, receives dispatch messages,
 * runs llama.cpp (wllama, WebGPU-enabled) fully in the browser, streams
 * tokens back to the server.
 *
 * LoRA: wllama supports `lora_adapters` referencing files inside its
 * Emscripten VFS. wllama names every blob passed to loadModel() as
 * "models/model-0000N-of-0000M.gguf", so we append the LoRA GGUF blob
 * after the base model blob and reference it as the last shard name.
 * Limitation: the base model must be a single-file GGUF for this trick.
 */
import { Wllama, LoggerWithoutDebug } from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.13.0/esm/index.js';
import WasmFromCDN from 'https://cdn.jsdelivr.net/npm/@wllama/wllama@3.13.0/esm/wasm-from-cdn.js';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const qs = new URLSearchParams(location.search);
const WORKER_TOKEN = qs.get('token') || '';
const AUTO_N_CTX = Number(qs.get('n_ctx') || 2048);
const NO_GPU = qs.get('cpu') === '1';
const WLLAMA_VERSION = '3.13.0';

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const logEl = $('log');
function log(line, cls = '') {
  const t = new Date().toLocaleTimeString();
  logEl.textContent = `[${t}] ${line}\n` + logEl.textContent;
  if (logEl.childElementCount === 0 && logEl.textContent.length > 20000) {
    logEl.textContent = logEl.textContent.slice(0, 20000);
  }
}
function setState(state) {
  const el = $('stState');
  el.textContent = state;
  el.className = 'value ' + (state === 'busy' || state === 'loading' ? 'warn' : 'ok');
}
function setWsState(text, ok) {
  const el = $('stWs');
  el.textContent = text;
  el.className = 'value ' + (ok ? 'ok' : 'bad');
}

$('stServer').textContent = location.host;
$('stGpu').textContent = NO_GPU ? 'disabled (cpu)' : navigator.gpu ? 'available' : 'not found';

// ---------------------------------------------------------------------------
// Model / LoRA cache helpers
// ---------------------------------------------------------------------------
const MODEL_CACHE = 'llm-models-v1';

async function cachedFetchBlob(url, onProgress) {
  const cache = await caches.open(MODEL_CACHE);
  const hit = await cache.match(url);
  if (hit) {
    log(`cache hit: ${short(url)}`);
    return await hit.blob();
  }
  log(`downloading: ${short(url)}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed: HTTP ${res.status} for ${url}`);
  const total = Number(res.headers.get('content-length') || 0);
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  let lastPct = -1;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    if (total && onProgress) {
      const pct = Math.floor((loaded / total) * 100);
      if (pct !== lastPct) {
        lastPct = pct;
        onProgress(pct);
      }
    }
  }
  const blob = new Blob(chunks);
  await cache.put(url, new Response(blob.slice(0)));
  return blob;
}

function short(s) {
  return s.length > 60 ? s.slice(0, 57) + '…' : s;
}

const pad5 = (n) => String(n).padStart(5, '0');

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------
let wllama = null;
let currentKey = ''; // modelUrl|loraKey of the currently loaded model
let loadedMeta = null;

async function unloadModel() {
  if (wllama) {
    try {
      await wllama.exit();
    } catch (e) {
      log(`exit error: ${e.message}`);
    }
    wllama = null;
    currentKey = '';
    loadedMeta = null;
    $('stModel').textContent = 'none';
    send({ type: 'status', state: 'ready', loaded: { model_url: null, lora_key: '' } });
  }
}

/**
 * Load a model (with optional LoRA adapters).
 * @param {string} modelUrl
 * @param {{url: string, scale: number}[]} loraList
 * @param {number} nCtx
 */
async function loadModel(modelUrl, loraList, nCtx, onProgress) {
  await unloadModel();
  setState('loading');
  wllama = new Wllama(WasmFromCDN, {
    parallelDownloads: 4,
    logger: LoggerWithoutDebug,
    allowOffline: true,
  });

  const loadParams = {
    n_ctx: nCtx,
    n_gpu_layers: NO_GPU ? 0 : 99999,
    progressCallback: ({ loaded, total }) => {
      if (total && onProgress) onProgress('model', Math.floor((loaded / total) * 100));
    },
  };

  if (loraList.length === 0) {
    await wllama.loadModelFromUrl(modelUrl, loadParams);
  } else {
    // Base model via wllama ModelManager (uses its own OPFS cache)
    const model = await wllama.modelManager.getModelOrDownload({ url: modelUrl }, loadParams);
    let blobs = await model.open();
    if (blobs.length !== 1) {
      throw new Error('LoRA requires a single-file GGUF base model (split shards are not supported for LoRA).');
    }
    // Fetch LoRA adapters (cached in Cache API)
    const loraBlobs = [];
    for (const lora of loraList) {
      loraBlobs.push(await cachedFetchBlob(lora.url, (p) => onProgress && onProgress('lora', p)));
    }
    const allBlobs = [...blobs, ...loraBlobs];
    const total = allBlobs.length;
    loadParams.lora_adapters = loraList.map((l, i) => ({
      path: `models/model-${pad5(i + 2)}-of-${pad5(total)}.gguf`, // i+2: base is shard 1
      scale: l.scale,
    }));
    log(`loading base+${loraBlobs.length} lora (scale=${loraList.map((l) => l.scale).join(',')})`);
    await wllama.loadModel(allBlobs, loadParams);
  }

  const info = wllama.getLoadedContextInfo();
  currentKey = makeKey(modelUrl, loraList);
  loadedMeta = {
    model_url: modelUrl,
    lora_key: loraList.map((l) => `${l.url}@${l.scale}`).join('|'),
    n_ctx: info?.n_ctx,
    n_gpu_layers: NO_GPU ? 0 : undefined,
  };
  $('stModel').textContent = short(modelUrl) + (loraList.length ? ` (+LoRA x${loraList.length})` : '');
  send({ type: 'status', state: 'ready', loaded: loadedMeta });
  log(`model loaded (ctx=${info?.n_ctx})`);
}

function makeKey(modelUrl, loraList) {
  return modelUrl + '|' + loraList.map((l) => `${l.url}@${l.scale}`).sort().join('|');
}

// ---------------------------------------------------------------------------
// WebSocket channel
// ---------------------------------------------------------------------------
let ws = null;
let reconnectDelay = 1000;
let jobsDone = 0;
const aborts = new Map(); // job_id -> AbortController

function send(obj) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = `${proto}://${location.host}/worker${WORKER_TOKEN ? `?token=${encodeURIComponent(WORKER_TOKEN)}` : ''}`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    reconnectDelay = 1000;
    setWsState('connected', true);
    send({
      type: 'hello',
      caps: {
        webgpu: !!navigator.gpu,
        ua: navigator.userAgent,
        cores: navigator.hardwareConcurrency,
        memory_gb: navigator.deviceMemory || null,
        version: 1,
      },
      loaded: loadedMeta ? { model_url: loadedMeta.model_url, lora_key: loadedMeta.lora_key } : null,
    });
    log('connected to server');
  };
  ws.onclose = () => {
    setWsState('disconnected', false);
    log(`connection lost, retrying in ${reconnectDelay / 1000}s`);
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 10000);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (ev) => {
    let msg;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    handleMessage(msg);
  };
}

async function handleMessage(msg) {
  switch (msg.type) {
    case 'ping':
      send({ type: 'pong', t: msg.t });
      break;
    case 'dispatch':
      runJob(msg).catch((e) => {
        log(`job ${msg.job_id} error: ${e.message}`);
        send({ type: 'job_error', job_id: msg.job_id, message: String(e.message || e), code: e.code || 'worker_error' });
        setState('ready');
        send({ type: 'status', state: currentKey ? 'ready' : 'ready', loaded: loadedMeta ? { model_url: loadedMeta.model_url, lora_key: loadedMeta.lora_key } : null });
      });
      break;
    case 'cancel': {
      const ctrl = aborts.get(msg.job_id);
      if (ctrl) ctrl.abort();
      break;
    }
    default:
      break;
  }
}

async function runJob(dispatch) {
  const { job_id, model_url, lora, params } = dispatch;
  log(`job ${job_id}: model=${short(model_url)} lora=${(lora || []).length}`);
  setState('busy');
  send({ type: 'job_accepted', job_id });

  const key = makeKey(model_url, lora || []);
  if (key !== currentKey) {
    send({ type: 'progress', job_id, phase: 'loading', pct: 0 });
    await loadModel(model_url, lora || [], params.n_ctx || AUTO_N_CTX, (phase, pct) => {
      send({ type: 'progress', job_id, phase, pct });
    });
  }

  const ctrl = new AbortController();
  aborts.set(job_id, ctrl);
  let finishReason = 'stop';
  let usage = null;
  let timings = null;

  try {
    const response = await wllama.createChatCompletion({
      messages: params.messages,
      stream: true,
      max_tokens: params.max_tokens,
      temperature: params.temperature,
      top_p: params.top_p,
      top_k: params.top_k,
      stop: params.stop,
      seed: params.seed,
      abortSignal: ctrl.signal,
      cache_prompt: true,
      onData: (chunk) => {
        const delta = chunk?.choices?.[0]?.delta?.content;
        if (delta) send({ type: 'chunk', job_id, delta });
        if (chunk?.usage) usage = chunk.usage;
        if (chunk?.timings) timings = chunk.timings;
      },
    });
    finishReason = response?.choices?.[0]?.finish_reason || 'stop';
    usage = response?.usage || usage;
    timings = response?.timings || timings;
    if (!usage && timings) {
      usage = {
        prompt_tokens: timings.prompt_n ?? 0,
        completion_tokens: timings.predicted_n ?? 0,
        total_tokens: (timings.prompt_n ?? 0) + (timings.predicted_n ?? 0),
      };
    }
  } catch (e) {
    aborts.delete(job_id);
    if (e.name === 'AbortError' || /abort/i.test(String(e.message))) {
      send({ type: 'job_cancelled', job_id });
      setState('ready');
      return;
    }
    throw e;
  }
  aborts.delete(job_id);

  jobsDone++;
  $('stJobs').textContent = String(jobsDone);
  if (timings?.predicted_per_second) {
    log(`job ${job_id} done: ${finishReason}, ${timings.predicted_n} tok @ ${timings.predicted_per_second.toFixed(1)} tok/s`);
  } else {
    log(`job ${job_id} done: ${finishReason}`);
  }
  send({ type: 'job_done', job_id, finish_reason: finishReason, usage, timings });
  setState('ready');
  send({ type: 'status', state: 'ready', loaded: loadedMeta });
}

// heartbeat
setInterval(() => {
  if (ws && ws.readyState === 1) {
    send({
      type: 'heartbeat',
      state: currentKey ? 'ready' : 'ready',
      loaded: loadedMeta ? { model_url: loadedMeta.model_url, lora_key: loadedMeta.lora_key } : null,
      jobs_done: jobsDone,
    });
  }
}, 15000);

// keep-awake nudge: Safari suspends background tabs; a periodic no-op rAF keeps
// the page alive while visible, and we detect suspension via visibilitychange
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    log('page visible again — reconnecting if needed');
    if (!ws || ws.readyState > 1) connect();
  }
});

// ---------------------------------------------------------------------------
// Local test UI
// ---------------------------------------------------------------------------
$('btnLoad').onclick = async () => {
  const ref = $('testModel').value.trim();
  if (!ref) return alert('モデル URL を入力してください');
  try {
    await loadModel(ref, [], AUTO_N_CTX, (phase, pct) => log(`load ${phase}: ${pct}%`));
  } catch (e) {
    log(`load error: ${e.message}`);
    alert('ロード失敗: ' + e.message);
  }
};
$('btnUnload').onclick = () => unloadModel();
$('btnSend').onclick = async () => {
  const out = $('chatOut');
  out.textContent = '';
  if (!wllama) {
    out.textContent = '先にモデルをロードしてください (またはサーバー経由のジョブを待機)';
    return;
  }
  try {
    await wllama.createChatCompletion({
      messages: [{ role: 'user', content: $('testPrompt').value }],
      stream: true,
      max_tokens: 256,
      onData: (chunk) => {
        out.textContent += chunk?.choices?.[0]?.delta?.content || '';
      },
    });
  } catch (e) {
    out.textContent += `\n[error] ${e.message}`;
  }
};

// boot
connect();
log(`worker started (wllama@${WLLAMA_VERSION}, webgpu=${!!navigator.gpu}, cpu=${NO_GPU})`);
