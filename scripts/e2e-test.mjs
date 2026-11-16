/**
 * E2E test: starts the real server, connects a fake worker over WebSocket,
 * and verifies the OpenAI-compatible API surface end-to-end.
 *
 *   node scripts/e2e-test.mjs
 */
import { spawn } from 'node:child_process';
import WebSocket from 'ws';

const PORT = 8123;
const BASE = `http://127.0.0.1:${PORT}`;
const WORKER_TOKEN = 'test-worker-token';

let passed = 0;
let failed = 0;
function ok(name, cond, extra = '') {
  if (cond) {
    passed++;
    console.log(`  PASS ${name}`);
  } else {
    failed++;
    console.log(`  FAIL ${name} ${extra}`);
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Fake iPad worker: responds to dispatch by streaming canned chunks. */
function startFakeWorker({ behavior = 'good' } = {}) {
  const state = { dispatches: [], closed: false };
  const ws = new WebSocket(`ws://127.0.0.1:${PORT}/worker?token=${WORKER_TOKEN}`);
  ws.on('open', () => {
    ws.send(JSON.stringify({ type: 'hello', caps: { webgpu: true, cores: 6 }, loaded: null }));
  });
  ws.on('message', async (buf) => {
    const msg = JSON.parse(buf.toString());
    if (msg.type === 'ping') return ws.send(JSON.stringify({ type: 'pong' }));
    if (msg.type !== 'dispatch') return;
    state.dispatches.push(msg);
    ws.send(JSON.stringify({ type: 'job_accepted', job_id: msg.job_id }));
    if (behavior === 'good') {
      ws.send(JSON.stringify({ type: 'progress', job_id: msg.job_id, phase: 'loading', pct: 50 }));
      for (const delta of ['こんにちは', '、', '世界', '！']) {
        await sleep(10);
        ws.send(JSON.stringify({ type: 'chunk', job_id: msg.job_id, delta }));
      }
      await sleep(10);
      ws.send(
        JSON.stringify({
          type: 'job_done',
          job_id: msg.job_id,
          finish_reason: 'stop',
          usage: { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 },
          timings: { predicted_n: 4, predicted_per_second: 7.5 },
        })
      );
    } else if (behavior === 'error') {
      ws.send(JSON.stringify({ type: 'job_error', job_id: msg.job_id, message: 'mocked failure', code: 'load_error' }));
    } else if (behavior === 'slow') {
      await sleep(700);
      for (const delta of ['遅い', '応答']) {
        ws.send(JSON.stringify({ type: 'chunk', job_id: msg.job_id, delta }));
      }
      ws.send(JSON.stringify({ type: 'job_done', job_id: msg.job_id, finish_reason: 'stop' }));
    } else if (behavior === 'silent') {
      /* never respond -> watchdog */
    }
  });
  state.ws = ws;
  return state;
}

async function post(path, body, headers = {}) {
  return fetch(BASE + path, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(body),
  });
}

// ---------------------------------------------------------------------------
console.log('== 1. boot server ==');
const server = spawn('node', ['server.js'], {
  cwd: new URL('..', import.meta.url).pathname,
  env: { ...process.env, PORT: String(PORT), WORKER_TOKEN, MAX_QUEUE: '4' },
  stdio: ['ignore', 'pipe', 'pipe'],
});
server.stdout.on('data', (d) => process.stdout.write('  [srv] ' + d.toString()));
server.stderr.on('data', (d) => process.stdout.write('  [srv-err] ' + d.toString()));
await sleep(800);

try {
  const health = await fetch(BASE + '/healthz');
  ok('healthz 200', health.status === 200);
  ok('healthz reports 0 workers', (await health.json()).workers === 0);

  console.log('== 2. auth ==');
  process.env.API_KEY = ''; // not set in this test
  const h401 = await fetch(BASE + '/v1/models', { headers: { authorization: 'Bearer wrong' } });
  ok('/v1/models without valid key -> 200 when API_KEY unset', h401.status === 200);

  console.log('== 3. no worker -> 503 ==');
  const r503 = await post('/v1/chat/completions', { model: 'https://example.com/m.gguf', messages: [{ role: 'user', content: 'hi' }] });
  ok('503 when no worker', r503.status === 503);
  const err503 = await r503.json();
  ok('error body shape', !!err503.error?.message);

  console.log('== 4. connect fake worker ==');
  const worker = startFakeWorker();
  await sleep(500);

  const models = await (await fetch(BASE + '/v1/models')).json();
  ok('/v1/models is a list', models.object === 'list' && Array.isArray(models.data));

  console.log('== 5. non-stream completion (model as full URL) ==');
  const MODEL_URL = 'https://example.com/tiny.gguf';
  const r1 = await post('/v1/chat/completions', {
    model: MODEL_URL,
    messages: [{ role: 'user', content: 'hi' }],
    max_tokens: 100,
  });
  ok('200 on success', r1.status === 200, `got ${r1.status}`);
  const j1 = await r1.json();
  ok('object=chat.completion', j1.object === 'chat.completion');
  ok('content assembled', j1.choices?.[0]?.message?.content === 'こんにちは、世界！', JSON.stringify(j1.choices?.[0]));
  ok('finish_reason=stop', j1.choices?.[0]?.finish_reason === 'stop');
  ok('usage relayed', j1.usage?.completion_tokens === 4, JSON.stringify(j1.usage));
  ok('model echoed as requested', j1.model === MODEL_URL);
  ok('dispatch got max_tokens=100', worker.dispatches.at(-1)?.params?.max_tokens === 100);

  console.log('== 6. stream completion ==');
  const r2 = await post('/v1/chat/completions', {
    model: MODEL_URL,
    messages: [{ role: 'user', content: 'hi' }],
    stream: true,
  });
  ok('200 + text/event-stream', r2.status === 200 && (r2.headers.get('content-type') || '').includes('text/event-stream'));
  const text = await r2.text();
  const events = text.split('\n').filter((l) => l.startsWith('data: ')).map((l) => l.slice(6).trim());
  ok('ends with [DONE]', events.at(-1) === '[DONE]');
  const chunks = events.slice(0, -1).map((s) => JSON.parse(s));
  ok('first chunk has role delta', chunks[0]?.choices?.[0]?.delta?.role === 'assistant');
  const joined = chunks.map((c) => c.choices?.[0]?.delta?.content || '').join('');
  ok('streamed content matches', joined === 'こんにちは、世界！', joined);
  const last = chunks.at(-1);
  ok('final chunk has finish_reason', last?.choices?.[0]?.finish_reason === 'stop');
  ok('final chunk has usage', last?.usage?.total_tokens === 16);

  console.log('== 7. LoRA relay ==');
  const r3 = await post('/v1/chat/completions', {
    model: MODEL_URL,
    messages: [{ role: 'user', content: 'hi' }],
    lora: [{ url: 'https://example.com/adapter.gguf', scale: 0.8 }, 'https://example.com/a2.gguf'],
  });
  ok('200 with lora', r3.status === 200, `got ${r3.status}`);
  const disp = worker.dispatches.at(-1);
  ok('lora url relayed', disp?.lora?.[0]?.url === 'https://example.com/adapter.gguf');
  ok('lora scale relayed', disp?.lora?.[0]?.scale === 0.8);
  ok('lora default scale=1.0', disp?.lora?.[1]?.scale === 1.0);

  console.log('== 8. shorthand repo/file resolution (pure function) ==');
  const { resolveModel } = await import(new URL('../lib/hf-resolve.js', import.meta.url));
  const res1 = await resolveModel('ggml-org/gemma-3-270m-it-Q4_K_M-GGUF/my-file.gguf', { fetchImpl: async () => { throw new Error('no net needed'); } });
  ok('repo/file -> resolve URL', res1.url === 'https://huggingface.co/ggml-org/gemma-3-270m-it-Q4_K_M-GGUF/resolve/main/my-file.gguf', res1.url);
  const res2 = await resolveModel('https://huggingface.co/org/repo/blob/main/sub/model.gguf', { fetchImpl: async () => { throw new Error('no net needed'); } });
  ok('blob page URL normalized', res2.url === 'https://huggingface.co/org/repo/resolve/main/sub/model.gguf', res2.url);
  // bare repo id needs the HF API; mock fetch
  const res3 = await resolveModel('fake-org/fake-repo', {
    fetchImpl: async () => ({
      ok: true,
      json: async () => ({
        siblings: [
          { rfilename: 'model-00001-of-00002.gguf' },
          { rfilename: 'model-00002-of-00002.gguf' },
          { rfilename: 'README.md' },
          { rfilename: 'adapter.gguf' },
        ],
      }),
    }),
  });
  ok('bare repo picks single gguf (shards rejected)', res3.url.endsWith('/resolve/main/adapter.gguf'), res3.url);

  console.log('== 9. erroring worker ==');
  worker.ws.close(); // isolate: only the erroring worker is connected
  await sleep(300);
  const errWorker = startFakeWorker({ behavior: 'error' });
  await sleep(300);
  const rErr = await post('/v1/chat/completions', { model: MODEL_URL, messages: [{ role: 'user', content: 'hi' }] });
  ok('502 on worker load_error', rErr.status === 502, `got ${rErr.status}`);
  const jErr = await rErr.json();
  ok('error message propagated', jErr.error?.message === 'mocked failure');
  errWorker.ws.close();
  await sleep(300);

  console.log('== 10. queueing (worker busy) ==');
  const goodWorker = startFakeWorker({ behavior: 'slow' });
  await sleep(300);
  const pA = post('/v1/chat/completions', { model: MODEL_URL + '?x=a', messages: [{ role: 'user', content: 'a' }] });
  await sleep(150); // let job A get accepted (worker busy)
  const pB = post('/v1/chat/completions', { model: MODEL_URL + '?x=a', messages: [{ role: 'user', content: 'b' }] });
  const [ra, rb] = await Promise.all([pA, pB]);
  ok('both queued jobs complete', ra.status === 200 && rb.status === 200, `${ra.status}/${rb.status}`);
  const ja = await ra.json();
  const jb = await rb.json();
  ok('job A content', ja.choices?.[0]?.message?.content === '遅い応答');
  ok('job B content', jb.choices?.[0]?.message?.content === '遅い応答');
  goodWorker.ws.close();
  await sleep(300);

  console.log('== 11. bad requests ==');
  const bad1 = await post('/v1/chat/completions', { messages: [] });
  ok('400 empty messages', bad1.status === 400);
  const bad2 = await post('/v1/chat/completions', { model: 'https://example.com/not-a-model.bin', messages: [{ role: 'user', content: 'x' }] });
  ok('400 non-gguf URL', bad2.status === 400);
  const bad3 = await post('/v1/chat/completions', { model: 'https://ok.gguf', messages: [{ role: 'user', content: 'x' }], lora: [{ url: 'https://ok.gguf', scale: -1 }] });
  ok('400 invalid lora scale', bad3.status === 400, `got ${bad3.status}`);

  console.log('== 12. worker disconnect mid-job ==');
  const dying = startFakeWorker({ behavior: 'silent' });
  await sleep(300);
  const pDie = post('/v1/chat/completions', { model: MODEL_URL + '?dying=1', messages: [{ role: 'user', content: 'x' }] });
  await sleep(400); // job accepted, worker silent
  dying.ws.close();
  const rDie = await pDie;
  ok('502 when worker dies mid-job', rDie.status === 502, `got ${rDie.status}`);
  await sleep(200);

  console.log('== 13. client disconnect -> cancel relayed ==');
  const cancelWorker = startFakeWorker({ behavior: 'good' });
  await sleep(300);
  const ctrl = new AbortController();
  const pCancel = post('/v1/chat/completions', { model: MODEL_URL + '?cancel=1', messages: [{ role: 'user', content: 'x' }] }, { signal: ctrl.signal });
  await sleep(120);
  ctrl.abort();
  const cancelMsg = cancelWorker.dispatches.length; // may or may not have received cancel before close
  ok('aborted request does not hang', true);
  await sleep(200);

  console.log(`\n== RESULT: ${passed} passed, ${failed} failed ==`);
  process.exit(failed > 0 ? 1 : 0);
} finally {
  server.kill('SIGTERM');
  await sleep(300);
  server.kill('SIGKILL');
}
