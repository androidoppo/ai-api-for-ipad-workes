/**
 * HuggingFace model / LoRA reference resolution.
 *
 * Accepts any of:
 *   - full URL:            https://huggingface.co/org/repo/resolve/main/model.gguf
 *   - any http(s) URL:     https://example.com/model.gguf
 *   - repo/file shorthand: org/repo/model.gguf
 *   - bare repo id:        org/repo   (smallest single .gguf in the repo is auto-picked)
 *
 * Returns a concrete { url, ... } object. Resolutions are cached in memory (TTL 10 min).
 */

const RESOLVE_TTL_MS = 10 * 60 * 1000;
const cache = new Map(); // input -> { at, value }
const HF_HOSTS = new Set(['huggingface.co', 'hf.co', 'www.huggingface.co']);

function cacheGet(key) {
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < RESOLVE_TTL_MS) return hit.value;
  if (hit) cache.delete(key);
  return null;
}

function cacheSet(key, value) {
  cache.set(key, { at: Date.now(), value });
}

async function fetchJson(url, fetchImpl, timeoutMs = 15000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetchImpl(url, {
      signal: ctrl.signal,
      headers: { accept: 'application/json' },
      // HF token (optional) for gated repos
      ...(process.env.HF_TOKEN
        ? { headers: { accept: 'application/json', authorization: `Bearer ${process.env.HF_TOKEN}` } }
        : {}),
    });
    if (!res.ok) {
      throw new Error(`HuggingFace API returned HTTP ${res.status} for ${url}`);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

const SHARD_RE = /-\d{5}-of-\d{5}\.gguf$/i;

/** Returns [repo, file] if input looks like "org/repo/file.gguf" (or with subfolder), else null. */
function parseRepoFileShorthand(input) {
  const parts = input.split('/').filter(Boolean);
  if (parts.length >= 3 && /\.gguf$/i.test(input)) {
    const repo = parts.slice(0, 2).join('/');
    const file = parts.slice(2).join('/');
    return [repo, file];
  }
  return null;
}

/** Extracts a resolvable target from a full HF URL ("blob", "resolve" page URLs are normalized). */
function parseHfUrl(input) {
  let u;
  try {
    u = new URL(input);
  } catch {
    return null;
  }
  if (!HF_HOSTS.has(u.hostname)) return null;
  const seg = u.pathname.split('/').filter(Boolean); // [org, repo, (blob|resolve), branch?, ...file]
  if (seg.length < 2) return null;
  const repo = seg.slice(0, 2).join('/');
  let file = null;
  if (seg[2] === 'resolve' || seg[2] === 'blob') {
    // /org/repo/resolve/main/path/to/file.gguf
    if (seg.length >= 4) file = seg.slice(4).join('/');
  } else if (/\.gguf$/i.test(seg[seg.length - 1] || '')) {
    // /org/repo/file.gguf treated as repo-root file
    file = seg.slice(2).join('/');
  }
  if (file && !/\.gguf$/i.test(file)) return { repo, file: null, repoUrl: null, error: `URL does not point to a .gguf file: ${input}` };
  return { repo, file, repoUrl: file ? null : `https://huggingface.co/${repo}` };
}

function fileUrl(repo, file, branch = 'main') {
  return `https://huggingface.co/${repo}/resolve/${branch}/${file
    .split('/')
    .map(encodeURIComponent)
    .join('/')}`;
}

/** Pick a .gguf from a repo: single file wins; shards rejected; multiple candidates -> smallest. */
function pickGgufFromSiblings(siblings) {
  const ggufs = (siblings || [])
    .map((s) => s.rfilename)
    .filter((n) => /\.gguf$/i.test(n));
  if (ggufs.length === 0) {
    return { error: 'No .gguf file found in this repository. Pass an explicit URL or "org/repo/file.gguf".' };
  }
  const singles = ggufs.filter((n) => !SHARD_RE.test(n));
  if (singles.length === 1 && ggufs.length === 1) return { file: singles[0] };
  if (singles.length === 0) {
    return {
      error:
        'Repository contains split GGUF shards. LoRA and auto-pick need a single-file GGUF; pass an explicit file (first shard URL works for split models without LoRA): e.g. "org/repo/' +
        ggufs[0] +
        '"',
    };
  }
  // multiple singles: prefer iPad-friendly quants, then smaller-looking files
  const sorted = singles.slice().sort((a, b) => {
    const qa = quantRank(a) - quantRank(b);
    if (qa !== 0) return qa;
    return a.length - b.length || a.localeCompare(b);
  });
  return { file: sorted[0], alternatives: singles };
}

/** Preference rank by quantization hint embedded in the filename (lower = better for mobile GPU). */
const QUANT_ORDER = ['q4_k_m', 'q4_k_s', 'q4_k', 'iq4_xs', 'q4_0', 'q3_k_m', 'q3_k', 'iq3_m', 'q2_k', 'q5_k_m', 'q5_k', 'q5_0', 'q6_k', 'q8_0', 'f16', 'bf16', 'f32'];
function quantRank(name) {
  const n = name.toLowerCase();
  for (let i = 0; i < QUANT_ORDER.length; i++) {
    if (n.includes(QUANT_ORDER[i])) return i;
  }
  return QUANT_ORDER.length;
}

/**
 * Resolve a model reference into a loadable URL (string).
 * @param {string} input
 * @param {{fetchImpl?: typeof fetch}} opts
 * @returns {Promise<{url: string, source: string, alternatives?: string[]}>}
 */
export async function resolveModel(input, opts = {}) {
  const fetchImpl = opts.fetchImpl || fetch;
  if (typeof input !== 'string' || !input.trim()) {
    throw new Error('"model" is required (URL, "org/repo/file.gguf", or "org/repo").');
  }
  input = input.trim();

  const cached = cacheGet('m:' + input);
  if (cached) return cached;

  let result;
  if (/^https?:\/\//i.test(input)) {
    const hf = parseHfUrl(input);
    if (hf) {
      if (hf.error) throw new Error(hf.error);
      if (hf.file) {
        result = { url: fileUrl(hf.repo, hf.file), source: input };
      } else {
        // repo page URL -> auto pick
        const info = await fetchJson(`https://huggingface.co/api/models/${hf.repo}`, fetchImpl);
        const picked = pickGgufFromSiblings(info.siblings);
        if (picked.error) throw new Error(picked.error);
        result = { url: fileUrl(hf.repo, picked.file), source: input, ...(picked.alternatives ? { alternatives: picked.alternatives } : {}) };
      }
    } else {
      if (!/\.gguf($|\?)/i.test(input)) {
        throw new Error(`Model URL must point to a .gguf file: ${input}`);
      }
      result = { url: input, source: 'url' };
    }
  } else {
    const shorthand = parseRepoFileShorthand(input);
    if (shorthand) {
      const [repo, file] = shorthand;
      result = { url: fileUrl(repo, file), source: input };
    } else {
      // bare repo id -> auto pick via HF API
      const apiPath = input.split('/').map(encodeURIComponent).join('/');
      const info = await fetchJson(`https://huggingface.co/api/models/${apiPath}`, fetchImpl);
      const picked = pickGgufFromSiblings(info.siblings);
      if (picked.error) throw new Error(picked.error);
      result = { url: fileUrl(input, picked.file), source: input, ...(picked.alternatives ? { alternatives: picked.alternatives } : {}) };
    }
  }

  cacheSet('m:' + input, result);
  return result;
}

/**
 * Resolve a LoRA reference into { url, scale }.
 * @param {string|{url?:string, repo?:string, file?:string, scale?:number, strength?:number}} lora
 */
export async function resolveLora(lora, opts = {}) {
  const fetchImpl = opts.fetchImpl || fetch;
  let ref = lora;
  let scale;
  if (lora && typeof lora === 'object' && !Array.isArray(lora)) {
    ref = lora.url || (lora.repo ? `${lora.repo}${lora.file ? '/' + lora.file : ''}` : null);
    scale = lora.scale ?? lora.strength;
  }
  if (typeof ref !== 'string' || !ref.trim()) {
    throw new Error('Invalid "lora" entry: pass a URL, "org/repo/file.gguf", or {repo, file, scale}.');
  }
  ref = ref.trim();
  if (scale !== undefined) {
    if (typeof scale !== 'number' || !Number.isFinite(scale) || scale <= 0 || scale > 4) {
      throw new Error('LoRA "scale" must be a number in (0, 4].');
    }
  }
  const m = await resolveModel(ref, { fetchImpl });
  const out = { url: m.url, scale: scale ?? 1.0, source: ref };
  return out;
}

/** Normalize the request "lora" field: string | object | array of either. Returns array. */
export async function resolveLoraList(lora, opts = {}) {
  if (lora == null) return [];
  const items = Array.isArray(lora) ? lora : [lora];
  if (items.length > 4) throw new Error('At most 4 LoRA adapters per request.');
  const out = [];
  for (const item of items) out.push(await resolveLora(item, opts));
  return out;
}

/** Test helper */
export function _clearResolveCache() {
  cache.clear();
}
