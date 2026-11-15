# iPad LLM API Server

余っている A16 iPad の GPU(WebGPU)で llama.cpp(wllama)を動かし、OpenAI 互換 API として結果を返す分散推論システム。Render にホスティングし、iPad はブラウザ(PWA)を開くだけの「推論ワーカー」になります。

```
クライアント (OpenAI SDK / curl / デモUI)
      │ POST /v1/chat/completions   ← model: HF URL / org/repo/file.gguf、lora: アダプタ指定
      ▼
Render: Node.js サーバー (このリポジトリ)
      │  ・OpenAI 互換 API / ジョブキュー / ワーカー管理
      │  ・"org/repo" 等の表記を具体的な GGUF URL に解決
      ▼ WebSocket (wss)
iPad Safari ワーカー (/worker)
      ・wllama (llama.cpp / WebGPU) でブラウザ内推論
      ・モデルは初回のみダウンロード → OPFS/Cache にキャッシュ
      ・トークンをストリーミングでサーバーへ返す → SSE でクライアントへ
```

## ファイル構成

| パス | 内容 |
|---|---|
| `server.js` | API サーバー本体(ジョブ管理 / WS ハブ / SSE 中継) |
| `lib/hf-resolve.js` | モデル参照の解決(URL / `org/repo/file.gguf` / `org/repo`) |
| `public/index.html` | デモクライアント(チャット + サーバー状態表示) |
| `public/worker/` | iPad ワーカーページ(wllama 統合、LoRA 対応) |
| `scripts/e2e-test.mjs` | E2E テスト(偽ワーカーで API 全面検証、36 ケース) |
| `render.yaml` | Render Blueprint |

## Render へのデプロイ

1. このリポジトリを GitHub に push
2. Render Dashboard → **New → Blueprint** → リポジトリを選択(`render.yaml` を自動認識)
3. 環境変数は Blueprint で自動生成(API_KEY / WORKER_TOKEN は Render の Environment タブで確認)

手動デプロイの場合:

```bash
npm install
API_KEY=your-key WORKER_TOKEN=ipad-secret npm start   # ローカル
```

## iPad のセットアップ

1. Safari(iPadOS 26 以降、WebGPU 標準有効)で `https://<your-service>.onrender.com/worker?token=<WORKER_TOKEN>` を開く
2. **ホーム画面に追加**(PWA 化)し、追加したアイコンから起動
3. 設定アプリで「画面の自動ロック → なし」+ 常時充電。バックグラウンドに回ると推論が止まるため、**ワーカータブを常に前面**にしておく
4. 初回ジョブ時にモデルがダウンロードされキャッシュされる(2 回目以降は再ダウンロードなし)

URL パラメータ:

| パラメータ | 説明 |
|---|---|
| `token` | WORKER_TOKEN(必須、サーバー側で設定時) |
| `n_ctx` | コンテキスト長の既定値(既定 2048、ジョブごとに上書き可) |
| `cpu` | `1` で WebGPU を無効化(CPU WASM 強制、デバッグ用) |

## API

### POST /v1/chat/completions(OpenAI 互換 + 拡張)

```bash
curl -N https://<your-service>.onrender.com/v1/chat/completions \
  -H "authorization: Bearer $API_KEY" \
  -H "content-type: application/json" \
  -d '{
    "model": "ggml-org/gemma-3-270m-it-Q4_K_M-GGUF",
    "messages": [{"role": "user", "content": "こんにちは"}],
    "stream": true,
    "max_tokens": 512,
    "temperature": 0.7,
    "lora": [{"url": "https://example.com/my-adapter.gguf", "scale": 0.9}]
  }'
```

**`model` の指定方法(いずれも可)**

| 表記 | 動作 |
|---|---|
| `https://.../model.gguf` | その URL を直接使用(HF 以外も可) |
| `org/repo/model.gguf` | `https://huggingface.co/org/repo/resolve/main/model.gguf` に解決 |
| `org/repo` | HF API でリポジトリ内の `.gguf` を一覧し、iPad 向け量子化(Q4_K_M 等)を自動選択。分割シャードのみの場合は 400 エラー(明示指定が必要) |
| HF の blob/resolve ページ URL | 自動正規化 |

**`lora`(拡張フィールド)**

- `string` / `{repo, file, scale}` / それらの配列(最大 4 個)を `model` と同じ表記で指定
- `scale` はアダプタ適用強度(既定 1.0)。PEFT の学習設定で `lora_alpha != rank` の場合は `alpha / rank` を指定してください
- 実装は wllama のネイティブ `lora_adapters`(llama.cpp の LoRA 適用)を使用。**ベースモデルは単一ファイル GGUF である必要があります**(分割モデル + LoRA は未対応)

**その他の拡張フィールド**: `n_ctx`(コンテキスト長、最大 8192)、`chat_template`(テンプレート上書き)、`top_k`、`seed`。

### GET /v1/models
接続中ワーカーがロードしているモデル + `DEFAULT_MODEL` の一覧。

### GET /admin/status
ワーカーの一覧(状態 / WebGPU 有無 / ロード済みモデル / 最終応答)とジョブ状況。

### WS /worker(ワーカー用)
プロトコル: `hello` / `status` / `job_accepted` / `progress` / `chunk` / `job_done` / `job_error` / `job_cancelled` / `heartbeat` ⇄ `dispatch` / `cancel` / `ping`。

## 環境変数

| 変数 | 既定 | 説明 |
|---|---|---|
| `PORT` | 8080 | ポート |
| `API_KEY` | (なし) | 設定時、クライアントは `Authorization: Bearer <key>` 必須 |
| `WORKER_TOKEN` | (なし) | 設定時、ワーカーは `?token=` 必須 |
| `DEFAULT_MODEL` | (なし) | リクエストで `model` 省略時に使用 |
| `MAX_TOKENS_CAP` | 2048 | `max_tokens` のハード上限 |
| `MAX_QUEUE` | 32 | 待ち行列上限(超過で 503) |
| `N_CTX` | 2048 | リクエストで `n_ctx` 未指定時のコンテキスト長 |
| `HF_TOKEN` | (なし) | gated リポジトリを解決する場合 |

## テスト

```bash
npm install
npm test   # 偽ワーカーを接続し、ストリーミング/LoRA中継/キュー/切断系を36ケース検証
```

## 既知の制限と注意

- **wllama バージョンは 3.13.0 にピン留め**(jsDelivr CDN から読み込み)。更新時は iPad 実機での動作確認が必要
- Safari は互換ビルドへ自動フォールバック(wllama 内蔵)。iPadOS 26 + JSPI 対応 Safari では通常ビルドが使われる
- モデルの同時ロードは 1 個/ワーカー。別モデルのジョブが来ると unload → load が走る
- Render の無料プランはスピンダウンするため、常時運用は有料プラン推奨
- Render は QUIC/HTTP3 非対応のため WebTransport は不使用(WebSocket のみ。WebRTC は将来拡張)
- 推論性能の目安: A16(iPad 10 世代/iPad mini 7)で 0.6B〜1B Q4_K_M が現実的なレンジ
