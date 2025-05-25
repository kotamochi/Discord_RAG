<!-- README.md -->

# Discord情報検索Botツール (RAG)

## 概要

本リポジトリは、Discord サーバーのメッセージ履歴を知識ベースとして、質問応答 (Retrieval‑Augmented Generation) を行うボットです。
実装には **discord.py**、検索パイプラインには **LlamaIndex**、ベクトルストアには **Milvus**、応答生成には **Gemini 2.5 Flash** を採用しています。

```
┌──────────┐     ┌──────────┐    ┌──────────────┐     ┌──────────────────┐
│ Discord  │ ⇄   │ Bot (Py) │ →  │ LlamaIndex   │ →   │ Gemini 2.5 Flash │
└──────────┘     │ crawler │     │ Retriever    │     │  (google‑genai)  │
                 │ chunker │     │              │     └──────────────────┘
                 │ embedder│     └────┬─────────┘
                 └────┬────┘          │
                      │               ↓
                      │       ┌──────────────────┐
                      └──────→│ Milvus Vector DB │
                              └──────────────────┘
```

## スラッシュコマンド

| コマンド                                        | 説明                                                               |
| ------------------------------------------- | ---------------------------------------------------------------- |
| `/search <query> [channel] [user]` | チャット履歴を検索し、Gemini に要約・回答させます。|
| `/crawl_history [guild_id]` (管理者限定)         | メッセージ履歴をクローリング → クレンジング → チャンク分割 → ベクトル化                         |
| `/help`                                     | 利用可能なコマンドを Embed 形式で表示                                           |

## ディレクトリ構成

```
Discord_RAG/
├── main.py
├── src/
│   ├── config.py
│   ├── data_collection/
│   ├── data_processing/
│   ├── embedding/
│   ├── rag_pipeline/
│   └── vector_store/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── 要件定義書.md
```

## .env で設定する項目

```dotenv
# Discord
DISCORD_BOT_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxx
DISCORD_ADMIN_ID=123456789012345678  # 管理者ユーザー ID (任意)

# Gemini
GEMINI_API_KEY=AIzaSy...

# Milvus
MILVUS_HOST=milvus          # docker-compose 内でリンクしている場合はそのまま
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=sever_meglog_collection
# Vector DB のスキーマは src/vector_store/milvus_handler.py を参照

# RAG パラメータ
CHUNK_SIZE=1024             # 省略時は 1024
CHUNK_OVERLAP=20            # 〃 20
LLM_TEMPERATURE=0.0         # 〃 0.0
```

## ローカル実行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Docker 実行

```bash
docker compose up -d   # Milvus + Bot
docker compose logs -f app
```

## カスタマイズ

| 項目              | 変更方法                                         |
| --------------- | -------------------------------------------- |
| 埋め込みモデル         | `src/config.py` の `EMBEDDING_MODEL_NAME` を変更 |
| Gemini モデル      | `LLM_MODEL_NAME` を変更 (API 対応モデルを指定)          |
| チャンクサイズ・重複      | `.env` の `CHUNK_SIZE` / `CHUNK_OVERLAP`      |
| 検索件数 (Top‑k)    | `src/config.py` の `SIMILARITY_TOP_K`         |
| 収集メッセージ上限       | `MAX_MESSAGES_PER_CHANNEL`                   |
| 他 Vector DB を使用 | `vector_store/` に新ハンドラを実装                    |

## ライセンス

MIT License
