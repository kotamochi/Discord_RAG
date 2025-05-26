# Discord RAG Bot

Discordサーバー内の会話ログを活用したRAG (Retrieval Augmented Generation) ボットです。
ユーザーの質問に対し、サーバー内の過去の会話から関連情報を検索し、大規模言語モデル (LLM) を用いて回答を生成します。

## 主な機能

- **Discordメッセージ収集**: 指定されたDiscordサーバーのメッセージ履歴を収集します。
- **データ処理**:
    - **クリーニング**: ボットの発言やURLのみのメッセージなど、不要な情報を除去します。
    - **チャンキング**: メッセージを適切なサイズのチャンクに分割します。
- **埋め込みベクトル生成**: チャンク化されたメッセージをHuggingFaceの埋め込みモデル (e.g., `cl-nagoya/ruri-v3-310m`) を用いてベクトル化します。
- **ベクトルストア**: 埋め込みベクトルをMilvusベクトルストアに格納・管理します。
- **RAGパイプライン**:
    - ユーザーの質問を埋め込みベクトル化し、Milvusから関連情報を検索します。
    - 検索結果と元の質問をGoogleのGemini API (Function Calling利用) に渡し、自然な回答を生成します。
- **スラッシュコマンド**:
    - `/search`: サーバー内の情報を検索します。
    - `/crawl_history`: (管理者専用) サーバーのメッセージ履歴を収集・処理し、検索可能にします。
    - `/help`: ボットのヘルプ情報を表示します。
- **リアルタイム更新**: 新規メッセージも処理パイプラインを通じてMilvusに登録され、検索対象となります（クロール済みのサーバーのみ）。

## 技術スタック

- **言語**: Python
- **主要ライブラリ**:
    - `discord.py`: Discord APIラッパー
    - `pymilvus`: Milvusクライアント
    - `google-generativeai`: Gemini APIクライアント
    - `llama-index`: データフレームワーク (チャンキング、埋め込み連携など)
    - `sentence-transformers` / `HuggingFaceEmbedding`: テキスト埋め込み
    - `torch`: 機械学習ライブラリ
- **ベクトルデータベース**: Milvus
- **LLM**: Google Gemini
- **コンテナ技術**: Docker, Docker Compose

## ディレクトリ構成

```
.
├── .git/
├── .gitignore
├── crawled_guilds.json         # クロール済みDiscordサーバーIDリスト
├── data/                       # (Milvusデータ永続化用 - docker-compose.yml参照)
├── docker-compose.yml          # Docker Compose設定
├── Dockerfile                  # ボットアプリケーションのDockerfile
├── logs/                       # ボットのログファイル
├── main.py                     # Discordボットのメイン処理
├── README.md                   # このファイル
├── requirements.txt            # Python依存関係
└── src/
    ├── config.py               # 設定ファイル (APIキー, モデル名など)
    ├── data_collection/
    │   └── discord_crawler.py  # Discordメッセージ収集
    ├── data_processing/
    │   ├── cleaner.py          # メッセージクリーニング
    │   └── chunker.py          # メッセージチャンキング
    ├── embedding/
    │   └── embedder.py         # テキスト埋め込み処理
    ├── rag_pipeline/
    │   └── rag_pipe_handler.py # RAGパイプライン処理
    └── vector_store/
        └── milvus_handler.py   # Milvus連携処理
```

## セットアップ

### 1. 前提条件

- Docker および Docker Compose がインストールされていること。
- Discordボットが作成され、ボットトークンが取得済みであること。
- Google AI Studio などで Gemini API キーが取得済みであること。

### 2. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成し、以下の情報を記述します。

```env
DISCORD_BOT_TOKEN=YOUR_DISCORD_BOT_TOKEN
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
DISCORD_ADMIN_ID=YOUR_DISCORD_USER_ID  # ボットの管理者ユーザーID (エラー通知などに使用)

# 以下は必要に応じて変更
# MILVUS_HOST=milvus  # docker-compose.ymlで定義されたサービス名
# MILVUS_PORT=19530
# MILVUS_COLLECTION_NAME=sever_meglog_collection
# EMBEDDING_MODEL_NAME=cl-nagoya/ruri-v3-310m
# CHUNK_SIZE=1024
# CHUNK_OVERLAP=20
# LLM_TEMPERATURE=0.0
```

### 3. 起動

プロジェクトルートで以下のコマンドを実行します。

```bash
docker-compose up -d --build
```

これにより、MilvusサーバーとDiscordボットアプリケーションがコンテナとして起動します。

### 4. Discordサーバーへのボット招待

作成したDiscordボットを、メッセージを収集・検索したいサーバーへ招待してください。
その際、以下の権限が必要となります（最低限）：
- メッセージを読む
- メッセージを送信する
- スラッシュコマンドを使用する
- (チャンネルによっては) チャンネルを見る
- (スレッドを利用する場合) スレッドを見る

## 使用方法

### 1. 履歴のクロール (管理者のみ)

ボットを導入したサーバーで、管理者ユーザーが以下のスラッシュコマンドを実行します。

```
/crawl_history
```

オプションで `guild_id` を指定することも可能です（ボットが複数のサーバーに参加している場合）。
これにより、サーバー内の過去のメッセージが収集・処理され、Milvusに登録されます。処理には時間がかかる場合があります。
進捗はコマンドを実行したチャンネルに通知されます。

一度クロールが完了したサーバーの情報は `crawled_guilds.json` に保存され、ボット再起動後も記憶されます。
再クロール（既存データを削除して再登録）もこのコマンドで行います。

### 2. 情報検索

クロール済みのサーバーであれば、どのユーザーでも以下のスラッシュコマンドで情報を検索できます。

```
/search query:<あなたの質問>
```

例:
```
/search query:このチャンネルの概要を教えて channel:#チャンネル名
/search query:○年○月○日のタスクって何だっけ？ user:@自分のユーザー名
/search query:話し合いの内容を纏めて
```

オプションで以下のパラメータを指定して検索範囲を絞り込めます。
- `channel`: 特定のチャンネル名またはID
- `user`: 特定のユーザー名またはID
- `date_from`: 検索開始日 (YYYY-MM-DD形式)
- `date_to`: 検索終了日 (YYYY-MM-DD形式)

### 3. ヘルプ

ボットが提供するコマンドの一覧や使用方法は、以下のコマンドで確認できます。

```
/help
```

## 注意事項

- 大量のメッセージを処理する場合、Milvusのデータ量やリソース消費に注意してください。
- `EMBEDDING_MODEL_NAME` を変更した場合、`src/config.py` の `EMBEDDING_MODEL_DIMENSION` も対応する次元数に更新し、既存のMilvusコレクションを再作成（またはデータをクリアして再クロール）する必要がある場合があります。
- エラーや警告は `logs/bot.log` に出力されます。また、重要なエラーは環境変数 `DISCORD_ADMIN_ID` で指定されたユーザーにDMで通知されます。
