import discord
from discord import app_commands
import asyncio
import logging
import os
from datetime import datetime
import traceback
import json
from typing import Optional, Set

from src.config import ( # 設定値をまとめてインポート
    DISCORD_BOT_TOKEN, # ボットトークン
    DISCORD_ADMIN_ID, # 管理者ID
    EMBEDDING_MODEL_NAME, # Embedder, MilvusHandlerで使用
    CHUNK_SIZE,           # Chunkerで使用
    CHUNK_OVERLAP,        # Chunkerで使用
    MAX_MESSAGES_PER_CHANNEL, # DiscordCrawlerで使用
    SIMILARITY_TOP_K, # RAGPipelineHandlerで使用
    CRAWLED_GUILDS_FILE, # クロール済みギルドファイル
    MILVUS_COLLECTION_NAME # Milvusコレクション名
)
from src.rag_pipeline.rag_pipe_handler import RAGPipelineHandler
from src.data_collection.discord_crawler import DiscordCrawler
from src.data_processing.cleaner import clean_messages
from src.data_processing.chunker import chunk_messages
from src.embedding.embedder import MessageEmbedder
from src.vector_store.milvus_handler import MilvusHandler



# ロガー設定
# ログディレクトリが存在しない場合は作成
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "bot.log"))
    ]
)
logger = logging.getLogger(__name__)

# discordクライアント設定
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client) # CommandTreeを初期化
crawled_guild_ids: Set[int] = set() # クロール済みギルドIDを保持するセット


try:
    # 埋め込みモデルの初期化
    logger.info("Initializing MessageEmbedder globally...")
    message_embedder = MessageEmbedder(model_name=EMBEDDING_MODEL_NAME)
    logger.info("MessageEmbedder initialized successfully globally.")

except Exception as e:
    message_embedder = None # 初期化失敗時はNoneに設定
    logger.critical(f"Failed to initialize MessageEmbedder globally: {e}", exc_info=True)

try:
    logger.info(f"Initializing RAG Pipeline Handler with Milvus collection: {MILVUS_COLLECTION_NAME}...")
    rag_handler = RAGPipelineHandler(
        milvus_collection_name=MILVUS_COLLECTION_NAME,
        embedding_model=message_embedder,
        similarity_top_k=SIMILARITY_TOP_K
    )
    logger.info(f"RAG Pipeline Handler initialized successfully with collection: {rag_handler.milvus_collection_name}")

except Exception as e:
    rag_handler = None
    logger.error(f"Failed to initialize RAG Pipeline Handler: {e}", exc_info=True)


async def close_bot():
    """botを終了させる"""
    if client:
        logger.info("Closing bot...")
        await asyncio.create_task(client.close())


def load_crawled_guilds():
    """クロール済みギルドIDをファイルから読み込む"""
    global crawled_guild_ids
    try:
        if os.path.exists(CRAWLED_GUILDS_FILE):
            with open(CRAWLED_GUILDS_FILE, 'r') as f:
                data = json.load(f)
                crawled_guild_ids = set(data.get("crawled_ids", []))
                logger.info(f"Loaded {len(crawled_guild_ids)} crawled guild IDs from {CRAWLED_GUILDS_FILE}.")
        else:
            logger.info(f"{CRAWLED_GUILDS_FILE} not found. Initializing with empty set of crawled guilds.")
            crawled_guild_ids = set()
    except Exception as e:
        logger.error(f"Error loading crawled guilds file ({CRAWLED_GUILDS_FILE}): {e}", exc_info=True)
        crawled_guild_ids = set() # エラー時は空セットで初期化


def save_crawled_guild(guild_id: int):
    """クロール完了したギルドIDをセットとファイルに保存する"""
    global crawled_guild_ids
    if guild_id not in crawled_guild_ids:
        crawled_guild_ids.add(guild_id)
        try:
            with open(CRAWLED_GUILDS_FILE, 'w') as f:
                json.dump({"crawled_ids": list(crawled_guild_ids)}, f, indent=4)
            logger.info(f"Guild ID {guild_id} marked as crawled and saved to {CRAWLED_GUILDS_FILE}.")
        except Exception as e:
            logger.error(f"Error saving crawled guilds file ({CRAWLED_GUILDS_FILE}): {e}", exc_info=True)


async def notify_admin_on_startup_error(client: discord.Client, component_name: str, error: Exception):
    """起動時のコンポーネント初期化失敗を管理者にDMで通知する"""
    try:
        admin_user = await client.fetch_user(int(DISCORD_ADMIN_ID))
        if admin_user:
            error_message = f"[Bot Critical Error] Failed to initialize {component_name} on startup: {error}. Bot may not function correctly."
            await admin_user.send(error_message)
            logger.info(f"Sent startup error notification to admin for {component_name}.")
    except Exception as notify_e:
        logger.error(f"Failed to send startup error notification to admin for {component_name}: {notify_e}", exc_info=True)


@client.event
async def on_ready():
    global rag_handler # message_embedder はグローバルで初期化済みなので削除
    logger.info(f'Logged in as {client.user.name} (ID: {client.user.id})')
    logger.info('Bot is ready!')
    logger.info(f"Admin user ID for notifications: {DISCORD_ADMIN_ID}")

    load_crawled_guilds() # クロール済みギルドIDをロード

    # MessageEmbedderが初期化失敗している場合は、ここで管理者通知
    if message_embedder is None and DISCORD_ADMIN_ID:
        await notify_admin_on_startup_error(client, "MessageEmbedder", e)
        # botを終了させる
        await close_bot()

    # RAG Handlerが初期化失敗している場合は、ここで管理者通知
    if rag_handler is None and DISCORD_ADMIN_ID:
        await notify_admin_on_startup_error(client, "RAG Pipeline Handler", e)
        # botを終了させる
        await close_bot()
                
    # スラッシュコマンドを同期
    try:
        await tree.sync()
        logger.info("Slash commands synced successfully.")
    except Exception as e:
        logger.error(f"Failed to sync slash commands: {e}", exc_info=True)


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user: # ボット自身のメッセージは無視
        return

    if message.guild and not message.author.bot:
        # クロール済みのギルドからのメッセージのみを処理
        if message.guild.id in crawled_guild_ids:
            logger.info(f"New message in crawled guild '{message.guild.name}' (ID: {message.guild.id}) from {message.author.name}. Scheduling for processing.")
            asyncio.create_task(process_new_message_pipeline(message))
        else:
            logger.debug(f"Ignoring new message in non-crawled guild '{message.guild.name}' (ID: {message.guild.id}).")
    # 以前のメンションベースのコマンド処理、ヘルプ処理、crawl_history処理は削除 (スラッシュコマンドへ移行)


async def notify_admin_on_error(client: discord.Client, context_message: str, error: Exception):
    """エラー情報を管理者にDMで通知する"""
    if DISCORD_ADMIN_ID:
        try:
            admin_user = await client.fetch_user(int(DISCORD_ADMIN_ID))
            if admin_user:
                # スタックトレースの要約を取得
                tb_str = ''.join(traceback.format_exception(type(error), error, error.__traceback__, limit=5))
                error_details = f"Context: {context_message}\\nError Type: {type(error).__name__}\\nError: {error}\\nTraceback (partial):\\n```\\n{tb_str}\\n```"
                max_len = 1900 # Discordメッセージ長制限考慮
                if len(error_details) > max_len:
                    error_details = error_details[:max_len] + "... (truncated)"
                await admin_user.send(f"[Bot Error] {error_details}")
        except Exception as notify_e:
            logger.error(f"Failed to send error notification to admin: {notify_e}", exc_info=True)


async def process_new_message_pipeline(message: discord.Message):
    """
    単一の新規メッセージを処理し、Milvusに登録するパイプライン。
    エラーが発生してもユーザーには通知せず、管理者にのみ通知する。
    """
    try:
        logger.info(f"Processing new message ID {message.id} from guild '{message.guild.name if message.guild else 'N/A'}' channel '#{message.channel.name if hasattr(message.channel, 'name') else 'DM'}'")
        
        # guild_id を確実に取得・設定
        current_guild_id_str: str | None = None
        if message.guild:
            current_guild_id_str = str(message.guild.id)
        
        if not current_guild_id_str:
            logger.error(f"Guild ID is missing or None for message {message.id}. Skipping insertion.")
            return

        message_data = {
            "id": message.id,
            "content": message.content,
            "user_id": message.author.id,
            "user_name": message.author.name,
            "channel_id": message.channel.id,
            "channel_name": message.channel.name if hasattr(message.channel, 'name') else 'DM',
            "created_at": int(message.created_at.timestamp() * 1000),
            "guild_id": current_guild_id_str # guild_id を明示的に文字列で設定
        }
        
        # 1. データクリーニング (リスト形式で渡す)
        # ボット自身の発言は既に on_message の最初で弾かれているはずだが、念のため
        if str(message.author.id) == str(client.user.id):
             logger.debug(f"Skipping self message {message.id} in pipeline (should have been caught earlier).")
             return

        cleaned_messages = clean_messages([message_data]) # bot_id は渡さない (デフォルトNone)
        if not cleaned_messages:
            logger.info(f"Message {message.id} was removed after cleaning.")
            return
        
        # cleaned_messages[0] に guild_id が含まれているか確認し、なければ設定
        if isinstance(cleaned_messages[0], dict):
            if 'guild_id' not in cleaned_messages[0] or not cleaned_messages[0]['guild_id']:
                logger.warning(f"Guild ID missing in cleaned_messages for message {message.id}. Restoring from original: {current_guild_id_str}")
                cleaned_messages[0]['guild_id'] = current_guild_id_str
        else:
            logger.error(f"cleaned_messages[0] is not a dict for message {message.id}. Cannot ensure guild_id.")
            return # or handle appropriately
        
        # 2. チャンキング
        chunked_data = chunk_messages(
            cleaned_messages, 
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        if not chunked_data:
            logger.info(f"Message {message.id} resulted in no chunks after chunking.")
            return

        # 3. エンベディング
        global message_embedder # グローバル変数を参照
        if not message_embedder:
            logger.error("MessageEmbedder is not initialized. Cannot process new message.")
            # 管理者に通知することも検討
            await notify_admin_on_error(client, f"MessageEmbedder not initialized during process_new_message_pipeline for message {message.id}", RuntimeError("MessageEmbedder not initialized"))
            return
        
        embedded_data = await asyncio.to_thread(message_embedder.embed_chunks, chunked_data) # 変更後: グローバルインスタンスを使用
        if not embedded_data or not any(d.get("embedding") for d in embedded_data):
            logger.warning(f"Failed to generate embeddings for message {message.id} or no valid embeddings produced.")
            return

        # embedded_data の各要素の metadata に guild_id を文字列として確実に付与
        # run_crawl_and_process_pipeline と同様のロジック
        if current_guild_id_str: # guild_id が取得できている場合のみ処理
            for item in embedded_data:
                if isinstance(item, dict):
                    if "metadata" not in item or not isinstance(item.get("metadata"), dict):
                        item["metadata"] = {} # metadata がないか、辞書でない場合は初期化
                    item["metadata"]["guild_id"] = current_guild_id_str
                    # logger.debug(f"Ensured guild_id '{current_guild_id_str}' in metadata for chunk {item.get('id')}") # 詳細ログは必要に応じて
                else:
                    logger.warning(f"Item in embedded_data for message {message.id} is not a dict. Cannot ensure guild_id in metadata.")
        else:
            # current_guild_id_str がない場合、ここまでの処理でエラーになっているはずだが念のためログ
            logger.error(f"Critical: current_guild_id_str is None before inserting to Milvus for message {message.id}. This should not happen.")
            # このケースでは挿入をスキップするか、エラー処理を強化する必要があるかもしれない
            return

        # 4. Milvusへの挿入
        # MilvusHandlerも同様に、接続を都度開閉するか、プールするか検討の余地あり
        milvus_handler = MilvusHandler(
            collection_name=MILVUS_COLLECTION_NAME
        )
        await asyncio.to_thread(milvus_handler.insert_data, embedded_data)
        # Milvusの件数フィードバック有効化
        count = await asyncio.to_thread(milvus_handler.count_entities)
        logger.info(f"New message {message.id} processed and {len(embedded_data)} chunks inserted. Total entities in '{MILVUS_COLLECTION_NAME}': {count}")
        
        await asyncio.to_thread(milvus_handler.close_connection)

    except Exception as e:
        logger.error(f"Error processing new message {message.id} in pipeline: {e}", exc_info=True)
        await notify_admin_on_error(client, f"New message processing pipeline failed for message ID {message.id} from user {message.author.name}", e)


async def run_crawl_and_process_pipeline(channel: Optional[discord.TextChannel | discord.DMChannel], guild: discord.Guild): # channelをOptionalに変更
    """
    指定されたDiscordギルドのメッセージをクロールし、処理してMilvusに登録するパイプライン。
    処理完了後、ギルドIDをクロール済みとして記録する。
    """
    progress_channel = channel # 進捗通知用チャンネルを保持

    try:
        if progress_channel: await progress_channel.send(f"サーバー「{guild.name}」のデータ再登録処理を開始します。")
        else: logger.info(f"Data re-registration process started for guild '{guild.name}' (ID: {guild.id}) without a specific progress channel.")
        
        # 1. 指定されたギルドの既存データをMilvusから削除
        try:
            guild_id_str = str(guild.id)
            await progress_channel.send(f"サーバー「{guild.name}」の既存データを確認しています...")
            logger.info(f"Attempting to delete data for guild ID {guild_id_str} from Milvus collection: {MILVUS_COLLECTION_NAME}")
            
            milvus_handler_for_delete = MilvusHandler(collection_name=MILVUS_COLLECTION_NAME)
            
            affected_rows = await asyncio.to_thread(milvus_handler_for_delete.delete_data_by_guild_id, guild_id_str)

            await asyncio.to_thread(milvus_handler_for_delete.close_connection)

            if affected_rows > 0:
                await progress_channel.send(f"サーバー「{guild.name}」の既存データを削除しました。")
                logger.info(f"Successfully deleted data for guild ID {guild_id_str} from '{MILVUS_COLLECTION_NAME}'. Affected rows: {affected_rows}")
            elif affected_rows == 0:
                await progress_channel.send(f"サーバー「{guild.name}」の既存データは見つかりませんでした。")
                logger.info(f"No data found for guild ID {guild_id_str} in '{MILVUS_COLLECTION_NAME}' to delete.")
            else: # マイナス値などが返ってきた場合 (エラーケース)
                await progress_channel.send(f"サーバー「{guild.name}」(ID: {guild_id_str}) のデータ削除処理中に問題が発生した可能性があります。")
                logger.warning(f"Deletion of data for guild ID {guild_id_str} from '{MILVUS_COLLECTION_NAME}' reported {affected_rows} affected rows.")

        except Exception as e:
            logger.error(f"Error during Milvus data deletion phase for guild {guild.id}: {e}", exc_info=True)
            await progress_channel.send("既存データ削除中にエラーが発生しました: {e}")
            # ここで処理を中断するかどうかは要件によるが、今回は続行してクロールと挿入を試みる
            # return # 中断する場合はコメントアウトを解除

        # 2. Discordメッセージ履歴をクロール
        await progress_channel.send(f"サーバー「{guild.name}」のメッセージ履歴を収集しています...")
        
        logger.info(f"Starting crawl for guild: {guild.name} ({guild.id}) with message limit per channel: {MAX_MESSAGES_PER_CHANNEL if MAX_MESSAGES_PER_CHANNEL is not None else 'None'}")
        crawler = DiscordCrawler(client=client) # guild_id と max_messages_per_channel は渡さない
        collected_messages = await crawler.crawl_specific_guild(guild, max_messages_per_channel=MAX_MESSAGES_PER_CHANNEL) 

        if not collected_messages:
            await progress_channel.send(f"メッセージが収集されませんでした。サーバー「{guild.name}」にメッセージが存在しないか、ボットにアクセス権限がない可能性があります。")
            return
        await progress_channel.send(f"収集完了。")

        # 3. データクリーニング
        await progress_channel.send(f"**ステップ2/5: データクリーニング開始...**")
        cleaned_messages = await asyncio.to_thread(clean_messages, collected_messages)
        await progress_channel.send(f"クリーニング完了。")
        if not cleaned_messages:
            await progress_channel.send("クリーニング後、処理対象のメッセージが残っていません。")
            return

        # 4. チャンキング
        await progress_channel.send(f"**ステップ3/5: チャンキング開始...**")
        chunked_data = await asyncio.to_thread(
            chunk_messages, 
            cleaned_messages, 
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        await progress_channel.send(f"チャンキング完了。")
        if not chunked_data:
            await progress_channel.send("チャンキング後、処理対象のデータが残っていません。")
            return
        
        # 各チャンクのメタデータに guild_id を追加
        # guild.id は数値型なので、MilvusのVARCHARスキーマに合わせて文字列に変換する
        current_guild_id_str = str(guild.id)
        for chunk_item in chunked_data:
            if "metadata" not in chunk_item or not isinstance(chunk_item["metadata"], dict):
                chunk_item["metadata"] = {}
            chunk_item["metadata"]["guild_id"] = current_guild_id_str
        logger.info(f"Added guild_id '{current_guild_id_str}' to metadata of {len(chunked_data)} chunks.")
        
        # 5. エンベディング
        await progress_channel.send(f"**ステップ4/5: エンベディング生成開始...**")
        if not message_embedder:
            logger.error("MessageEmbedder is not initialized. Cannot run crawl and process pipeline.")
            await progress_channel.send("エラー: エンベディングモデルが初期化されていません。管理者に連絡してください。")
            await notify_admin_on_error(client, f"MessageEmbedder not initialized during run_crawl_and_process_pipeline for guild {guild.id}", RuntimeError("MessageEmbedder not initialized"))
            return
        
        embedded_data = await asyncio.to_thread(message_embedder.embed_chunks, chunked_data) # 変更後: グローバルインスタンスを使用
        await progress_channel.send(f"エンベディング生成完了。")
        if not embedded_data or not any(d.get("embedding") for d in embedded_data):
             await progress_channel.send("エンベディングの生成に失敗したか、有効なエンベディングを持つデータがありません。")
             return

        # 6. Milvusへの挿入
        await progress_channel.send(f"**ステップ5/5: データ登録開始...**")
        
        if embedded_data:
            logger.info(f"Sample data for Milvus insertion (first 2 items): {embedded_data[:2]}")
        else:
            logger.warning("No embedded data to insert.")
            await progress_channel.send("エンベディングされたデータがありません。挿入をスキップします。")
            return

        milvus_handler = await asyncio.to_thread(
            MilvusHandler,
            collection_name=MILVUS_COLLECTION_NAME # ここでコレクションがなければ新規作成されるはず
        )
        
        inserted_pks = await asyncio.to_thread(milvus_handler.insert_data, embedded_data)
        
        if inserted_pks is not None:
            logger.info(f"Milvus insert_data reported {len(inserted_pks)} primary keys inserted for guild {guild.id}. Sample PKs: {inserted_pks[:10]}")
            if not inserted_pks: 
                 logger.warning(f"Milvus insert_data returned an empty list of primary keys for guild {guild.id}.")
        else:
            logger.error(f"Milvus insert_data returned None for guild {guild.id}, indicating a failure during insertion.")
            await progress_channel.send("データ登録中にエラーが発生した可能性があります。詳細はログを確認してください。")
            await asyncio.to_thread(milvus_handler.close_connection)
            return

        count_immediately_after_insert = await asyncio.to_thread(milvus_handler.count_entities)
        logger.info(f"Count from the same MilvusHandler for guild {guild.id} immediately after insert: {count_immediately_after_insert}")
        
        await progress_channel.send(f"データ登録完了。")
        
        await asyncio.to_thread(milvus_handler.close_connection)

        # 処理完了後、ギルドIDをクロール済みとして記録
        save_crawled_guild(guild.id)
        await progress_channel.send(f"サーバー「{guild.name}」の履歴収集と処理がすべて完了しました。このサーバーで検索機能が利用可能になりました。")

    except Exception as e:
        logger.error(f"Error during crawl and process pipeline for guild {guild.id} ('{guild.name}'): {e}", exc_info=True)
        await progress_channel.send(f"履歴収集・処理中にエラーが発生しました。管理者、もしくは開発者にお問い合わせください。")
        await notify_admin_on_error(client, f"Crawl & Process Pipeline Failed. Guild: {guild.name}({guild.id})", e)


# --- スラッシュコマンドの定義 --- #

@tree.command(name="help", description="Botが提供するスラッシュコマンドの一覧と、それぞれの使用方法に関するヘルプ情報を表示します。")
async def help_command(interaction: discord.Interaction):
    logger.info(f"Executing /help command for {interaction.user.name}")
    help_embed = discord.Embed(
        title="Bot ヘルプ",
        description="Discordサーバー内の情報を検索するボットです。",
        color=discord.Color.blue()
    )
    help_embed.add_field(
        name="主な機能",
        value="- `/search <質問文> [channel] [user]`: 指定された条件で情報を検索します。\n"
              "- `/crawl_history [guild_id]`: [管理者専用] サーバーのメッセージ履歴を収集・処理し、検索可能にします。\n" # crawl_historyの説明を少し変更
              "- `/help`: このヘルプメッセージを表示します。",
        inline=False
    )
    help_embed.add_field(
        name="`/search` コマンドの詳細",
        value="`質問文`: 検索したい内容を自然言語で入力します。(必須)\n"
              "`channel`: 特定のチャンネル名またはIDで検索範囲を限定します。(任意)\n"
              "`user`: 特定のユーザー名またはIDで検索範囲を限定します。(任意)\n"
              "`date_from`: 検索開始日をYYYY-MM-DD形式で指定します。(任意)\n"
              "`date_to`: 検索終了日をYYYY-MM-DD形式で指定します。(任意)\n"
              "**注:** `/search` コマンドを利用するには、事前に管理者が対象サーバーで `/crawl_history` を実行している必要があります。\n"
              "date_from,toは片方を入力する事でその日付以降,以前を検索できます。", # 注意書き追加
        inline=False
    )
    help_embed.add_field(
        name="利用例",
        value="- `/search ○○について`\n"
              "- `/search 最新のお知らせ channel: #アナウンス`\n"
              "- `/search 過去のイベント情報 user: @ユーザー名`",
        inline=False
    )
    help_embed.set_footer(text="Bot by Gemini")
    await interaction.response.send_message(embed=help_embed, ephemeral=True) # ephemeral=True で本人にのみ表示


@tree.command(name="search", description="指定された質問文に基づいて情報を検索します。")
@app_commands.describe(
    query="検索したい質問文",
    channel="検索対象のチャンネル (任意)",
    user="検索対象のユーザー (任意)",
    date_from="検索日(○○以降 YYYY-MM-DD形式 任意)",
    date_to="~検索日(○○以前 YYYY-MM-DD形式 任意)"
)
async def search_command(interaction: discord.Interaction, query: str, channel: Optional[discord.TextChannel] = None, user: Optional[discord.User] = None, date_from: Optional[str] = None, date_to: Optional[str] = None):
    logger.info(f"Executing /search command for {interaction.user.name}. Guild: {interaction.guild.name if interaction.guild else 'DM'}. Query: '{query}', Channel: {channel.name if channel else 'None'}, User: {user.name if user else 'None'}, Date From: {date_from}, Date To: {date_to}")

    if not interaction.guild:
        await interaction.response.send_message("このコマンドはDiscordサーバー内でのみ使用できます。", ephemeral=True)
        return

    if interaction.guild.id not in crawled_guild_ids:
        logger.warning(f"/search command denied for guild '{interaction.guild.name}' (ID: {interaction.guild.id}) as it has not been crawled.")
        await interaction.response.send_message(
            f"サーバー「{interaction.guild.name}」のデータはまだ検索できません。\n"
            "管理者が `/crawl_history` コマンドを実行して、このサーバーのメッセージ履歴を処理する必要があります。",
            ephemeral=True
        )
        return

    if not rag_handler:
        logger.warning("RAG Pipeline Handler is not initialized. Cannot process /search command.")
        await interaction.response.send_message("ボットのコア機能が初期化されていません。管理者にお問い合わせください。", ephemeral=True)
        return

    await interaction.response.defer(thinking=True) 

    try:
        # フィルタ用のIDを取得
        filter_channel_id_str = str(channel.id) if channel else None
        filter_user_id_str = str(user.id) if user else None
        # 日付のフォーマットチェックと変換
        try:
            filter_date_from_timestamp = int(datetime.strptime(date_from, "%Y-%m-%d").timestamp() * 1000) if date_from else None  # ミリ秒に変換
            filter_date_to_timestamp = int(datetime.strptime(date_to, "%Y-%m-%d").timestamp() * 1000) if date_to else None  # ミリ秒に変換
        except ValueError:
            logger.error(f"Invalid date format: {date_from} - {date_to}")
            await interaction.response.send_message("無効な日付形式です。YYYY-MM-DD形式で指定してください。", ephemeral=True)
            return

        # 日付の順序チェックを追加
        if filter_date_from_timestamp is not None and filter_date_to_timestamp is not None:
            if filter_date_to_timestamp < filter_date_from_timestamp:
                await interaction.response.send_message("「検索終了日」が「検索開始日」より前の日付になっています。", ephemeral=True)
                return

        result = await asyncio.to_thread(
            rag_handler.query,
            query_text=query, 
            filter_channel=filter_channel_id_str,
            filter_user=filter_user_id_str,
            guild_id=str(interaction.guild.id),
            filter_date_from=filter_date_from_timestamp,
            filter_date_to=filter_date_to_timestamp
        )
        answer = result.get("answer", "すみません、うまく応答を生成できませんでした。")

        logger.info(f"RAG pipeline returned answer for /search: '{answer[:100]}...'")

        response_embed = discord.Embed(
            title="検索結果",
            description=answer,
            color=discord.Color.green()
        )

        response_embed.set_footer(text=f"Powered by Bot System | 質問: {query[:50]}...")
        await interaction.followup.send(embed=response_embed)

    except Exception as e:
        logger.error(f"Error processing /search command with RAG pipeline: {e}", exc_info=True)
        error_message_user = f"申し訳ありません、質問の処理中にエラーが発生しました。管理者にご連絡ください。\\nエラー: {type(e).__name__}"
        if interaction.response.is_done(): # defer()後にエラーが発生した場合
            await interaction.followup.send(error_message_user, ephemeral=True)
        else: # defer()前にエラーが発生した場合 (RAGハンドラ未初期化など)
            await interaction.response.send_message(error_message_user, ephemeral=True)
        await notify_admin_on_error(
            client,
            f"/search command processing failed. User: {interaction.user.name}, Query: '{query}', Channel: {channel.name if channel else 'None'}, User: {user.name if user else 'None'}",
            e
        )


@tree.command(name="crawl_history", description="[管理者専用] Discordサーバーの過去ログを収集・処理します。")
@app_commands.describe(
    guild_id="収集対象のサーバーID (ボットが参加している必要があります。未指定の場合はコマンド実行サーバー)" # 説明を修正
)
async def crawl_history_command(interaction: discord.Interaction, guild_id: Optional[str] = None):
    logger.info(f"Executing /crawl_history command for {interaction.user.name}. Specified Guild ID: {guild_id}, Invoked in Guild: {interaction.guild.name if interaction.guild else 'DM'}")    
    
    if str(interaction.user.id) != DISCORD_ADMIN_ID:
        logger.warning(f"User {interaction.user.name} (ID: {interaction.user.id}) tried to use /crawl_history but is not the configured admin (ID: {DISCORD_ADMIN_ID}).")
        await interaction.response.send_message("このコマンドは設定されたボット管理者専用です。", ephemeral=True)
        return

    target_guild: discord.Guild | None = None
    if guild_id:
        try:
            parsed_guild_id = int(guild_id)
            target_guild = client.get_guild(parsed_guild_id)
            if not target_guild:
                await interaction.response.send_message(f"指定されたサーバーID `{guild_id}` が見つかりません。ボットがそのサーバーに参加しているか確認してください。", ephemeral=True)
                return
        except ValueError:
            await interaction.response.send_message(f"無効なサーバーID形式です: `{guild_id}`。数値のIDを指定してください。", ephemeral=True)
            return
    elif interaction.guild: # guild_id が指定されず、コマンドがサーバー内で実行された場合
        target_guild = interaction.guild
    else: # guild_id が指定されず、DMから実行された場合
        await interaction.response.send_message("DMからこのコマンドを実行する場合、収集対象のサーバーIDを `guild_id` オプションで指定してください。", ephemeral=True)
        return

    if target_guild:
        # ephemeral=Falseにすると、コマンド実行者以外にも見える初期メッセージ
        await interaction.response.send_message(f"サーバー「{target_guild.name}」(ID: {target_guild.id}) の履歴収集と処理を開始します。完了まで時間がかかる場合があります。進捗はこのチャンネルでお知らせします。") 
        
        progress_channel: Optional[discord.TextChannel | discord.DMChannel] = None
        if isinstance(interaction.channel, (discord.TextChannel, discord.DMChannel)):
            progress_channel = interaction.channel
        else: # 予期しないチャンネルタイプの場合、管理者にDMを送る試み
            logger.warning(f"/crawl_history was invoked from an unexpected channel type: {type(interaction.channel)}. Attempting to send progress to admin DM.")
            try:
                admin_user = await client.fetch_user(int(DISCORD_ADMIN_ID))
                if admin_user:
                    progress_channel = await admin_user.create_dm() # 管理者とのDMチャンネルを取得/作成
                    await progress_channel.send(f"サーバー「{target_guild.name}」(ID: {target_guild.id}) の履歴収集と処理を開始しました。(進捗はこちらのDMに通知)")
            except Exception as dm_err:
                logger.error(f"Failed to create DM channel with admin for progress updates: {dm_err}")
                await interaction.followup.send("進捗通知用のDMチャンネルを作成できませんでした。", ephemeral=True)

        asyncio.create_task(run_crawl_and_process_pipeline(progress_channel, target_guild))


@crawl_history_command.error
async def crawl_history_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message("このコマンドを実行する権限がありません。(サーバー管理者権限が必要です)", ephemeral=True)
    else:
        logger.error(f"Error in /crawl_history command: {error}", exc_info=True)
        await interaction.response.send_message("コマンドの実行中にエラーが発生しました。", ephemeral=True)
        await notify_admin_on_error(client, f"/crawl_history command failed. User: {interaction.user.name}", error)


def main():
    if not DISCORD_BOT_TOKEN:
        logger.critical("DISCORD_BOT_TOKEN is not set. Bot cannot start.")
        return

    try:
        logger.info("Starting Bot...")
        client.run(DISCORD_BOT_TOKEN)
    except discord.LoginFailure:
        logger.critical("Failed to log in to Discord. Check your bot token.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred while starting the bot: {e}", exc_info=True)


if __name__ == "__main__":
    main()