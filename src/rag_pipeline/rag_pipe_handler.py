import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from statistics import mean

from src.config import (
    GEMINI_API_KEY, 
    LLM_MODEL_NAME, 
    EMBEDDING_MODEL_NAME, 
    SIMILARITY_TOP_K, LLM_TEMPERATURE,
    MILVUS_COLLECTION_NAME,
    EMBEDDING_MODEL_DIMENSION
)
from src.vector_store.milvus_handler import MilvusHandler # MilvusHandler をインポート
from src.embedding.embedder import MessageEmbedder # MessageEmbedder をインポート

# Google Gen AI SDK を使用したAPI呼び出し
from google import genai # 新SDK
from google.genai import types # 新SDK
from google.genai import errors # 新SDK。エラーは errors.BlockedPromptError のように参照

logger = logging.getLogger(__name__)

# システムプロンプト
SYSTEM_PROMPT_FOR_RAG_UPDATED = (
    "あなたは情報検索のエキスパートです。"
    "ユーザーの自然言語による質問から、関連情報を最大限に引き出す最適な検索クエリを自動生成してください。"
    "質問が曖昧・広範な場合は、「サーバーの目的」「ルール」「チャンネル構成」「総合案内」「自己紹介」など、具体かつ多様なキーワード群を作成し、Function Calling を用いて検索に渡してください。"
    "検索結果をもとに最終回答を生成しますが、検索クエリの質を最優先してください。"
)

class RAGPipelineHandler:
    """
    RAG (Retrieval Augmented Generation) パイプラインを処理するハンドラクラス。
    Google Gen AI SDKのFunction Calling機能を利用して、Milvusベクトルストアから情報を検索し、
    LLM (Gemini) を用いて応答を生成する。
    """
    def __init__(self,
                 milvus_collection_name: str = MILVUS_COLLECTION_NAME,
                 embedding_model_name: str = EMBEDDING_MODEL_NAME,
                 llm_model_name: str = LLM_MODEL_NAME, # Gemini APIで使用するモデル名
                 milvus_dim: int = EMBEDDING_MODEL_DIMENSION, # DEFAULT_DIMENSION を EMBEDDING_MODEL_DIMENSION に変更
                 similarity_top_k: int = SIMILARITY_TOP_K, # main.pyのconfigから渡される値で上書き
                 verbose_logging: bool = False):
        """
        RAGPipelineHandlerを初期化する。

        Args:
            milvus_collection_name (str): Milvusコレクション名。
            embedding_model_name (str): 使用する埋め込みモデル名。
            llm_model_name (str): 使用するLLMモデル名 (Gemini API用)。
            milvus_dim (int): Milvusベクトルストアの次元数。
            similarity_top_k (int): ベクトル検索時に取得する上位K件。
            verbose_logging (bool): 詳細なログ出力を行うか否か。
        """
        logger.info(f"RAGPipelineHandlerを初期化中... similarity_top_k: {similarity_top_k}")
        self.milvus_collection_name = milvus_collection_name
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name 
        self.milvus_dim = milvus_dim
        self.similarity_top_k = similarity_top_k
        self.verbose_logging = verbose_logging # 現在は直接利用されていないが、将来的な拡張のため保持
        # LLM生成時の温度設定
        self.temperature = LLM_TEMPERATURE

        if not GEMINI_API_KEY:
            raise ValueError("環境変数 GEMINI_API_KEY が設定されていません。")
        
        # Google Gen AI SDK クライアントを初期化
        self.client = genai.Client(api_key=GEMINI_API_KEY) # genai.Client を使用
        logger.info(f"Google Gen AI SDKクライアントを初期化しました。")

        try:
            logger.info(f"Initializing MessageEmbedder with model: {self.embedding_model_name}")
            embedder_instance = MessageEmbedder(model_name=self.embedding_model_name)
            self.embed_model = embedder_instance.model # MessageEmbedderが持つHuggingFaceEmbeddingインスタンスを使用
            if not self.embed_model:
                # このチェックはMessageEmbedderのコンストラクタでエラーが発生すれば到達しないはずだが念のため
                raise ValueError("Failed to get a valid embedding model from MessageEmbedder.") 
            logger.info(f"Embedding model (via MessageEmbedder) '{self.embedding_model_name}' initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model via MessageEmbedder: {e}", exc_info=True)
            raise

        # ベクトルストア (MilvusVectorStore) の初期化
        try:
            logger.info(f"MilvusVectorStoreを初期化中 (コレクション: {self.milvus_collection_name})")
            self.milvus_handler = MilvusHandler(collection_name=self.milvus_collection_name)
            logger.info("MilvusVectorStoreの初期化完了。")
        except Exception as e:
            logger.error(f"MilvusVectorStoreの初期化に失敗しました: {e}", exc_info=True)
            raise

        # Function Calling用の関数定義
        self.search_milvus_declaration = {
            "name": "search_milvus",
            "description": "ベクトルDBから、ユーザーの質問や指示に合致する関連情報を検索",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "検索クエリ文字列。ユーザーの質問を元に、検索に適したキーワードを生成してください。"
                    },
                    "start_timestamp_ms": {
                        "type": "integer",
                        "description": "検索対象の開始日時 (UTC) をUnixタイムスタンプ (ミリ秒) で指定",
                        "nullable": True
                    },
                    "end_timestamp_ms": {
                        "type": "integer",
                        "description": "検索対象の終了日時 (UTC) をUnixタイムスタンプ (ミリ秒) で指定",
                        "nullable": True
                    },
                },
                "required": ["search_query"] # search_query は必須
            }
        }
        
        logger.info("RAGPipelineHandler (Function Callingモード) のセットアップ完了。")


    def search_milvus(self, 
                      search_query: str, 
                      channel_filter: Optional[str] = None, 
                      user_filter: Optional[str] = None, 
                      guild_id: Optional[str] = None,
                      start_timestamp_ms: Optional[int] = None, # New
                      end_timestamp_ms: Optional[int] = None) -> List[Dict[str, Any]]: # New
        """
        DiscordメッセージをMilvusから検索する内部関数 (GeminiのFunction Callingによって呼び出される)。

        Args:
            search_query (str): LLMによって生成された検索クエリ。
            channel_filter (Optional[str]): 検索対象のチャンネル名またはID。
            user_filter (Optional[str]): 検索対象のユーザー名またはID。
            guild_id (Optional[int]): 検索対象のギルドID。
            start_timestamp_ms (Optional[int]): 検索対象の開始日時 (UTC) をUnixタイムスタンプ (ミリ秒) で指定。
            end_timestamp_ms (Optional[int]): 検索対象の終了日時 (UTC) をUnixタイムスタンプ (ミリ秒) で指定。

        Returns:
            Dict[str, Any]: 検索結果、またはエラー情報を含む辞書。
                             成功時は {"retrieved_contexts": [context_dict, ...]}
                             失敗時は {"error": "エラーメッセージ"}
        """
        
        logger.info(f"関数呼び出し: search_milvus (検索クエリ: '{search_query}', チャンネル: {channel_filter}, ユーザー: {user_filter}, Guild ID: {guild_id}, StartTS: {start_timestamp_ms}, EndTS: {end_timestamp_ms})")
        
        if not self.embed_model:
            logger.error("埋め込みモデルが初期化されていません (search_milvus内)。")
            return [{"error": "埋め込みモデルが初期化されていません。"}]
        if not self.milvus_handler:
            logger.error("Milvusベクトルストアが初期化されていません (search_milvus内)。")
            return [{"error": "Milvusベクトルストアが初期化されていません。"}]

        try:
            # メタデータフィルターの構築
            filters_list = []
            if channel_filter:
                filters_list.append(f"channel_id == '{channel_filter}'")
                logger.info(f"Applying channel_id filter: {channel_filter}")
            if user_filter:
                filters_list.append(f"user_id == '{user_filter}'")
                logger.info(f"Applying user_id filter: {user_filter}")
            if guild_id:
                filters_list.append(f"guild_id == '{guild_id}'")
                logger.info(f"Applying guild_id filter: {guild_id}")
            
            # Add timestamp filters
            if start_timestamp_ms is not None:
                filters_list.append(f"timestamp >= {start_timestamp_ms}")
                logger.info(f"Applying start_timestamp_ms filter: timestamp >= {start_timestamp_ms}")
            if end_timestamp_ms is not None:
                filters_list.append(f"timestamp <= {end_timestamp_ms}")
                logger.info(f"Applying end_timestamp_ms filter: timestamp <= {end_timestamp_ms}")
            
            filter_expr = " and ".join(filters_list) if filters_list else None

            # 埋め込みベクトルの生成
            embs = []
            # search_query が文字列の場合とリストの場合があり得るため、リストに統一
            query_keywords = [search_query] if isinstance(search_query, str) else search_query
            for kw in query_keywords:
                embs.append(self.embed_model.get_query_embedding(kw))
            # キーワード毎の埋め込みを平均
            embedding = [mean(vals) for vals in zip(*embs)] if embs else []
            
            if not embedding:
                logger.warning("Embedding for search_query resulted in an empty list. Skipping Milvus query.")
                return {"retrieved_contexts": [], "message": "検索クエリから有効な埋め込みを生成できませんでした。"}

            # ベクトル検索の実行
            results = self.milvus_handler.search(
                query=[embedding],
                top_k=self.similarity_top_k,
                filters=filter_expr,
                output_fields=["text"]
            )

            logger.info(f"Milvusから {len(results)} 件のノードを取得しました。")

            if results[0] != []:
                documents = []
                for result in results[0]:
                    documents.append({
                        "content": result["entity"]["text"],
                        "score":   result["distance"]
                    })
                return documents
            else:
                # 検索結果が0件の場合
                logger.info("Milvusでの検索結果、該当するドキュメントは見つかりませんでした。")
                return [{"error": "関連する情報は見つかりませんでした。"}]

        except Exception as e:
            logger.error(f"Milvus検索処理中 (search_milvus) にエラーが発生しました: {e}", exc_info=True)
            return [{"error": f"検索処理中にエラーが発生しました: {str(e)}"}]


    def query(self, 
              query_text: str, 
              filter_channel: Optional[str] = None, 
              filter_user: Optional[str] = None, 
              guild_id: Optional[str] = None,
              filter_date_from: Optional[int] = None,
              filter_date_to: Optional[int] = None) -> Dict[str, Any]: # New
        """
        ユーザーからの質問に対してRAGパイプラインを実行し、応答と参照元情報を返す。

        Args:
            query_text (str): ユーザーからの質問文。
            filter_channel (Optional[str]): 検索対象のチャンネル名またはID。
            filter_user (Optional[str]): 検索対象のユーザー名またはID。
            guild_id (Optional[str]): 検索対象のギルドID。

        Returns:
            Dict[str, Any]: LLMによって生成された回答と、参照元のソース情報を含む辞書。
                            例: {"answer": "回答文", "sources": [source_dict_1, ...]}
        """
        logger.info(f"RAG query処理開始: '{query_text}', チャンネル: {filter_channel}, ユーザー: {filter_user}, Guild ID: {guild_id}, 日付フィルター: {filter_date_from} から {filter_date_to}")

        if not self.llm_model_name: # client のチェックではなく、モデル名が設定されているかで判定
            logger.error("LLMモデル名が設定されていません。処理を中断します。")
            return {"answer": "エラー: LLMモデルが設定されていません。"}

        try:
            logger.info(f"Gemini APIを呼び出して応答を取得中...")
            # Gemini APIを呼び出して応答を取得
            response = self.client.models.generate_content(
                model=self.llm_model_name,
                contents=[types.Content(role="user", parts=[types.Part(text=f"ユーザーの質問: {query_text}")])],
                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT_FOR_RAG_UPDATED, # システムプロンプトを設定
                    temperature=self.temperature, # 温度パラメータ
                    tools=[types.Tool(function_declarations=[self.search_milvus_declaration])], # 使用する関数の宣言    
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY", # 関数呼び出しのモード
                            allowed_function_names=["search_milvus"] # 許可された関数名
                        )
                    )
                )
            )
            
            # Function Calling がない場合は直接回答
            if not response.candidates[0].content.parts[0].function_call:
                 logger.warning("Geminiからの応答に 'function_call' が含まれていません。直接回答またはエラーの可能性があります。")
                 # .text は存在すればそれを、なければ固定メッセージ
                 return {"answer": response.text if hasattr(response, 'text') and response.text else "LLMからの応答形式が予期したものではありませんでした。"} 

            tool_call = response.candidates[0].content.parts[0].function_call

            # Gemini が関数呼び出しを要求してきた場合
            if tool_call.name == "search_milvus":

                # 日付フィルターが直接指定されている場合は、そっちを優先
                if filter_date_from:
                    extracted_start_ts = filter_date_from
                else:
                    extracted_start_ts = tool_call.args["start_timestamp_ms"] if "start_timestamp_ms" in tool_call.args else None

                if filter_date_to:
                    extracted_end_ts = filter_date_to
                else:
                    extracted_end_ts = tool_call.args["end_timestamp_ms"] if "end_timestamp_ms" in tool_call.args else None

                logger.info(f"Geminiが関数 'search_milvus' の呼び出しを要求しました。 日付フィルター: {extracted_start_ts} から {extracted_end_ts}")
                search_result = self.search_milvus(
                    tool_call.args["search_query"],
                    channel_filter=filter_channel,
                    user_filter=filter_user,
                    guild_id=guild_id,
                    start_timestamp_ms=extracted_start_ts, # 日付フィルター
                    end_timestamp_ms=extracted_end_ts    # 日付フィルター
                )
                    
                logger.info(f"search_milvusからの結果をGeminiに返送します。結果概要: {search_result}")

                # エラーチェック追加
                if isinstance(search_result, list) and 'error' in search_result[0]:
                    logger.warning(f"search_milvusからエラーが返されました: {search_result[0]['error']}")
                    return {"answer": f"申し訳ありません、情報の検索中に問題が発生しました。"}

                # 成功時の処理
                context = "\n\n".join(
                    f"score={d['score']:.3f}): {d['content']}"
                    for i, d in enumerate(search_result)
                )

                prompt = (
                    f"以下の文書を参照して回答を生成してください。\n"
                    f"{query_text}\n\n"
                    f"文書一覧:\n{context}"
                )
                resp = self.client.models.generate_content(
                    model=self.llm_model_name,
                    contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                    config=types.GenerateContentConfig(
                        temperature=self.temperature
                    )
                )

                final_answer = resp.text
                date_filter_message = ""
                if extracted_start_ts and extracted_end_ts:
                    s_date = datetime.fromtimestamp(extracted_start_ts/1000, timezone.utc).strftime('%Y-%m-%d')
                    e_date = datetime.fromtimestamp(extracted_end_ts/1000, timezone.utc).strftime('%Y-%m-%d')
                    date_filter_message = f"\n(検索結果は {s_date} から {e_date} の期間でフィルタリングされています)"
                elif extracted_start_ts:
                    s_date = datetime.fromtimestamp(extracted_start_ts/1000, timezone.utc).strftime('%Y-%m-%d')
                    date_filter_message = f"\n(検索結果は {s_date} 以降でフィルタリングされています)"
                elif extracted_end_ts:
                    e_date = datetime.fromtimestamp(extracted_end_ts/1000, timezone.utc).strftime('%Y-%m-%d')
                    date_filter_message = f"\n(検索結果は {e_date} 以前でフィルタリングされています)"
            
                if date_filter_message:
                    final_answer += date_filter_message
                    logger.info(f"Appended date filter information to answer: {date_filter_message}")

                logger.info(f"Geminiによる最終回答 (日付フィルタ情報付加後): {final_answer[:100]}...")
                return {"answer": final_answer}

            else:
                # LLMがFunction Callを要求せず、直接回答を生成した場合は許可せず、安全なメッセージを返す
                logger.info("GeminiはFunction Callを行わず、直接回答しました。関連情報がないものとみなします。")
                return {"answer": "申し訳ありません、関連する情報は見つかりませんでした。"}

        except errors.APIError as api_err:
            logger.warning(f"Google GenAI API でエラーが発生しました (code={getattr(api_err,'code', 'N/A')}): {api_err.message}")
            # エラーコード 403 や 429 等に応じてメッセージを変えても良いが、ここでは汎用的に返す
            return {"answer": "申し訳ありません、現在リクエストを処理できませんでした。少し時間をおいて再度お試しください。"}
        except Exception as e:
            logger.error(f"RAGクエリ実行中 (Gemini Function Calling) に予期せぬエラーが発生しました: {e}", exc_info=True)
            return {"answer": f"申し訳ありません、質問の処理中にエラーが発生しました: {type(e).__name__}"}
