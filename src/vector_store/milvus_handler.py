import logging
from typing import List, Dict, Any
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from src.config import MILVUS_HOST, MILVUS_PORT, EMBEDDING_MODEL_DIMENSION, MILVUS_ALIAS

logger = logging.getLogger(__name__)


class MilvusHandler:
    def __init__(self, collection_name: str, embedding_dim: int = EMBEDDING_MODEL_DIMENSION, alias: str = MILVUS_ALIAS):
        """
        Milvusへの接続とコレクションの初期化を行う。
        collection_name: 使用するコレクション名
        embedding_dim: ベクトルの次元数
        """
        self.alias = alias
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection = None
        self._connect()
        self._create_collection_if_not_exists()


    def _connect(self):
        """Milvusサーバーに接続する"""
        try:
            logger.info(f"Connecting to Milvus server at {MILVUS_HOST}:{MILVUS_PORT} with alias '{self.alias}'")
            connections.connect(alias=self.alias, host=MILVUS_HOST, port=MILVUS_PORT)
            logger.info("Successfully connected to Milvus.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}", exc_info=True)
            raise


    def _create_collection_if_not_exists(self):
        """
        指定されたコレクションが存在しない場合、適切なスキーマで作成する。
        """
        if not utility.has_collection(self.collection_name, using=self.alias):
            logger.info(f"Collection '{self.collection_name}' does not exist. Creating now...")
            
            # プライマリキー (チャンクID)
            pk_field = FieldSchema(
                name="id",  # フィールド名を "id" に変更
                dtype=DataType.VARCHAR, 
                is_primary=True,
                auto_id=False, # 外部からIDを指定
                max_length=512 # 例: message_id_chunk_index (十分な長さを確保)
            )
            # ベクトルフィールド
            vector_field = FieldSchema(
                name="embedding", 
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_dim
            )
            # テキストフィールド (LlamaIndexが参照するため追加)
            text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
 
            # 格納したいメタデータに応じて追加・変更する
            original_message_id_field = FieldSchema(name="message_id", dtype=DataType.VARCHAR, max_length=255)
            channel_id_field = FieldSchema(name="channel_id", dtype=DataType.VARCHAR, max_length=255)
            channel_name_field = FieldSchema(name="channel_name", dtype=DataType.VARCHAR, max_length=255)
            user_id_field = FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=255)
            user_name_field = FieldSchema(name="user_name", dtype=DataType.VARCHAR, max_length=255)
            timestamp_field = FieldSchema(name="timestamp", dtype=DataType.INT64, description="Unix Epoch timestamp in milliseconds")
            guild_id_field = FieldSchema(name="guild_id", dtype=DataType.VARCHAR, max_length=255)

            schema = CollectionSchema(
                fields=[pk_field, vector_field, text_field, original_message_id_field, channel_id_field, channel_name_field, user_id_field, user_name_field, timestamp_field, guild_id_field],
                description=f"Collection Discord server messages ({self.collection_name})",
                enable_dynamic_field=False # これをTrueにするとスキーマにないフィールドもmetadataとして格納可能
            )

            try:
                self.collection = Collection(self.collection_name, schema=schema, using=self.alias)
                logger.info(f"Collection '{self.collection_name}' created successfully.")
                
                # ベクトルフィールドにインデックスを作成 (検索効率のため)
                # IVF_FLATはバランス型、HNSWは高速だがビルド時間とメモリ消費大
                # FLATは小規模データセット向け
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 256}, # nlistはクラスタ数。データサイズに応じて調整
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)
                logger.info(f"Index created for field 'embedding' in collection '{self.collection_name}'.")

                self.collection.load() # ★新規作成後にもロード★
                logger.info(f"Collection '{self.collection_name}' loaded after creation.")

            except Exception as e:
                logger.error(f"Failed to create collection, index or load '{self.collection_name}': {e}", exc_info=True)
                raise
        else:
            logger.info(f"Collection '{self.collection_name}' already exists. Loading...")
            self.collection = Collection(self.collection_name, using=self.alias)
            # コレクションをロードして検索可能にする
            self.collection.load()
            logger.info(f"Collection '{self.collection_name}' loaded.")

    
    def search(self, query: list[float], top_k: int = 10, filters: str = None, output_fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        ベクトル検索を実行し、指定されたクエリに関連するデータを取得する。
        
        Args:
            query: 検索クエリ文字列
            top_k: 返す結果の数
            filters: フィルター式 (例: "channel_id == '1234567890'")
            output_fields: 返すフィールド名のリスト (例: ["id", "text", "channel_name", "user_name", "created_at"])
            
        Returns:
            List[Dict[str, Any]]: 検索結果のリスト
        """
        if not self.collection:
            logger.error("Milvus collection is not initialized. Cannot search.")
            return []
        
        results = self.collection.search(
            data=query,
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            expr=filters,
            output_fields=output_fields
        )
        
        return results
    

    def insert_data(self, data_with_embeddings: List[Dict[str, Any]]) -> List[Any] | None:
        """
        エンベディングを含むデータのリストをMilvusに挿入する。
        各辞書は 'id' (chunk_id), 'embedding', および 'metadata' を含むことを期待する。
        'metadata' 内のキーがスキーマフィールド名と一致する場合、それらがスカラフィールドに格納される。
        enable_dynamic_field=True の場合、スキーマ外のメタデータも格納される。
        """
        if not self.collection:
            logger.error("Milvus collection is not initialized. Cannot insert data.")
            return None
        if not data_with_embeddings:
            logger.info("No data provided for insertion.")
            return []

        logger.info(f"Preparing to insert {len(data_with_embeddings)} records into '{self.collection_name}'.")
        
        # Milvusのinsertメソッドに適した形式にデータを整形
        # スキーマで定義したフィールドを抽出する
        entities_to_insert = []
        for item in data_with_embeddings:
            metadata = item.get("metadata", {})
            entity = {
                "id": item.get("id"), 
                "embedding": item.get("embedding"),
                "text": metadata.get("original_chunk_text") or "", 
                "message_id": str(metadata.get("original_message_id", "")), # Noneの場合も考慮
                "channel_id": str(metadata.get("channel_id", "")),       # Noneの場合も考慮
                "channel_name": metadata.get("channel_name") or "",
                "user_id": str(metadata.get("user_id", "")),             # Noneの場合も考慮
                "user_name": metadata.get("user_name") or "",
                "guild_id": str(metadata.get("guild_id", ""))           # Noneの場合も考慮
            }
            raw_ts = metadata.get("timestamp")
            entity["timestamp"] = int(raw_ts) if raw_ts is not None else 0

            entities_to_insert.append(entity)

        if entities_to_insert:
            logger.info(f"Sample entity to be inserted (first one if multiple): {entities_to_insert[0]}")

        try:
            logger.info(f"Inserting {len(entities_to_insert)} entities...")
            insert_result = self.collection.insert(entities_to_insert)
            self.collection.flush() # データがディスクに書き込まれ、検索可能になるようにフラッシュ
            logger.info(f"Successfully inserted {len(insert_result.primary_keys)} records. PKs: {insert_result.primary_keys[:10]}...")
            return insert_result.primary_keys
        except Exception as e:
            logger.error(f"Failed to insert data into Milvus. Type: {type(e).__name__}, Message: {getattr(e, 'message', str(e))}", exc_info=True)
            return None


    def count_entities(self) -> int:
        """コレクション内のエンティティ数を返す"""
        if not self.collection:
            logger.warning("Milvus collection is not initialized. Cannot count entities.")
            return 0
        try:
            # 確実なカウントのためにフラッシュ推奨
            self.collection.flush() # flushを有効化
            return self.collection.num_entities
        except Exception as e:
            logger.error(f"Failed to count entities in Milvus: {e}", exc_info=True)
            return 0


    def delete_collection(self):
        """現在のインスタンスが指すコレクションを削除する"""
        if utility.has_collection(self.collection_name, using=self.alias):
            logger.info(f"Attempting to drop collection '{self.collection_name}'...")
            
            # コレクションオブジェクトがNoneでなく、かつ実際にMilvusサーバー上に存在する場合
            if self.collection:
                try:
                    # ロードされているかどうかの直接的な判定は難しい場合があるため、
                    # 安全のためにリリースを試みる
                    logger.info(f"Attempting to release collection '{self.collection_name}' before dropping (if loaded).")
                    self.collection.release() # ロードされていなければ何もしないか、エラーにならないはず
                    logger.info(f"Collection '{self.collection_name}' released (or was not loaded).")
                except Exception as e:
                    # release時にエラーが発生しても、dropは試みる (例: コレクションが存在しない、ロードされていない等)
                    logger.warning(f"Could not explicitly release collection '{self.collection_name}', proceeding with drop attempt. Error: {e}")

            utility.drop_collection(self.collection_name, using=self.alias)
            logger.info(f"Collection '{self.collection_name}' dropped successfully.")
            self.collection = None # 削除後はNoneにする
        else:
            logger.info(f"Collection '{self.collection_name}' does not exist, nothing to delete.")

    def close_connection(self):
        """Milvusサーバーとの接続を切断する"""
        if self.alias in connections.list_connections():
            connections.disconnect(self.alias)
            logger.info(f"Disconnected from Milvus server (alias: '{self.alias}').")

    def delete_data_by_guild_id(self, guild_id: str) -> int:
        """
        指定されたguild_idに一致するエンティティをコレクションから削除する。

        Args:
            guild_id (str): 削除対象のギルドID。

        Returns:
            int: 削除されたエンティティの数。エラー時は0を返す。
        """
        if not self.collection:
            logger.warning("Milvus collection is not initialized. Cannot delete data.")
            return 0
        
        expr = f"guild_id == '{guild_id}'"
        logger.info(f"Attempting to delete entities from '{self.collection_name}' with expression: {expr}")
        
        try:
            delete_result = self.collection.delete(expr)
            self.collection.flush() # 変更を永続化
            
            if delete_result.delete_count >= 0: # 成功とみなす (エラー時は例外がスローされる想定)
                logger.info(f"Delete operation for guild_id '{guild_id}' in '{self.collection_name}' completed. MutationResult: {delete_result}")
                return delete_result.delete_count
            else:
                logger.warning(f"Delete operation for guild_id '{guild_id}' in '{self.collection_name}' might not have been successful. MutationResult: {delete_result}")
                return 0 # 失敗または0件影響として扱う

        except Exception as e:
            logger.error(f"Failed to delete data for guild_id '{guild_id}' from Milvus: {e}", exc_info=True)
            return 0
