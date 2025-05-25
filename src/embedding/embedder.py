import logging
from typing import List, Dict, Any
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from src.config import EMBEDDING_MODEL_NAME
from src.data_processing.chunker import get_prefix_for_embedding_model
logger = logging.getLogger(__name__)

class MessageEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str | None = None):
        """
        device: 'cuda', 'cpu', None (自動検出)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing SentenceTransformer model: {model_name} on device: {self.device}")
        try:
            query_prefix_for_model, document_prefix_for_model = get_prefix_for_embedding_model(model_name) # 両方取得
            self.model = HuggingFaceEmbedding(
                model_name=model_name, 
                device=self.device,
                query_instruction=query_prefix_for_model, # モデル推奨のクエリプレフィックス
                text_instruction=document_prefix_for_model,  # モデル推奨のドキュメントプレフィックス
                )
            logger.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model {model_name}: {e}")
            raise

    def embed_chunks(self, chunked_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        チャンクデータのリストを受け取り、各チャンクの'text_for_embedding'をベクトル化する。
        結果は元のチャンクデータに 'embedding' キーとして追加される。
        """
        if not chunked_data:
            return []
        if not hasattr(self, 'model'): # モデルが正常にロードされていない場合
            logger.error("Embedding model is not loaded. Cannot perform embeddings.")
            return chunked_data # or raise an error

        logger.info(f"Starting embedding for {len(chunked_data)} chunks...")

        texts_to_embed = [chunk["text_for_embedding"] for chunk in chunked_data]
        
        try:
            embeddings = [self.model.get_text_embedding(text) for text in texts_to_embed]
        except Exception as e:
            logger.error(f"Error during model.encode: {e}")
            return chunked_data # or raise an error

        if len(embeddings) != len(chunked_data):
            logger.error("Mismatch between number of embeddings and number of chunks.")
            return chunked_data # or raise an error

        for i, chunk in enumerate(chunked_data):
            emb = embeddings[i]
            # NumPy 配列なら tolist()、すでにリストならそのまま
            chunk["embedding"] = emb.tolist() if hasattr(emb, "tolist") else emb

        logger.info(f"Embedding finished. Embeddings generated for {len(chunked_data)} chunks.")
        return chunked_data
