import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch
#from src.config import EMBEDDING_MODEL_NAME
logger = logging.getLogger(__name__)
EMBEDDING_MODEL_NAME = "cl-nagoya/ruri-v3-310m"

# Embeddingモデルに応じたプレフィックス処理
# (将来的にはモデル名から動的にプレフィックスを決定するロジックも検討)
DEFAULT_QUERY_PREFIX = "query: "
DEFAULT_PASSAGE_PREFIX = "passage: "

# ruri-v3-310m 用のプレフィックス
RURI_QUERY_PREFIX = "検索クエリ: "
RURI_PASSAGE_PREFIX = "検索文書: "

def get_prefix_for_embedding_model(model_name: str) -> tuple[str, str]:
    """Embeddingモデル名に基づいてクエリとパッセージのプレフィックスを返す"""
    model_name_lower = model_name.lower()
    if "e5-large-instruct" in model_name_lower:
        return DEFAULT_QUERY_PREFIX, DEFAULT_PASSAGE_PREFIX
    elif "ruri-v3" in model_name_lower: # ruri-v3 シリーズを想定
        return RURI_QUERY_PREFIX, RURI_PASSAGE_PREFIX
    # 他のモデルに対応する場合はここに追加
    return "", "" # デフォルトはプレフィックスなし

class MessageEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str | None = None):
        """
        device: 'cuda', 'cpu', None (自動検出)
        """
        self.model_name = model_name # model_name をインスタンス変数として保存
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing SentenceTransformer model: {self.model_name} on device: {self.device}")
        try:
            self.query_prefix_for_model, self.document_prefix_for_model = get_prefix_for_embedding_model(self.model_name) # 両方取得
            self.model = SentenceTransformer(model_name_or_path=self.model_name, device=self.device)
            logger.info(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model {self.model_name}: {e}")
            raise

    @property
    def name(self) -> str:
        return self.model_name

    def embed_text(self, text: str, batch_size: int = 64) -> List[List[torch.Tensor]]:
        """
        テキストを受け取り、ベクトル化する。
        """
        if not hasattr(self, 'model'): # モデルが正常にロードされていない場合
            logger.error("Embedding model is not loaded. Cannot perform embeddings.")
            return []

        try:
            return self.model.encode(text, batch_size=batch_size, convert_to_tensor=True).tolist()
        except Exception as e:
            logger.error(f"Error during model.encode: {e}")
            return []
        

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

        texts_to_embed = [self.document_prefix_for_model + chunk["text_for_embedding"] for chunk in chunked_data]

        try:
            embeddings = self.embed_text(texts_to_embed, batch_size=64)
        except Exception as e:
            logger.error(f"Error during model.encode: {e}")
            return chunked_data # or raise an error

        if len(embeddings) != len(chunked_data):
            logger.error("Mismatch between number of embeddings and number of chunks.")
            return chunked_data # or raise an error

        for i, chunk in enumerate(chunked_data):
            emb = embeddings[i]
            chunk["embedding"] = emb

        logger.info(f"Embedding finished. Embeddings generated for {len(chunked_data)} chunks.")
        return chunked_data


if __name__ == "__main__":
    embedder = MessageEmbedder()
    text = "こんにちは"
    embeddings = embedder.embed_text(text)
    print(embeddings)