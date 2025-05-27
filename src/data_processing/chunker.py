import logging
from typing import List, Dict, Any
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

def chunk_messages(cleaned_messages: List[Dict[str, Any]], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    クリーニングされたメッセージリストをチャンクに分割し、メタデータを付与する。
    各メッセージの'content'をチャンキング対象とする。
    """
    if not cleaned_messages:
        return []

    logger.info(f"Starting message chunking. Message count: {len(cleaned_messages)}, Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

    node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_data = []
    message_counter = 0

    # メタデータとして保持するキーのリスト
    # guild_id もRAG検索時のフィルタリングに必要なので追加
    # content は Document の text になるため、metadataからは除外も検討
    # ここでは chunk_metadata には含めるが、Documentのmetadataからは除外する
    essential_metadata_keys = [
        "id", "user_id", "user_name",
        "channel_id", "channel_name", "timestamp", "guild_id" # Changed "created_at" to "timestamp"
    ]

    for msg_data in cleaned_messages:
        message_counter += 1
        content = msg_data.get("content", "")
        if not content.strip(): # 空白のみのメッセージはスキップ
            logger.debug(f"Skipping empty content for message ID {msg_data.get('id')}")
            continue

        # LlamaIndexのDocumentオブジェクトを作成してNodeParserに渡す
        # メタデータは元メッセージの情報を厳選して引き継ぐ
        # contentはDocumentのtextになるため、Documentのmetadataには含めない
        doc_metadata = {key: msg_data.get(key) for key in essential_metadata_keys if key in msg_data}
        doc = Document(text=content, metadata=doc_metadata) 
        
        nodes = node_parser.get_nodes_from_documents([doc])
        
        for i, node in enumerate(nodes):
            chunk_text = node.get_content()
            
            # チャンクごとのメタデータを作成
            # 厳選したメタデータをベースに、チャンク固有の情報を追加
            chunk_metadata = {key: msg_data.get(key) for key in essential_metadata_keys if key in msg_data}
            chunk_metadata["chunk_id"] = f"{msg_data.get('id')}_{i}" # チャンクの一意なID
            chunk_metadata["original_message_id"] = msg_data.get('id')
            chunk_metadata["chunk_text_length"] = len(chunk_text)
            chunk_metadata["original_chunk_text"] = chunk_text # 冗長なので削除も検討 (text_for_embeddingで代替)
            # content もRAGの応答生成時に参照する可能性が低い場合は削除検討 (元メッセージのcontentはmsg_dataにある)
            # chunk_metadata["content"] = content # 必要であれば追加
            
            chunk_entry = {
                "id": chunk_metadata["chunk_id"],
                "text_for_embedding": chunk_text,
                "metadata": chunk_metadata
            }
            chunked_data.append(chunk_entry)
            logger.debug(f"Created chunk: {chunk_metadata['chunk_id']} from message {msg_data.get('id')}")

        if message_counter % 100 == 0:
            logger.info(f"Processed {message_counter}/{len(cleaned_messages)} messages for chunking...")

    logger.info(f"Message chunking finished. Total chunks created: {len(chunked_data)}")
    return chunked_data
