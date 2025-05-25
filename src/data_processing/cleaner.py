import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# URLのみかどうかを判定する正規表現 (簡易版)
# より厳密な判定が必要な場合はライブラリ (例: furl, yarl) の使用も検討
URL_ONLY_PATTERN = re.compile(r"^https?://[\w\.\-/:#\?=&%~]+$")

def is_url_only(text: str) -> bool:
    """文字列がURLのみで構成されているか判定する"""
    return bool(URL_ONLY_PATTERN.fullmatch(text.strip()))

def clean_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    収集されたメッセージリストに対してクリーニング処理を適用する。

    クリーニングルール:
    1. ボット自身の発言は無視 (bot_idが指定されている場合)
    2. 他のBotの発言も全て無視
    3. メッセージ内容がURLのみの場合は無視
    """
    cleaned_messages = []
    if not messages:
        return cleaned_messages

    logger.info(f"Starting message cleaning. Original message count: {len(messages)}")
    
    # botIDが設定されていれば、取得しておく (現状はconfigから直接は読まない想定)
    # 将来的にはconfigから取得するか、Botクライアントから動的に取得する

    for msg in messages:

        # ルール1: 他のBotの発言を無視
        if msg.get("author_is_bot", False):
            logger.debug(f"Skipping message from other bot (Author ID: {msg.get('author_id')}, Name: {msg.get('author_name')}, Msg ID: {msg.get('id')})")
            continue

        # ルール2: メッセージ内容がURLのみの場合は無視
        content = msg.get("content", "")
        if is_url_only(content):
            logger.debug(f"Skipping URL-only message (ID: {msg.get('id')}, Content: {content})")
            continue
        
        # すべてのルールをパスしたメッセージを追加
        cleaned_messages.append(msg)

    logger.info(f"Message cleaning finished. Cleaned message count: {len(cleaned_messages)}")
    return cleaned_messages
