import discord
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class DiscordCrawler:
    def __init__(self, client: discord.Client):
        """
        DiscordCrawlerを初期化します。
        client: 呼び出し元で使用しているdiscord.Clientインスタンス
        """
        self.client = client
        self.collected_messages: List[Dict[str, Any]] = []

    async def crawl_specific_guild(self, guild: discord.Guild, max_messages_per_channel: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        指定されたサーバー内の全テキスト/フォーラムチャンネルおよびスレッドからメッセージを収集します。
        guild: 対象のdiscord.Guildオブジェクト
        max_messages_per_channel: 各チャンネルから取得する最大メッセージ数（Noneで全件）
        戻り値: メッセージデータのリスト
        """
        
        self.collected_messages = []
        logger.info(f'Starting crawl for server: {guild.name} (ID: {guild.id}), limit={max_messages_per_channel}')

        # 通常のテキストチャンネルとフォーラムチャンネル
        channels_to_crawl = list(guild.channels) + list(guild.threads)

        skipped_channels: List[str] = []
        # 各チャンネル・スレッドからクロール
        for channel in channels_to_crawl:
            if not channel.permissions_for(guild.me).read_messages:
                logger.warning(f"Missing read permission: {channel.name} (ID: {channel.id}), skipping.")
                skipped_channels.append(channel.name)
                continue
            
            if channel.type == discord.ChannelType.category:
                continue
            await self._crawl_channel(channel, max_messages_per_channel)
            await asyncio.sleep(0.5)

        if skipped_channels:
            logger.info(f"Skipped {len(skipped_channels)} channels due to permissions: {', '.join(skipped_channels)}")

        logger.info(f"Crawl complete for {guild.name}. Total messages collected: {len(self.collected_messages)}")
        return self.collected_messages
    
    
    def _set_channel_id(self, channel):
        """
        チャンネルIDを設定します。
        """
        if channel.type == discord.ChannelType.public_thread or channel.type == discord.ChannelType.private_thread:
            return channel.parent.id
        else:
            return channel.id
        
    
    def _set_channel_name(self, channel):
        """
        チャンネル名を設定します。
        """
        if channel.type == discord.ChannelType.public_thread or channel.type == discord.ChannelType.private_thread:
            return channel.parent.name
        else:
            return channel.name
        
        
    def _resolve_mentions(self, message: discord.Message) -> str:
        """
        メッセージ内のユーザー、チャンネル、ロールメンション(<#ID>, <@ID>, <@!ID>, <@&ID>)を名前表記に変換
        """
        content = message.content
        # チャンネルメンション
        for ch in message.channel_mentions:
            content = content.replace(f'<#{ch.id}>', f'#{ch.name}')
        # ユーザーメンション
        for user in message.mentions:
            content = content.replace(f'<@{user.id}>', f'@{user.name}')
            content = content.replace(f'<@!{user.id}>', f'@{user.name}')
        # ロールメンション
        for role in message.role_mentions:
            content = content.replace(f'<@&{role.id}>', f'@&{role.name}')
        return content
        

    async def _crawl_channel(self, channel, max_messages_per_channel: Optional[int]):
        """
        単一チャンネルまたはスレッドからメッセージを収集。
        max_messages_per_channel: 取得上限
        """
        logger.info(f"Crawling {channel.type}: #{channel.name} (ID: {channel.id})")
        count = 0
        try:
            limit = max_messages_per_channel if max_messages_per_channel is not None else None
            async for message in channel.history(limit=limit):
                resolved_content = self._resolve_mentions(message)
                message_data = {
                    "id": message.id,
                    "content": resolved_content,
                    "user_id": message.author.id,
                    "user_name": message.author.name,
                    "author_is_bot": message.author.bot,
                    "channel_id": self._set_channel_id(channel),
                    "channel_name": self._set_channel_name(channel),
                    "timestamp": int(message.created_at.timestamp() * 1000),
                    "attachments": [att.url for att in message.attachments],
                    "jump_url": message.jump_url,
                }
                self.collected_messages.append(message_data)
                count += 1
                # 500件ごとにログ出力
                if count % 500 == 0:
                    logger.info(f"Collected {count} messages so far from {channel.name}")
            logger.info(f"Finished {channel.type} {channel.name}: collected {count} messages.")
        except discord.Forbidden:
            logger.error(f"Forbidden to read {channel.name} (ID: {channel.id})")
        except discord.HTTPException as e:
            logger.error(f"HTTPException in {channel.name}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in {channel.name}: {e}", exc_info=True)
