import os
import logging
from telegram import Bot
from telegram.constants import ParseMode

logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


async def send_alert(message):

    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN missing")
        return

    if not CHAT_ID:
        logger.error("TELEGRAM_CHAT_ID missing")
        return

    try:

        bot = Bot(token=TOKEN)

        await bot.send_message(
            chat_id=CHAT_ID,
            text=message,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True
        )

        logger.info("✅ Telegram alert sent")

    except Exception as e:

        logger.exception(
            f"Telegram send failed: {e}"
        )
