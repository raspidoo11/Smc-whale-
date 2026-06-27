from telegram import Bot
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

bot = Bot(token=TELEGRAM_BOT_TOKEN)

async def send_alert(message):

    try:

        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message
        )

    except Exception as e:

        print(f"Telegram Error: {e}")
