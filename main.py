import asyncio
import schedule
import time

from exchange import get_exchange
from telegram_alerts import send_alert

exchange = get_exchange()

async def startup():

    await send_alert(
        "🚀 SMC Whale AI Started"
    )

def heartbeat():

    print("Worker Alive")

def main():

    asyncio.run(startup())

    schedule.every(1).minutes.do(heartbeat)

    while True:

        schedule.run_pending()

        time.sleep(5)

if __name__ == "__main__":
    main()
