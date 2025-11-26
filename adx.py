import time
import json
import requests

URL = "https://adx.cryptothunder.net/bot_adx"
OUTPUT_FILE = "adx.json"
INTERVAL_SECONDS = 1  


def main():
    print(f"Start feed data from {URL} for {INTERVAL_SECONDS} seconds. "
          f"Save to file: {OUTPUT_FILE}")
    try:
        while True:
            try:
                response = requests.get(URL, timeout=5)
                response.raise_for_status()

                data = response.json()

                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Saved {OUTPUT_FILE}")
            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}")

            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped (CTRL+C).")


if __name__ == "__main__":
    main()
