import json
import os
import requests
from pathlib import Path
from urllib.parse import urlparse

# ====== POSTAVKE ======
JSON_FILES = {
    "train": "C:/Users/user/Downloads/Face Anti-Spoofing.v4i.openai/_annotations.train.jsonl",
    "test": "C:/Users/user/Downloads/Face Anti-Spoofing.v4i.openai/_annotations.test.jsonl",
    "valid": "C:/Users/user/Downloads/Face Anti-Spoofing.v4i.openai/_annotations.valid.jsonl",
}

DOWNLOADS_DIR = Path.home() / "Downloads"
TIMEOUT = 15
# ====================


def get_next_index(folder):
    """
    Pronalazi sljedeći slobodni broj u folderu (1,2,3,...)
    """
    existing_numbers = []

    for file in folder.iterdir():
        if file.is_file() and file.stem.isdigit():
            existing_numbers.append(int(file.stem))

    return max(existing_numbers, default=0) + 1


def extract_image_url(record):
    messages = record.get("messages", [])

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "image_url"
                    and "image_url" in item
                    and "url" in item["image_url"]
                ):
                    return item["image_url"]["url"]
    return None


def download_image(url, save_dir, index):
    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()

        suffix = Path(urlparse(url).path).suffix or ".jpg"
        filename = f"{index}{suffix}"
        save_path = save_dir / filename

        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"✔ Preuzeto: {filename}")

    except Exception as e:
        print(f"✖ Greška kod {url}: {e}")


def process_jsonl(jsonl_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    index = get_next_index(output_dir)

    with open(jsonl_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
                url = extract_image_url(record)

                if not url:
                    print(f"⚠ Linija {line_number}: URL nije pronađen")
                    continue

                download_image(url, output_dir, index)
                index += 1

            except json.JSONDecodeError:
                print(f"✖ Neispravan JSON u liniji {line_number}")


def main():
    for split, json_file in JSON_FILES.items():
        print(f"\n📂 Obrada: {split}")

        json_path = Path(json_file)
        if not json_path.exists():
            print(f"✖ Datoteka ne postoji: {json_file}")
            continue

        output_dir = DOWNLOADS_DIR / split
        process_jsonl(json_path, output_dir)


if __name__ == "__main__":
    main()
