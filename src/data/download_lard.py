"""LARD V2 subset indirici.

HuggingFace `DEEL-AI/LARD_V2` veri setinden küçük bir subset indirir:

    data/lard/
        images/00000.jpg ...
        labels.csv          # dosya adı + 4 köşe + metadata

Yöntem: Önce streaming denedik ama HF bağlantısı büyük parquet shard'larını tamponlarken
sürekli düşüyordu (peer closed connection) → çok yavaş. Bunun yerine parquet shard'larını
`hf_hub_download` ile doğrudan indiriyoruz: resumable + cache'li, bağlantı düşmelerine
dayanıklı. Bir shard ~1700 satır içerdiğinden config başına tek shard yeterli.

Etiket formatı: LARD maske vermez, pistin 4 köşesini piksel koordinatı olarak verir.
Maske üretimi ayrı bir adımda yapılır (bkz. masks.py).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO = "DEEL-AI/LARD_V2"

# V2'nin sentetik kaynakları (config isimleri). Çeşitlilik için hepsinden dengeli çekiyoruz.
CONFIGS: list[str] = ["arcgis", "bingmaps", "flsim", "ges", "xplane"]

# labels.csv'ye yazılacak sütunlar: köşeler + failure analizinde işe yarayacak metadata.
CORNER_COLS = ["x_TL", "y_TL", "x_TR", "y_TR", "x_BR", "y_BR", "x_BL", "y_BL"]
META_COLS = [
    "type", "airport", "runway", "scenario",
    "slant_distance", "along_track_distance", "height_above_runway",
    "lateral_path_angle", "vertical_path_angle",
    "night", "weather", "runway_in_cone",
]
CSV_HEADER = ["filename", "width", "height", "source_config"] + CORNER_COLS + META_COLS


def _shard_files(all_files: list[str], config: str, split: str) -> list[str]:
    """Bir config/split için parquet shard dosya yollarını sıralı döndürür."""
    prefix = f"{config}/{split}-"
    return sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".parquet"))


def download_subset(
    out_dir: Path,
    n_total: int,
    split: str,
    configs: list[str],
    seed: int,
) -> int:
    """Her config'ten dengeli örnek alıp subset indirir (shard tabanlı, resumable).

    Args:
        out_dir: Çıktı kök dizini (images/ ve labels.csv burada oluşur).
        n_total: Toplam indirilecek görüntü sayısı.
        split: HF split adı ("train" veya "test").
        configs: Çekilecek config (kaynak) listesi.
        seed: Örnek seçimi için seed (tekrar üretilebilirlik).

    Returns:
        Kaydedilen görüntü sayısı.
    """
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    all_files = HfApi().list_repo_files(REPO, repo_type="dataset")
    per_config = math.ceil(n_total / len(configs))

    rows: list[dict] = []
    idx = 0

    for cfg in configs:
        shards = _shard_files(all_files, cfg, split)
        if not shards:
            print(f"  UYARI: {cfg}/{split} için shard bulunamadı, atlanıyor.")
            continue

        # Yeterli satır toplanana kadar shard indir (genelde tek shard yeter).
        collected: list[pd.DataFrame] = []
        for shard in shards:
            local = hf_hub_download(REPO, shard, repo_type="dataset")  # resumable + cache
            collected.append(pd.read_parquet(local))
            if sum(len(d) for d in collected) >= per_config:
                break

        # Sabit seed ile karıştır ve config kotası kadarını al (sahne çeşitliliği için).
        df = pd.concat(collected, ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed).head(per_config)

        for _, ex in df.iterrows():
            filename = f"{idx:05d}.jpg"
            img = Image.open(io.BytesIO(ex["image"]["bytes"])).convert("RGB")
            img.save(img_dir / filename, quality=95)

            row = {"filename": filename, "width": int(ex["width"]),
                   "height": int(ex["height"]), "source_config": cfg}
            for col in CORNER_COLS + META_COLS:
                val = ex.get(col)
                # pandas NaN -> boş bırak (CSV'de temiz görünsün)
                row[col] = "" if pd.isna(val) else val
            rows.append(row)

            idx += 1
            if idx % 50 == 0:
                print(f"  {idx}/{n_total} kaydedildi...")

    with open(out_dir / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="LARD V2 subset indirici (shard tabanlı)")
    parser.add_argument("--out", type=Path, default=Path("data/lard"),
                        help="Çıktı dizini (varsayılan: data/lard)")
    parser.add_argument("--n", type=int, default=800,
                        help="Toplam görüntü sayısı (varsayılan: 800)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"],
                        help="HF split (varsayılan: train)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Tekrar üretilebilirlik için seed")
    args = parser.parse_args()

    print(f"LARD V2 subset indiriliyor: n={args.n}, split={args.split}, "
          f"kaynaklar={CONFIGS}")
    saved = download_subset(args.out, args.n, args.split, CONFIGS, args.seed)
    print(f"Bitti. {saved} görüntü + labels.csv -> {args.out}")


if __name__ == "__main__":
    main()
