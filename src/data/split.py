"""Train/val/test bölme — scenario bazlı (gruplu), tekrar üretilebilir.

Neden gruplu split: LARD subset'imizde yalnızca ~20 benzersiz scenario/airport var ve
her sahne onlarca kareye sahip. Rastgele bölme aynı havaalanını hem train hem test'e
koyar → veri sızıntısı → yanıltıcı iyimser metrik. Sahneyi grup kabul edip
`GroupShuffleSplit` ile bölüyoruz: bir sahne yalnızca tek bir split'te bulunur.

Ödün: sahne sayısı az olduğundan bölme oranları tam 70/15/15 tutmaz ve test seti az
sayıda havaalanı kapsar. Bu bilinçli bir tercih; sınır TESTING.md'de dürüstçe belirtilir.

Çıktı: labels.csv'ye `split` sütunu eklenir (train/val/test).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def assign_splits(
    df: pd.DataFrame,
    group_col: str = "scenario",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """DataFrame'e scenario bazlı `split` sütunu ekler.

    Args:
        df: labels.csv içeriği.
        group_col: Sızıntıyı önlemek için grup sütunu (sahne).
        val_frac: Validation oranı (yaklaşık, görüntü bazında).
        test_frac: Test oranı (yaklaşık).
        seed: Tekrar üretilebilirlik.

    Returns:
        `split` sütunu eklenmiş DataFrame.
    """
    groups = df[group_col].astype(str).values

    # 1) test'i ayır (grupları bölmeden)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))

    # 2) kalanı train/val'e böl (val oranını kalan içindeki paya çevir)
    trainval = df.iloc[trainval_idx]
    val_ratio_within = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio_within, random_state=seed)
    tr_rel, val_rel = next(gss2.split(trainval, groups=groups[trainval_idx]))

    split = pd.Series(index=df.index, dtype="object")
    split.iloc[test_idx] = "test"
    split.iloc[trainval_idx[tr_rel]] = "train"
    split.iloc[trainval_idx[val_rel]] = "val"

    df = df.copy()
    df["split"] = split.values
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario bazlı train/val/test split")
    parser.add_argument("--labels", type=Path, default=Path("data/lard/labels.csv"))
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    df = assign_splits(df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    df.to_csv(args.labels, index=False)

    print("Split tamamlandı (scenario bazlı, sızıntısız):")
    print(df["split"].value_counts().to_string())
    # Sızıntı kontrolü: hiçbir sahne birden fazla split'te olmamalı
    leak = df.groupby("scenario")["split"].nunique()
    n_leak = int((leak > 1).sum())
    print(f"Sızıntı kontrolü — birden fazla split'te olan sahne: {n_leak} (0 olmalı)")


if __name__ == "__main__":
    main()
