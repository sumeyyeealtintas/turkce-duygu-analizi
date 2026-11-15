"""Veri setini indirir ve yükler.

Veri: winvoker/turkish-sentiment-analysis-dataset (Hugging Face), CC BY-SA 4.0 lisanslıdır.
Lisans gereği veriyi bu repoya KOYMUYORUZ; ilk çalıştırmada indirilir ve data/ klasöründe tutulur.
"""
from pathlib import Path

import pandas as pd
import requests

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data"
TABAN = "https://huggingface.co/datasets/winvoker/turkish-sentiment-analysis-dataset/resolve/main"

# Veri setindeki etiketler -> Türkçe sınıf adları
ETIKETLER = {"Positive": "olumlu", "Notr": "nötr", "Negative": "olumsuz"}


def indir(dosya: str) -> Path:
    hedef = VERI / dosya
    if hedef.exists():
        return hedef
    VERI.mkdir(exist_ok=True)
    print(f"{dosya} indiriliyor...")
    with requests.get(f"{TABAN}/{dosya}", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(hedef, "wb") as f:
            for parca in r.iter_content(1 << 20):
                f.write(parca)
    return hedef


def yukle(dosya: str, ornek: int | None = None, seed: int = 42) -> pd.DataFrame:
    """CSV'yi okur, etiketleri Türkçeleştirir; `ornek` verilirse o kadar satır örnekler."""
    df = pd.read_csv(indir(dosya)).dropna(subset=["text", "label"])
    df["etiket"] = df["label"].map(ETIKETLER)
    df = df.dropna(subset=["etiket"])
    if ornek and len(df) > ornek:
        df = df.sample(ornek, random_state=seed)
    return df.reset_index(drop=True)
