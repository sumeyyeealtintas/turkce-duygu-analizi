"""Eğitilmiş modelle yazdığınız cümlelerin duygusunu tahmin eder.

Çalıştırma:  python src/tahmin.py "Kargo çok hızlıydı, ürün harika!"
             python src/tahmin.py            (etkileşimli mod, çıkmak için boş satır)
"""
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metin  # noqa: E402,F401  (kayıtlı modelin kullandığı `temizle` fonksiyonu için)

MODEL_YOLU = Path(__file__).resolve().parent.parent / "model" / "duygu_modeli.joblib"


def modeli_yukle():
    if not MODEL_YOLU.exists():
        raise SystemExit("Model bulunamadı. Önce şunu çalıştırın:  python src/egit.py")
    return joblib.load(MODEL_YOLU)


def tahmin_et(model, cumleler: list[str]) -> list[str]:
    return list(model.predict(cumleler))


if __name__ == "__main__":
    model = modeli_yukle()
    if len(sys.argv) > 1:
        girdiler = [" ".join(sys.argv[1:])]
    else:
        girdiler = []
        print("Cümle yazın (çıkmak için boş satır):")
        while (satir := input("> ").strip()):
            girdiler.append(satir)
    for cumle, sonuc in zip(girdiler, tahmin_et(model, girdiler)):
        print(f"[{sonuc}] {cumle}")
