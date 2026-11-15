"""Türkçe duygu analizi modellerini eğitir ve dürüstçe değerlendirir.

Çalıştırma:  python src/egit.py            (varsayılan: 120.000 eğitim örneği)
             python src/egit.py 300000     (daha fazla örnek)
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metin import temizle  # noqa: E402
from veri import yukle  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
RAPOR = KOK / "raporlar"
MODEL = KOK / "model"
SINIFLAR = ["olumsuz", "nötr", "olumlu"]


def modelleri_olustur() -> dict:
    return {
        "Çoğunluk sınıfı (referans)": Pipeline([
            ("vek", TfidfVectorizer(preprocessor=temizle, max_features=1000)),
            ("sinif", DummyClassifier(strategy="most_frequent")),
        ]),
        # Sözcük tabanlı: sözcük ve ikili sözcük öbekleri
        "Sözcük TF-IDF + Lojistik Regresyon": Pipeline([
            ("vek", TfidfVectorizer(preprocessor=temizle, ngram_range=(1, 2), min_df=3,
                                    max_features=300_000, sublinear_tf=True)),
            ("sinif", LogisticRegression(max_iter=300, class_weight="balanced", C=5)),
        ]),
        # Karakter tabanlı: Türkçe gibi eklemeli dillerde ("güzeldi", "güzelmiş" -> "güzel") işe yarar
        "Karakter TF-IDF + Lineer SVM": Pipeline([
            ("vek", TfidfVectorizer(preprocessor=temizle, analyzer="char_wb", ngram_range=(2, 5),
                                    min_df=3, max_features=400_000, sublinear_tf=True)),
            ("sinif", LinearSVC(class_weight="balanced", C=0.5)),
        ]),
    }


def main(ornek: int = 120_000) -> None:
    RAPOR.mkdir(exist_ok=True)
    MODEL.mkdir(exist_ok=True)

    egitim = yukle("train.csv", ornek=ornek)
    test = yukle("test.csv")
    # Veri sızıntısı kontrolü: eğitimde de bulunan test metinlerini testten çıkar
    ham_egitim = set(yukle("train.csv")["text"])
    onceki = len(test)
    test = test[~test["text"].isin(ham_egitim)].reset_index(drop=True)
    print(f"Eğitim: {len(egitim):,} | Test: {len(test):,} (eğitimle çakışan {onceki - len(test)} metin çıkarıldı)")

    sonuclar, tahminler, egitilen = {}, {}, {}
    kutup_maske = test["etiket"].isin(["olumlu", "olumsuz"]).to_numpy()  # asıl duygu görevi

    for ad, model in modelleri_olustur().items():
        t0 = time.time()
        model.fit(egitim["text"], egitim["etiket"])
        tahmin = model.predict(test["text"])
        sure = time.time() - t0
        tahminler[ad], egitilen[ad] = tahmin, model

        y = test["etiket"].to_numpy()
        sonuclar[ad] = {
            "dogruluk": float(accuracy_score(y, tahmin)),
            "makro_f1_3sinif": float(f1_score(y, tahmin, average="macro")),
            # Yalnızca gerçekte olumlu/olumsuz olan metinler; olumlu mu olumsuz mu ayırt edebiliyor mu?
            "makro_f1_olumlu_olumsuz": float(
                f1_score(y[kutup_maske], tahmin[kutup_maske], labels=["olumlu", "olumsuz"], average="macro")
            ),
            "olumsuz_f1": float(f1_score(y, tahmin, labels=["olumsuz"], average=None)[0]),
            "sure_sn": round(sure, 1),
        }
        print(f"\n{ad}  ({sure:.0f} sn)")
        for k, v in sonuclar[ad].items():
            if k != "sure_sn":
                print(f"   {k:26s} {v:.3f}")

    en_iyi = max(sonuclar, key=lambda k: sonuclar[k]["makro_f1_olumlu_olumsuz"])
    print(f"\nEn iyi (olumlu/olumsuz makro-F1'e göre): {en_iyi}")

    # Kaynağa göre doğruluk: modelin hangi alanda zorlandığını gösterir
    test["tahmin"] = tahminler[en_iyi]
    test["dogru"] = test["tahmin"] == test["etiket"]
    kaynak = test.groupby("dataset")["dogru"].agg(["mean", "size"]).sort_values("size", ascending=False)
    kaynak = kaynak[kaynak["size"] >= 100]
    print("\nKaynağa göre doğruluk:\n", kaynak.round(3).to_string())

    # Grafikler
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        test["etiket"], test["tahmin"], labels=SINIFLAR, normalize="true",
        cmap="Blues", ax=ax, values_format=".2f")
    ax.set_title(f"Karışıklık Matrisi (satır bazlı oran)\n{en_iyi}", fontsize=9)
    fig.tight_layout()
    fig.savefig(RAPOR / "karisiklik_matrisi.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    kaynak["mean"].sort_values().plot.barh(ax=ax, color="#2f6fed")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Doğruluk")
    ax.set_title("Veri kaynağına göre doğruluk")
    fig.tight_layout()
    fig.savefig(RAPOR / "kaynaga_gore_dogruluk.png", dpi=150)
    plt.close(fig)

    # En ayırt edici sözcükler (sözcük modelinden)
    sozcuk = egitilen["Sözcük TF-IDF + Lojistik Regresyon"]
    adlar = np.array(sozcuk["vek"].get_feature_names_out())
    lr = sozcuk["sinif"]
    kayit = {}
    for i, sinif in enumerate(lr.classes_):
        if sinif == "nötr":
            continue
        en = np.argsort(lr.coef_[i])[-12:][::-1]
        kayit[sinif] = adlar[en].tolist()
    print("\nEn ayırt edici sözcükler:", json.dumps(kayit, ensure_ascii=False))

    (RAPOR / "metrikler.json").write_text(
        json.dumps({"egitim_ornek": len(egitim), "test_ornek": len(test), "en_iyi": en_iyi,
                    "sonuclar": sonuclar, "ayirt_edici_sozcukler": kayit},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(egitilen[en_iyi], MODEL / "duygu_modeli.joblib", compress=3)
    print("Model kaydedildi.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120_000)
