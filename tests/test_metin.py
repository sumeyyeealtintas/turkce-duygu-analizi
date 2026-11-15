import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from metin import temizle, turkce_kucuk_harf  # noqa: E402


def test_turkce_i_harfleri():
    # Python'un lower() fonksiyonu burada yanlış sonuç verir: "ISPARTA".lower() == "isparta"
    assert turkce_kucuk_harf("ISPARTA") == "ısparta"
    assert turkce_kucuk_harf("İSTANBUL") == "istanbul"
    assert turkce_kucuk_harf("Işık") == "ışık"


def test_python_lower_turkce_icin_yanlis_oldugunu_gosterir():
    assert "ISPARTA".lower() != "ısparta"


def test_url_etiket_ve_mention_temizlenir():
    t = temizle("Harika! https://ornek.com/x @kullanici #urun çok iyi")
    assert "http" not in t and "@" not in t and "#" not in t
    assert "harika" in t and "iyi" in t


def test_tekrar_eden_harfler_kisaltilir():
    assert temizle("çoooooook güzel") == "çook güzel"


def test_sayilar_tek_belirtece_indirgenir():
    assert temizle("5 yıldız 100 puan") == "0 yıldız 0 puan"


def test_fazla_bosluk_ve_bos_girdi():
    assert temizle("  merhaba   dünya  ") == "merhaba dünya"
    assert temizle(None) == ""
    assert temizle(123) == ""
