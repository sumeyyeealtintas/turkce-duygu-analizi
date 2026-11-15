"""Türkçe metin ön işleme.

Türkçe'ye özgü tuzak: Python'un `str.lower()` fonksiyonu "I" harfini "i" yapar, oysa Türkçe'de
"I" -> "ı" ve "İ" -> "i" olmalıdır. ("ISPARTA".lower() -> "ispartа" hatalı, doğrusu "ısparta".)
Bunu düzeltmeden sözcük sayımı yaparsak aynı sözcük iki farklı özellik olarak sayılır.
"""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_ETIKET = re.compile(r"[@#]\w+")
_TEKRAR = re.compile(r"(.)\1{2,}")      # "çoooook" -> "çook" (3+ tekrarı 2'ye indirir)
_FAZLA_BOSLUK = re.compile(r"\s+")
_SAYI = re.compile(r"\d+")


def turkce_kucuk_harf(metin: str) -> str:
    """Türkçe kurallarına uygun küçük harfe çevirme."""
    return metin.replace("İ", "i").replace("I", "ı").lower()


def temizle(metin: str) -> str:
    """Modele girmeden önce metni sadeleştirir."""
    if not isinstance(metin, str):
        return ""
    metin = turkce_kucuk_harf(metin)
    metin = _URL.sub(" ", metin)
    metin = _ETIKET.sub(" ", metin)
    metin = _SAYI.sub(" 0 ", metin)
    metin = _TEKRAR.sub(r"\1\1", metin)
    return _FAZLA_BOSLUK.sub(" ", metin).strip()
