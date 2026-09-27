"""Unit tests: python -m pytest tests/  (or python tests/test_core.py)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

from src.blocking import build_index, generate_candidates
from src.normalize import normalize_address, normalize_frame, normalize_name
from src.score import f_beta_half, macro_score

HARD_NAMES = [("Sharma Pvt. Ltd.", "SHARMA PRIVATE LIMITED"), ("Shri Ganesh Traders & Co", "Sri Ganesh Traders and Company"),
              ("Acme Corp", "ACME Corporation"), ("S.B.I. Life", "SBI Life"), ("Société Générale SARL", "Societe Generale")]


def test_name_equivalences():
    for a, b in HARD_NAMES:
        na, nb = normalize_name(a), normalize_name(b)
        assert na["name_core"] == nb["name_core"] or na["name_phon"] == nb["name_phon"], (a, b, na, nb)


def test_name_idempotent():
    for a, _ in HARD_NAMES:
        full = normalize_name(a)["name_full"]
        assert normalize_name(full)["name_full"] == full


def test_address_components():
    a = normalize_address("12, MG Rd, Near SBI ATM, Indore 452 001")
    b = normalize_address("12 Mahatma Gandhi Road, Indore, MP 452001")
    assert a["addr_postcode"] == b["addr_postcode"] == "452001"
    assert a["addr_numbers"] == b["addr_numbers"] == "12"
    assert "sbi" in a["addr_landmarks"] and "near" not in a["addr_core"]
    assert normalize_address("1600 Pennsylvania Ave NW, DC 20500-0003")["addr_postcode"] == "20500"
    assert normalize_address("15 Rue de Rivoli, 75004 Paris")["addr_postcode"] == "75004"
    assert normalize_address("")["addr_core"] == ""


def test_metric():
    assert abs(f_beta_half({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) - 0.714) < 1e-3
    assert macro_score({}, {"a": set(), "b": {"x"}}) == 0.5


def _frame(rows):
    return normalize_frame(pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"]))


def test_unseen_country_gets_candidates_and_rare_index():
    cfg = dict(country_mode="same_first", cross_country_k=2, char_tfidf_k=5, word_tfidf_k=5,
               char_ngram_range=[3, 5], rare_token_max_df=0.5, rare_token_k=5, key_block_max_size=50,
               chunk_size=100)
    s1 = _frame([("S1-1", "Boulangerie Zephyrine SARL", "3 Rue X, 75001 Paris", "France"),
                 ("S1-2", "Acme Corp", "1 Main St, Austin TX 73301", "US")])
    pool = _frame([("S2-1", "Boulangerie Zephirine", "3 rue X Paris", "France"),
                   ("S3-1", "ACME Corporation", "1 Main Street Austin", "US"),
                   ("S3-2", "Other Thing LLC", "9 Elm St", "US")])
    c = generate_candidates(build_index(s1, pool, cfg), cfg)
    got = set(zip(c.s1_id, c.cand_id))
    assert ("S1-1", "S2-1") in got and ("S1-2", "S3-1") in got
    assert c["blk_rare_score"].notna().any()


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
            print("ok", k)
