"""Phase 2 — deterministic, country-agnostic normalization.

All maps below are small, in-repo text-normalization rules (standard abbreviations), not an
external lookup. Postcodes are detected by digit *pattern* only — no gazetteer, no geocoding.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------- abbreviation maps
NAME_ABBR = {
    "pvt": "private", "pvte": "private", "prvt": "private", "priv": "private",
    "ltd": "limited", "ltda": "limited", "lmtd": "limited", "ld": "limited",
    "corp": "corporation", "corpn": "corporation", "co": "company", "cos": "companies",
    "inc": "incorporated", "incorp": "incorporated",
    "intl": "international", "int'l": "international", "mfg": "manufacturing",
    "mfrs": "manufacturers", "bros": "brothers", "svc": "services", "svcs": "services",
    "assoc": "associates", "assocs": "associates", "tech": "technologies",
    "techs": "technologies", "sys": "systems", "mgmt": "management", "dev": "development",
    "grp": "group", "hldgs": "holdings", "ind": "industries", "inds": "industries",
    "natl": "national", "univ": "university", "ctr": "center", "centre": "center",
    "&": "and", "et": "and",
}
# Legal/entity-form tokens (after expansion). Stripped to form name_core.
LEGAL_TOKENS = {
    "private", "limited", "corporation", "company", "companies", "incorporated",
    "llc", "llp", "lp", "plc", "pllc", "pc", "opc", "the", "and",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "gmbh", "ag", "bv", "nv",
    "pty", "pte", "ltee",
}
ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "apt": "apartment", "apts": "apartments",
    "flr": "floor", "fl": "floor", "bldg": "building", "bld": "building",
    "opp": "opposite", "nr": "near", "ste": "suite", "hwy": "highway", "ln": "lane",
    "dr": "drive", "ct": "court", "pl": "place", "sq": "square", "mkt": "market",
    "ngr": "nagar", "clny": "colony", "sec": "sector", "sect": "sector", "ph": "phase",
    "hsg": "housing", "soc": "society", "chs": "society", "cplx": "complex",
    "pkwy": "parkway", "cir": "circle", "crt": "court", "ter": "terrace",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "mg": "mahatma gandhi", "fbg": "faubourg", "rte": "route", "chem": "chemin",
    "imp": "impasse", "pte": "porte", "no": "", "num": "", "nos": "",
}
ADDR_STOP = {"road", "street", "avenue", "lane", "floor", "building", "the", "of", "and",
             "suite", "unit", "shop", "plot", "house", "flat", "de", "du", "des", "la", "le"}
LANDMARK_LEADS = ("near", "nr", "opposite", "opp", "behind", "beside", "next to",
                  "adjacent to", "in front of", "close to", "besides", "above", "below")

_COMBINING = re.compile(r"[\u0300-\u036f]")
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_WS = re.compile(r"\s+")
_LANDMARK_RE = re.compile(r"^\s*(?:" + "|".join(re.escape(x) for x in LANDMARK_LEADS) + r")\b\.?",
                          re.IGNORECASE)


def base_clean(s: str) -> str:
    """NFKC -> casefold -> strip Latin accents (keeps Devanagari vowel signs) -> &->and."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = _COMBINING.sub("", unicodedata.normalize("NFKD", s))
    s = unicodedata.normalize("NFC", s)
    s = s.replace("&", " and ").replace("@", " at ").replace("'", "").replace("’", "")
    return s


def _tokens(s: str) -> list[str]:
    toks = _WS.sub(" ", _NON_WORD.sub(" ", s)).strip().split()
    # merge runs of single letters: "s b i" -> "sbi", "i b m" -> "ibm"
    out, run = [], []
    for t in toks:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return [t for t in out if t != "_"]


def _expand(tokens: list[str], table: dict[str, str]) -> list[str]:
    out = []
    for t in tokens:
        rep = table.get(t, t)
        if rep:
            out.extend(rep.split())
    return out


# ---------------------------------------------------------------- phonetic key
_PH_RULES = [("ph", "f"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("th", "t"),
             ("bh", "b"), ("dh", "d"), ("jh", "j"), ("ck", "k"), ("q", "k"), ("w", "v"),
             ("z", "s"), ("x", "ks"), ("ee", "i"), ("oo", "u"), ("ou", "u"), ("y", "i")]


def phonetic_token(t: str) -> str:
    """Consonant skeleton robust to common transliteration variation (sri/shri, -ee/-i, ph/f)."""
    if not t or not t.isascii() or t.isdigit():
        return t
    for a, b in _PH_RULES:
        t = t.replace(a, b)
    t = t.replace("c", "k")
    head, rest = t[0], t[1:]
    rest = re.sub(r"[aeiouh]", "", rest)
    s = head + rest
    return re.sub(r"(.)\1+", r"\1", s)


# ---------------------------------------------------------------- names
def normalize_name(raw: str) -> dict:
    toks = _expand(_tokens(base_clean(raw)), NAME_ABBR)
    full = " ".join(toks)
    core_toks = [t for t in toks if t not in LEGAL_TOKENS] or toks
    core = " ".join(core_toks)
    return {
        "name_full": full,
        "name_core": core,
        "name_sorted": " ".join(sorted(core_toks)),
        "name_initials": "".join(t[0] for t in core_toks if t),
        "name_phon": " ".join(phonetic_token(t) for t in core_toks),
        "name_nospace": core.replace(" ", ""),
    }


# ---------------------------------------------------------------- addresses
_POST6 = re.compile(r"(?<!\d)(\d{3})\s?(\d{3})(?!\d)")
_DIGITS = re.compile(r"\d")


def normalize_address(raw: str) -> dict:
    s = base_clean(raw)
    landmarks, kept = [], []
    for seg in re.split(r"[,;\n|]", s):
        seg = seg.strip()
        if not seg:
            continue
        if _LANDMARK_RE.match(seg):
            landmarks.extend(t for t in _tokens(_LANDMARK_RE.sub("", seg)) if not t.isdigit())
        else:
            kept.append(seg)
    s = " , ".join(kept)

    s = re.sub(r"(?<!\d)(\d{5})\s?-\s?\d{4}(?!\d)", r"\1", s)   # ZIP+4 -> ZIP
    # postcode by pattern: prefer a 6-digit (PIN, "110 001" spacing tolerated), else last 5-digit.
    postcode = ""
    m6 = [m for m in _POST6.finditer(s)]
    if m6:
        m = m6[-1]
        postcode = m.group(1) + m.group(2)
        s = s[:m.start()] + " " + s[m.end():]
    else:
        m5 = re.findall(r"(?<!\d)\d{5}(?!\d)", s)
        if m5:
            postcode = m5[-1]
            s = re.sub(rf"(?<!\d){postcode}(?!\d)", " ", s, count=1)

    toks = _expand(_tokens(s), ADDR_ABBR)
    numbers = sorted({re.sub(r"\D", "", t) for t in toks if _DIGITS.search(t)} - {""})
    words = [t for t in toks if not _DIGITS.search(t)]
    return {
        "addr_full": " ".join(toks),
        "addr_core": " ".join(words),
        "addr_key": " ".join(t for t in words if t not in ADDR_STOP),
        "addr_numbers": " ".join(numbers),
        "addr_postcode": postcode,
        "addr_landmarks": " ".join(sorted(set(landmarks))),
    }


def _normalize_chunk(names: list[str], addrs: list[str], countries: list[str]):
    names_df = pd.DataFrame([normalize_name(x) for x in names])
    addrs_df = pd.DataFrame([normalize_address(x) for x in addrs])
    country_norm = pd.Series([base_clean(c).strip() for c in countries], name="country_norm")
    return names_df, addrs_df, country_norm


def normalize_frame(df: pd.DataFrame, chunk_size: int = 50_000, n_jobs: int | None = None) -> pd.DataFrame:
    """Adds normalized columns; raw columns are kept for exact-string features.

    Processes in row-chunks (each chunk builds its own small DataFrame instead of one
    Python list-of-dicts spanning the whole input) and, above `chunk_size` rows,
    parallelizes chunks across processes: at multi-million-row scale a single Python
    process doing this row-by-row is both memory- and CPU-bound.
    """
    n = len(df)
    if n <= chunk_size:
        names_df, addrs_df, country_norm = _normalize_chunk(
            df["business_name"].tolist(), df["business_address"].tolist(), df["country"].tolist())
        names_df.index = addrs_df.index = country_norm.index = df.index
        return pd.concat([df, names_df, addrs_df, country_norm], axis=1)

    import os
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    chunks = [df.iloc[i:i + chunk_size] for i in range(0, n, chunk_size)]
    n_jobs = n_jobs or min(4, os.cpu_count() or 1, len(chunks))
    results = [None] * len(chunks)
    # spawn, not the platform default fork: fork would hand every worker a
    # copy-on-write snapshot of this process's *entire* memory (including whatever
    # multi-GB frame the caller already holds), which then balloons for real as each
    # worker's own refcounting touches those inherited pages. spawn starts each
    # worker as a fresh interpreter that only ever receives the pickled chunk args.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        futs = {ex.submit(_normalize_chunk, c["business_name"].tolist(),
                          c["business_address"].tolist(), c["country"].tolist()): i
               for i, c in enumerate(chunks)}
        for fut in futs:
            results[futs[fut]] = fut.result()

    parts = []
    for c, (names_df, addrs_df, country_norm) in zip(chunks, results):
        names_df.index = addrs_df.index = country_norm.index = c.index
        parts.append(pd.concat([c, names_df, addrs_df, country_norm], axis=1))
    return pd.concat(parts)
