"""String similarities in [0,1]. Uses rapidfuzz (MIT) when installed, pure-Python fallback otherwise.

The fallback's values are close but not identical to rapidfuzz, so never train with one backend
and predict with the other. The real run must have rapidfuzz installed (it is in requirements.txt)."""
from __future__ import annotations

try:
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz.distance import JaroWinkler as _JW, Levenshtein as _LEV
    HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - exercised only without rapidfuzz
    HAVE_RAPIDFUZZ = False

from difflib import SequenceMatcher


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    win = max(max(la, lb) // 2 - 1, 0)
    ma, mb = [False] * la, [False] * lb
    m = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - win), min(lb, i + win + 1)):
            if not mb[j] and b[j] == ca:
                ma[i] = mb[j] = True
                m += 1
                break
    if not m:
        return 0.0
    t, k = 0, 0
    for i in range(la):
        if ma[i]:
            while not mb[k]:
                k += 1
            t += a[i] != b[k]
            k += 1
    return (m / la + m / lb + (m - t / 2) / m) / 3


def jaro_winkler(a: str, b: str) -> float:
    if HAVE_RAPIDFUZZ:
        return _JW.similarity(a, b)
    j = _jaro(a, b)
    p = 0
    for x, y in zip(a[:4], b[:4]):
        if x != y:
            break
        p += 1
    return j + p * 0.1 * (1 - j)


def lev_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if HAVE_RAPIDFUZZ:
        return _LEV.normalized_similarity(a, b)
    return 1 - _lev(a, b) / max(len(a), len(b))


def token_sort(a: str, b: str) -> float:
    if HAVE_RAPIDFUZZ:
        return _fuzz.token_sort_ratio(a, b) / 100
    return SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split()))).ratio()


def token_set(a: str, b: str) -> float:
    if HAVE_RAPIDFUZZ:
        return _fuzz.token_set_ratio(a, b) / 100
    sa, sb = set(a.split()), set(b.split())
    inter = " ".join(sorted(sa & sb))
    da, db = " ".join(sorted(sa - sb)), " ".join(sorted(sb - sa))
    c1, c2 = (inter + " " + da).strip(), (inter + " " + db).strip()
    if inter and (not da or not db):
        return 1.0
    r = SequenceMatcher(None, c1, c2).ratio()
    if inter:
        r = max(r, SequenceMatcher(None, inter, c1).ratio(), SequenceMatcher(None, inter, c2).ratio())
    return r


def partial_ratio(a: str, b: str) -> float:
    if HAVE_RAPIDFUZZ:
        return _fuzz.partial_ratio(a, b) / 100
    if not a or not b:
        return 0.0
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    best = 0.0
    for i in range(0, len(l) - len(s) + 1):
        best = max(best, SequenceMatcher(None, s, l[i:i + len(s)]).ratio())
        if best == 1.0:
            break
    return best


def lcs_ratio(a: str, b: str) -> float:
    """Longest common substring / shorter length."""
    if not a or not b:
        return 0.0
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size / min(len(a), len(b))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def ngrams(s: str, n: int = 3) -> set:
    s = f" {s} "
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}
