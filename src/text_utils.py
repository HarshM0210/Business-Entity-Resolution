"""Country-agnostic normalisation. No hand-written dictionaries: legal forms are discovered per country
from unlabeled names, abbreviations are matched by a generic prefix/subsequence rule."""
import math
import re
from collections import Counter
from itertools import combinations

from anyascii import anyascii
from rapidfuzz.distance import JaroWinkler

_PRE = str.maketrans({"°": "o ", "º": "o ","’": "'", "`": "'", " ": " "})
_DOTTED = re.compile(r"\b(?:[a-z]\.){2,}")
_NONALNUM = re.compile(r"[^a-z0-9]+")
_DOMAIN = re.compile(r"^(?:www\.)?([a-z0-9][a-z0-9\-]*)\.[a-z]{2,4}$")
_JOIN_ALNUM = re.compile(r"(?<=[a-z0-9])[-/](?=[a-z0-9])")
_HOUSE_SEG = re.compile(r"[^a-z0-9#\-/. ]")
_HOUSE = re.compile(r"^(?:#|[a-z]{1,3}\.?)?\s*(\d{1,6})(?:\s*-\s*(\d{1,6})\b|([a-z])\b|\s*/\s*([a-z0-9]{1,3})\b)?")
_DIGITS = re.compile(r"\d+")
_LETTER = re.compile(r"[a-z]")


def is_nonlatin(s):
    return any(ord(ch) > 0x24F for ch in s) if s else False


def ascii_lower(s):
    if not s:
        return ""
    return anyascii(s.translate(_PRE)).lower()


def norm(s_ascii_lower):
    s = _DOTTED.sub(lambda m: m.group(0).replace(".", ""), s_ascii_lower)
    return _NONALNUM.sub(" ", s).strip()


# ----------------------------------------------------------------------------------------------- names

def name_tokens(raw):
    a = ascii_lower(raw).strip()
    if a and " " not in a:
        m = _DOMAIN.match(a)
        if m:
            a = m.group(1)
    return norm(a).split()


def discover_legal_forms(token_lists, min_frac, boundary_ratio, max_cooc, bigram_tight):
    """Legal forms = spans that sit at a name boundary, are frequent, and are mutually exclusive with each
    other (SARL/SAS/EURL never co-occur, while descriptors like 'groupe' co-occur with a legal form)."""
    n = len(token_lists)
    bcount, total_uni = Counter(), Counter()
    for t in token_lists:
        total_uni.update(set(t))
        if len(t) < 2:
            continue
        spans = {(t[-1],), (t[0],)}
        if len(t) > 2:
            spans |= {tuple(t[-2:]), tuple(t[:2])}
        bcount.update(spans)
    cands = [s for s, c in bcount.items() if c >= min_frac * n]
    cset = set(cands)
    total, occs, nbr = Counter(), [], {}
    for t in token_lists:
        if len(t) >= 2:
            for k in (1, 2):
                if len(t) > k:
                    for s, nb in ((tuple(t[-k:]), t[-k - 1]), (tuple(t[:k]), t[k])):
                        if s in cset:
                            nbr.setdefault(s, Counter())[nb] += 1
        occ = []
        for k in (1, 2):
            for i in range(len(t) - k + 1):
                s = tuple(t[i:i + k])
                if s in cset:
                    occ.append((s, i, i + k))
        if occ:
            total.update({o[0] for o in occ})
            occs.append(occ)
    cands = [s for s in cands if bcount[s] >= boundary_ratio * total[s]]
    # bigram must be "tight": its first token rarely appears outside it (private->private limited)
    cands = [s for s in cands if len(s) == 1 or total[s] >= bigram_tight * total_uni[s[0]]]
    # a legal form attaches to many different words; 'notre' is nearly always next to 'sainte'
    in_bigram = {x for s in cands if len(s) == 2 for x in s}

    def dominated(s):
        c = nbr.get(s)
        return bool(c) and s[0] not in in_bigram and c.most_common(1)[0][1] > 0.5 * sum(c.values())

    cands = [s for s in cands if not dominated(s)]
    keep = set(cands)
    co = Counter()
    for occ in occs:
        occ = [o for o in occ if o[0] in keep]
        for (a, ai, aj), (b, bi, bj) in combinations(occ, 2):
            if a != b and (aj <= bi or bj <= ai):
                co[frozenset((a, b))] += 1
    accepted = []
    for s in sorted(cands, key=lambda x: -bcount[x]):
        if all(co[frozenset((s, a))] <= max_cooc * min(total[s], total[a]) for a in accepted):
            accepted.append(s)
    return sorted(accepted, key=lambda x: (-len(x), x))


def strip_legal(tokens, forms, long_single):
    t, found = list(tokens), []
    changed = True
    while changed and len(t) > 1:
        changed = False
        for f in forms:
            k = len(f)
            if len(t) > k and tuple(t[-k:]) == f:
                found.append(" ".join(f)); del t[-k:]; changed = True; break
            if len(t) > k and tuple(t[:k]) == f:
                found.append(" ".join(f)); del t[:k]; changed = True; break
    if len(t) > 1:
        t2 = [x for x in t if x not in long_single]
        if t2 and len(t2) < len(t):
            found += [x for x in t if x in long_single]
            t = t2
    return t, " ".join(sorted(found))


def process_name(raw, forms, long_single):
    toks = name_tokens(raw)
    core, legal = strip_legal(toks, forms, long_single)
    return {"name_full": " ".join(toks), "name_core": " ".join(core), "legal": legal,
            "name_ns": "".join(core), "nonlatin": is_nonlatin(raw)}


# ----------------------------------------------------------------------------------------------- addresses

def parse_address(raw):
    raw = raw or ""
    segs = [s for s in re.split(r"[,;]", raw) if s.strip()]
    house, street_i, street_rest = "", -1, None
    for i, seg in enumerate(segs):
        s = _HOUSE_SEG.sub(" ", ascii_lower(seg)).strip()
        m = _HOUSE.match(s)
        if m and _LETTER.search(s[m.end():]):
            base, hi, suf1, suf2 = m.group(1), m.group(2), m.group(3), m.group(4)
            house = f"{int(base)}|{suf1 or suf2 or ''}|{int(hi) if hi else ''}"
            street_i, street_rest = i, s[m.end():]
            break
    if street_i < 0 and segs:
        street_i, street_rest = 0, ascii_lower(segs[0])
    street = norm(street_rest or "")
    loc = [norm(ascii_lower(s)) for j, s in enumerate(segs) if j != street_i]
    loc = [x for x in loc if x]
    al = ascii_lower(raw)
    full = norm(al)
    nums = _DIGITS.findall(full)
    hb = house.split("|")[0] if house else None
    postal = sorted({x for x in nums if len(x) in (5, 6) and x != hb})
    alnum = sorted({t for t in norm(_JOIN_ALNUM.sub("", al)).split() if _LETTER.search(t) and any(c.isdigit() for c in t)})
    return {"addr_norm": full, "street": street, "street_ns": street.replace(" ", ""), "loc": "|".join(loc),
            "house": house, "nums": " ".join(sorted(set(nums))), "postal": " ".join(postal),
            "alnum": " ".join(alnum), "addr_empty": not full}


def house_compare(ha, hb):
    """-> (state, log abs diff). 0 both missing, 1 one missing, 2 equal, 3 base equal/suffix differs,
    4 range contains, 5 conflict."""
    if not ha and not hb:
        return 0, math.nan
    if not ha or not hb:
        return 1, math.nan
    a, b = ha.split("|"), hb.split("|")
    ab, bb = int(a[0]), int(b[0])
    if ab == bb:
        return (2 if a[1] == b[1] else 3), 0.0
    if (a[2] and ab <= bb <= int(a[2])) or (b[2] and bb <= ab <= int(b[2])):
        return 4, 0.0
    return 5, math.log1p(abs(ab - bb))


def set_state(a, b):
    """0 both missing, 1 one missing, 2 share, 3 both present disjoint same lengths, 4 disjoint + length differs."""
    if not a and not b:
        return 0
    if not a or not b:
        return 1
    if a & b:
        return 2
    return 4 if {len(x) for x in a}.isdisjoint({len(x) for x in b}) else 3


# ----------------------------------------------------------------------------------------------- token matching

def is_subseq(short, long):
    it = iter(long)
    return all(c in it for c in short)


def abbrev_compat(a, b):
    if a == b:
        return True
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    return len(s) <= 4 and len(l) > len(s) and s[0] == l[0] and not s.isdigit() and is_subseq(s, l)


def tok_sim(a, b):
    if a == b:
        return 1.0
    if a.isdigit() or b.isdigit():
        return 0.0
    if abbrev_compat(a, b):
        return 0.9
    if len(a) >= 4 and len(b) >= 4:
        s = JaroWinkler.similarity(a, b)
        if s >= 0.9:
            return 0.9 * s
    return 0.0


def soft_cov(ta, tb):
    """Fraction of ta covered by tb (exact / abbreviation / typo tolerant)."""
    if not ta:
        return math.nan
    if not tb:
        return 0.0
    sb = set(tb)
    tot = 0.0
    for x in ta:
        if x in sb:
            tot += 1.0
            continue
        best = 0.0
        for y in tb:
            s = tok_sim(x, y)
            if s > best:
                best = s
        tot += best
    return tot / len(ta)


def contains_seq(a, b):
    k = len(a)
    if k == 0 or k > len(b):
        return False
    return any(b[i:i + k] == a for i in range(len(b) - k + 1))


def acronym_match(ta, tb):
    if len(ta) >= 2:
        ini = "".join(t[0] for t in ta)
        if ini in tb or ini == "".join(tb):
            return True
    return False


def legal_state(la, lb):
    if not la and not lb:
        return 0
    if not la or not lb:
        return 1
    if la == lb:
        return 2
    ta, tb = la.split(), lb.split()
    if len(ta) == len(tb) and all(abbrev_compat(x, y) for x, y in zip(ta, tb)):
        return 3
    if abbrev_compat(la.replace(" ", ""), lb.replace(" ", "")):
        return 3
    return 4
