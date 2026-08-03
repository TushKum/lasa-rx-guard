"""Build the drug-name corpus used for synthesis, OCR labels, and LASA matching.

Sources
-------
1. **RxNorm** (via the public RxNav REST API, no key required)
   - ``TTY=IN``  ingredient / generic names  (~14.6k raw)
   - ``TTY=BN``  brand names                 (~5.1k raw)
2. **NLEM India 2022** curated generics (:mod:`lasa.data.nlem_india`)
3. **ISMP/FDA LASA pairs** (:mod:`lasa.data.lasa_pairs`) — guaranteed inclusion,
   because the Stage-2 evaluation is meaningless if a ground-truth name is
   missing from the database being searched.

Why filtering is needed
-----------------------
RxNorm ``IN`` is full of IUPAC chemical strings and allergenic extracts, e.g.
``((4-hydroxybutyl)azanediyl)bis(hexane-6,1-diyl)bis(2-hexyldecanoate)`` and
``cat hair extract``. Those are not things a prescriber writes on a chit. We
keep only clean, drug-shaped tokens (see :func:`is_drug_like`), then rank so the
final corpus skews toward names a pharmacy counter actually dispenses.

Ranking (highest priority first) — documented because it directly shapes the
reported top-3 retrieval accuracy:
    0. appears in the LASA ground-truth pair list   (must be present)
    1. appears in NLEM India
    2. carries a recognised pharmacological stem (-olol, -pril, -statin, ...)
    3. everything else, shortest first
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from lasa.data.lasa_pairs import names_in_pairs
from lasa.data.nlem_india import nlem_names

RXNAV = "https://rxnav.nlm.nih.gov/REST/allconcepts.json"

# WHO INN pharmacological stems — a name carrying one of these is almost
# certainly a real therapeutic agent rather than a chemical reagent.
INN_STEMS: tuple[str, ...] = (
    "olol", "pril", "sartan", "statin", "dipine", "azepam", "azolam",
    "cillin", "mycin", "micin", "cycline", "floxacin", "conazole", "prazole",
    "tidine", "triptan", "gliptin", "glitazone", "flozin", "vastatin",
    "parin", "profen", "coxib", "caine", "curium", "onium", "terol",
    "sone", "solone", "asone", "onide", "vir", "navir", "ciclovir", "mab",
    "nib", "tinib", "ximab", "zumab", "setron", "prost", "tropium",
    "adol", "codone", "morphone", "fentanil", "barbital", "thiazide",
    "semide", "ridone", "apine", "pramine", "triptyline", "oxetine",
    "peridol", "azine", "dronate", "tecan", "rubicin", "platin", "taxel",
    "mustine", "arabine", "trexate", "stim", "poetin",
    # therapeutic enzymes — deliberately specific, NOT a bare "ase", which
    # would sweep in every metabolic enzyme in RxNorm (acetylcholinesterase...)
    "teplase", "plase", "kinase", "ginase", "cerase", "sidase", "uricase",
    "dornase",
)

_CLEAN = re.compile(r"^[a-z][a-z\-]{2,24}$")

# Substrings that mark a RxNorm IN entry as a biological extract, food, device
# or reagent rather than a prescribable small-molecule / biologic drug.
_REJECT_SUBSTR: tuple[str, ...] = (
    "extract", "pollen", "allergen", "venom", "dander", "epithelium",
    "vaccine", "antigen", "toxoid", "whole", "juice", "seed", "fruit",
    "flower", "leaf", "root", "bark", "hair", "feather", "serum", "diluent",
    "placebo", "excipient", "reagent", "kit", "strip", "tablet", "capsule",
    "solution", "injection", "cream", "ointment", "powder", "syrup",
    "acetate", "citrate", "tartrate", "besylate", "maleate", "mesylate",
    "fumarate", "succinate", "phosphate", "hydrochloride", "sulfate",
    "sodium", "potassium", "calcium", "magnesium", "zinc", "aluminum",
    "anhydrous", "monohydrate", "dihydrate", "labeled", "isotope",
    # inorganic salts, minerals, and metabolic enzymes that survive the
    # morphological filter but are not prescribable drug *names*
    "ammonium", "silicate", "carbonate", "chloride", "bromide", "iodide",
    "nitrate", "nitrite", "oxide", "hydroxide", "peroxide", "sulfide",
    "esterase", "phosphatase", "oxidase", "reductase", "synthase",
    "transferase", "dismutase", "hydrolase", "clase", "lyase", "polymerase",
    "globulin", "albumin", "collagen", "gelatin", "dextran", "cellulose",
)

# Below this length a RxNorm entry is almost always an element, ion or reagent
# ("lead", "iron", "urea") rather than a dispensable drug name. Curated sources
# (NLEM, LASA ground truth) bypass this check.
_MIN_RXNORM_LEN = 5


@dataclass(frozen=True)
class DrugName:
    name: str          # normalised, lowercase
    display: str       # original casing from source (nice for the UI)
    source: str        # rxnorm-in | rxnorm-bn | nlem-india | lasa-groundtruth
    kind: str          # generic | brand
    priority: int      # 0 = highest


def is_drug_like(name: str) -> bool:
    """Cheap morphological filter that removes IUPAC/extract noise."""
    n = name.strip().lower()
    if not _CLEAN.match(n):
        return False
    if len(n) < _MIN_RXNORM_LEN:
        return False
    if n.count("-") > 1:
        return False
    if any(s in n for s in _REJECT_SUBSTR):
        return False
    # Reject strings that read as chemistry rather than a drug name.
    if re.search(r"(meth|eth|prop|but|pent|hex|hept|oct)yl$", n):
        return False
    return True


def has_inn_stem(name: str) -> bool:
    return any(name.endswith(s) for s in INN_STEMS)


def fetch_rxnorm(tty: str, cache_dir: Path, timeout: int = 120) -> list[str]:
    """Fetch all RxNorm concepts of a given term type, with an on-disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"rxnorm_{tty.lower()}.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        for attempt in range(3):
            try:
                r = requests.get(RXNAV, params={"tty": tty}, timeout=timeout)
                r.raise_for_status()
                payload = r.json()
                cache.write_text(json.dumps(payload))
                break
            except requests.RequestException as exc:  # transient RxNav hiccup
                if attempt == 2:
                    raise RuntimeError(
                        f"RxNav fetch failed for tty={tty}. Re-run with network "
                        f"access, or drop --use-rxnorm to build an offline "
                        f"corpus from NLEM + LASA pairs only."
                    ) from exc
                time.sleep(2 * (attempt + 1))
    group = payload.get("minConceptGroup", {}).get("minConcept", []) or []
    return [c["name"] for c in group]


def build_corpus(
    target_size: int = 2000,
    use_rxnorm: bool = True,
    cache_dir: Path | None = None,
) -> list[DrugName]:
    from lasa.common.paths import RAW

    cache_dir = cache_dir or RAW
    gt = names_in_pairs()
    nlem = set(nlem_names())

    picked: dict[str, DrugName] = {}

    def add(display: str, source: str, kind: str, priority: int) -> None:
        key = display.strip().lower()
        if key in picked and picked[key].priority <= priority:
            return
        picked[key] = DrugName(key, display.strip(), source, kind, priority)

    # Priority 0 — ground truth must be searchable.
    for n in sorted(gt):
        add(n, "lasa-groundtruth", "brand" if n not in nlem else "generic", 0)

    # Priority 1 — Indian essential medicines.
    for n in sorted(nlem):
        add(n, "nlem-india", "generic", 1)

    if use_rxnorm:
        raw_in = fetch_rxnorm("IN", cache_dir)
        raw_bn = fetch_rxnorm("BN", cache_dir)

        clean_in = [n for n in raw_in if is_drug_like(n)]
        clean_bn = [n for n in raw_bn if is_drug_like(n)]

        # Priority 2 — recognised INN stem; 3 — remaining clean names.
        for n in sorted(clean_in, key=lambda s: (len(s), s)):
            add(n, "rxnorm-in", "generic", 2 if has_inn_stem(n.lower()) else 3)
        for n in sorted(clean_bn, key=lambda s: (len(s), s)):
            add(n, "rxnorm-bn", "brand", 3)

    ordered = sorted(picked.values(), key=lambda d: (d.priority, len(d.name), d.name))
    return ordered[:target_size]


def write_csv(rows: list[DrugName], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "display", "source", "kind", "priority"])
        for d in rows:
            w.writerow([d.name, d.display, d.source, d.kind, d.priority])


def load_csv(path: Path) -> list[DrugName]:
    with path.open(encoding="utf-8") as fh:
        return [
            DrugName(r["name"], r["display"], r["source"], r["kind"], int(r["priority"]))
            for r in csv.DictReader(fh)
        ]


def main() -> None:
    from lasa.common.paths import DRUG_NAMES_CSV, LASA_PAIRS_CSV, ensure_dirs
    from lasa.data import lasa_pairs

    ap = argparse.ArgumentParser(description="Build the drug-name corpus.")
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--no-rxnorm", action="store_true",
                    help="offline mode: NLEM + LASA ground truth only")
    args = ap.parse_args()

    ensure_dirs()
    n_pairs = lasa_pairs.write_csv(LASA_PAIRS_CSV)

    rows = build_corpus(target_size=args.size, use_rxnorm=not args.no_rxnorm)
    write_csv(rows, DRUG_NAMES_CSV)

    by_source: dict[str, int] = {}
    for d in rows:
        by_source[d.source] = by_source.get(d.source, 0) + 1

    covered = sum(1 for d in rows if d.name in lasa_pairs.names_in_pairs())
    total_gt = len(lasa_pairs.names_in_pairs())

    print(f"LASA pairs      : {n_pairs} -> {LASA_PAIRS_CSV}")
    print(f"Drug corpus     : {len(rows)} -> {DRUG_NAMES_CSV}")
    for s, c in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {s:<18} {c}")
    print(f"Ground-truth names present in corpus: {covered}/{total_gt}")
    lengths = [len(d.name) for d in rows]
    print(f"Name length: min={min(lengths)} max={max(lengths)} "
          f"mean={sum(lengths)/len(lengths):.1f}")


if __name__ == "__main__":
    main()
