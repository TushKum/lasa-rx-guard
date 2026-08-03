"""LASA (Look-Alike / Sound-Alike) confused drug-name pairs — evaluation ground truth.

PROVENANCE (read this before quoting numbers in a report)
---------------------------------------------------------
The pairs below are a **curated subset** of confusable drug-name pairs that are
documented in publicly published safety references:

  * ISMP "List of Confused Drug Names" (Institute for Safe Medication Practices)
  * FDA/ISMP "Tall Man Lettering" list (name-differentiation via mixed case)
  * FDA Drug Safety Communications on name confusion

These are *factual pairings*, transcribed by hand and cross-checked against the
tall-man capitalisation convention. This module is NOT a verbatim copy of the
ISMP table, and it is NOT complete — ISMP's current list runs to several hundred
entries and is redistributed under their own terms.

If you need the full official list for your report, download it yourself and run:

    python scripts/ingest_ismp.py --csv path/to/ismp_confused_names.csv

which will merge it into ``data/processed/lasa_pairs.csv``. Everything downstream
reads that CSV, so the evaluation automatically improves when you supply the
official file. Any results table must state which source was used.

The ``tall_man`` field preserves the published capitalisation (e.g.
``DOBUTamine``) because that IS the mitigation being recommended — it is useful
to surface it in the demo UI.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

# ISMP high-alert classes: an error with these has a markedly higher chance of
# causing significant patient harm. Used to escalate the risk flag.
HIGH_ALERT_TOKENS = {
    "insulin", "heparin", "warfarin", "morphine", "hydromorphone", "oxycodone",
    "hydrocodone", "fentanyl", "methadone", "meperidine", "vincristine",
    "vinblastine", "vinorelbine", "doxorubicin", "daunorubicin", "idarubicin",
    "cisplatin", "carboplatin", "methotrexate", "mitomycin", "mitoxantrone",
    "paclitaxel", "docetaxel", "dopamine", "dobutamine", "epinephrine",
    "ephedrine", "norepinephrine", "amiodarone", "digoxin", "potassium",
    "magnesium", "lithium", "clozapine", "colchicine",
}


@dataclass(frozen=True)
class LasaPair:
    a: str
    b: str
    source: str          # "ISMP" | "FDA-TallMan" | "FDA-DSC"
    a_tall_man: str = ""
    b_tall_man: str = ""

    def high_alert(self) -> bool:
        blob = f"{self.a} {self.b}".lower()
        return any(tok in blob for tok in HIGH_ALERT_TOKENS)


def _p(a: str, b: str, source: str = "ISMP", tm_a: str = "", tm_b: str = "") -> LasaPair:
    return LasaPair(a.lower().strip(), b.lower().strip(), source, tm_a, tm_b)


# --------------------------------------------------------------------------
# Generic-name confusions (tall man lettering recommended by FDA/ISMP)
# --------------------------------------------------------------------------
_TALL_MAN_PAIRS: list[LasaPair] = [
    _p("dopamine", "dobutamine", "FDA-TallMan", "DOPamine", "DOBUTamine"),
    _p("hydroxyzine", "hydralazine", "FDA-TallMan", "hydrOXYzine", "hydrALAZINE"),
    _p("hydralazine", "hydrochlorothiazide", "FDA-TallMan", "hydrALAZINE", "hydrochlorothiazide"),
    _p("glipizide", "glyburide", "FDA-TallMan", "glipiZIDE", "glyBURIDE"),
    _p("prednisone", "prednisolone", "FDA-TallMan", "predniSONE", "prednisoLONE"),
    _p("vinblastine", "vincristine", "FDA-TallMan", "vinBLAStine", "vinCRIStine"),
    _p("vincristine", "vinorelbine", "FDA-TallMan", "vinCRIStine", "vinORELBine"),
    _p("chlorpromazine", "chlorpropamide", "FDA-TallMan", "chlorproMAZINE", "chlorproPAMIDE"),
    _p("clomiphene", "clomipramine", "FDA-TallMan", "clomiPHENE", "clomiPRAMINE"),
    _p("cyclosporine", "cycloserine", "FDA-TallMan", "cycloSPORINE", "cycloSERINE"),
    _p("daunorubicin", "doxorubicin", "FDA-TallMan", "DAUNOrubicin", "DOXOrubicin"),
    _p("doxorubicin", "idarubicin", "FDA-TallMan", "DOXOrubicin", "IDArubicin"),
    _p("dimenhydrinate", "diphenhydramine", "FDA-TallMan", "dimenhyDRINATE", "diphenhydrAMINE"),
    _p("nicardipine", "nifedipine", "FDA-TallMan", "niCARdipine", "NIFEdipine"),
    _p("sulfadiazine", "sulfasalazine", "FDA-TallMan", "sulfADIAZINE", "sulfaSALAzine"),
    _p("tolazamide", "tolbutamide", "FDA-TallMan", "TOLAZamide", "TOLBUTamide"),
    _p("bupropion", "buspirone", "ISMP", "buPROPion", "busPIRone"),
    _p("olanzapine", "quetiapine", "ISMP", "OLANZapine", "QUEtiapine"),
    _p("risperidone", "ropinirole", "ISMP", "risperiDONE", "rOPINIRole"),
    _p("tramadol", "trazodone", "ISMP", "traMADol", "traZODone"),
    _p("valacyclovir", "valganciclovir", "ISMP", "valACYclovir", "valGANciclovir"),
    _p("sitagliptin", "sumatriptan", "ISMP", "SITagliptin", "SUMAtriptan"),
    _p("sorafenib", "sunitinib", "ISMP", "SORAfenib", "SUNItinib"),
    _p("metformin", "metronidazole", "ISMP", "metFORMIN", "metroNIDAZOLE"),
    _p("lamivudine", "lamotrigine", "ISMP", "lamiVUDine", "lamoTRIgine"),
    _p("amlodipine", "amiloride", "ISMP", "amLODIPine", "aMILoride"),
    _p("ephedrine", "epinephrine", "ISMP", "ePHEDrine", "EPINEPHrine"),
    _p("quinidine", "quinine", "ISMP", "quiNIDine", "quiNINE"),
    _p("tiagabine", "tizanidine", "ISMP", "tiaGABine", "tiZANidine"),
    _p("infliximab", "rituximab", "ISMP", "inFLIXimab", "riTUXimab"),
    _p("mitomycin", "mitoxantrone", "ISMP", "mitoMYcin", "mitoXANTRONE"),
    _p("isotretinoin", "tretinoin", "ISMP", "ISOtretinoin", "tretinoin"),
    _p("oxycodone", "hydrocodone", "ISMP", "oxyCODONE", "HYDROcodone"),
    _p("oxycodone", "oxymorphone", "ISMP", "oxyCODONE", "oxyMORphone"),
    _p("morphine", "hydromorphone", "ISMP", "morphine", "HYDROmorphone"),
    _p("methylprednisolone", "medroxyprogesterone", "ISMP", "methylPREDNISolone", "medroxyPROGESTERone"),
    _p("methylprednisolone", "methyltestosterone", "ISMP", "methylPREDNISolone", "methylTESTOSTERone"),
    _p("levetiracetam", "levocarnitine", "ISMP", "levETIRAcetam", "levOCARNitine"),
    _p("levetiracetam", "levofloxacin", "ISMP", "levETIRAcetam", "levOFLOXacin"),
    _p("alprazolam", "lorazepam", "ISMP", "ALPRAZolam", "LORazepam"),
    _p("clonazepam", "lorazepam", "ISMP", "clonazePAM", "LORazepam"),
    _p("clonazepam", "clonidine", "ISMP", "clonazePAM", "cloNIDine"),
    _p("chlorpromazine", "prochlorperazine", "ISMP", "chlorproMAZINE", "prochlorperazine"),
    _p("carbamazepine", "oxcarbazepine", "ISMP"),
    _p("amiodarone", "amrinone", "ISMP"),
    _p("carboplatin", "cisplatin", "ISMP"),
    _p("nelfinavir", "nevirapine", "ISMP"),
    _p("paclitaxel", "paroxetine", "ISMP"),
    _p("penicillin", "penicillamine", "ISMP"),
    _p("pioglitazone", "rosiglitazone", "ISMP"),
    _p("ranitidine", "rimantadine", "ISMP"),
    _p("azithromycin", "erythromycin", "ISMP"),
    _p("fluoxetine", "paroxetine", "ISMP"),
    _p("fluoxetine", "duloxetine", "ISMP"),
    _p("metoprolol", "metolazone", "ISMP"),
    _p("metolazone", "methotrexate", "ISMP"),
    _p("methotrexate", "metronidazole", "ISMP"),
    _p("cefazolin", "cefotaxime", "ISMP"),
    _p("ceftriaxone", "cefuroxime", "ISMP"),
    _p("atenolol", "albuterol", "ISMP"),
    _p("glimepiride", "glipizide", "ISMP"),
    _p("acetazolamide", "acetohexamide", "ISMP"),
    _p("chlorothiazide", "chlorpropamide", "ISMP"),
    _p("desipramine", "disopyramide", "ISMP"),
    _p("dactinomycin", "daptomycin", "ISMP"),
    _p("guanfacine", "guaifenesin", "ISMP"),
    _p("nystatin", "nitrostat", "ISMP"),
    _p("saquinavir", "sinequan", "ISMP"),
    _p("sumatriptan", "zolmitriptan", "ISMP"),
    _p("terbinafine", "terbutaline", "ISMP"),
    _p("tobramycin", "tobradex", "ISMP"),
    _p("vecuronium", "vancomycin", "ISMP"),
    _p("cisatracurium", "cisplatin", "ISMP"),
    _p("dexamethasone", "desmopressin", "ISMP"),
    _p("fluphenazine", "fluoxetine", "ISMP"),
    _p("leucovorin", "leukeran", "ISMP"),
    _p("propranolol", "propofol", "ISMP"),
    _p("sildenafil", "tadalafil", "ISMP"),
    _p("simvastatin", "sitagliptin", "ISMP"),
    _p("lithium", "lisinopril", "ISMP"),
    _p("cetirizine", "sertraline", "ISMP"),
    _p("cyclobenzaprine", "cyproheptadine", "ISMP"),
    _p("hydrocortisone", "hydrochlorothiazide", "ISMP"),
    _p("nifedipine", "nimodipine", "ISMP"),
    _p("clobazam", "clonazepam", "ISMP"),
    _p("cefepime", "cefixime", "ISMP"),
    _p("azathioprine", "azithromycin", "ISMP"),
]

# --------------------------------------------------------------------------
# Brand-name confusions
# --------------------------------------------------------------------------
_BRAND_PAIRS: list[LasaPair] = [
    _p("celebrex", "celexa", "ISMP"),
    _p("celebrex", "cerebyx", "ISMP"),
    _p("celexa", "cerebyx", "ISMP"),
    _p("zantac", "xanax", "ISMP"),
    _p("zantac", "zyrtec", "ISMP"),
    _p("zyrtec", "zyprexa", "ISMP"),
    _p("zyprexa", "zestril", "ISMP"),
    _p("zocor", "cozaar", "ISMP"),
    _p("zocor", "zyrtec", "ISMP"),
    _p("prilosec", "prozac", "ISMP"),
    _p("prilosec", "prinivil", "ISMP"),
    _p("prozac", "prilosec", "ISMP"),
    _p("paxil", "plavix", "ISMP"),
    _p("paxil", "taxol", "ISMP"),
    _p("taxol", "taxotere", "ISMP"),
    _p("lamictal", "lamisil", "ISMP"),
    _p("lamictal", "lomotil", "ISMP"),
    _p("lamisil", "lomotil", "ISMP"),
    _p("depakote", "depakene", "ISMP"),
    _p("percocet", "percodan", "ISMP"),
    _p("fioricet", "fiorinal", "ISMP"),
    _p("seroquel", "serzone", "ISMP"),
    _p("seroquel", "sinequan", "ISMP"),
    _p("norvasc", "navane", "ISMP"),
    _p("norvasc", "norvir", "ISMP"),
    _p("diflucan", "diprivan", "ISMP"),
    _p("tegretol", "toprol", "ISMP"),
    _p("tegretol", "trental", "ISMP"),
    _p("coumadin", "kemadrin", "ISMP"),
    _p("avandia", "prandin", "ISMP"),
    _p("avandia", "coumadin", "ISMP"),
    _p("humalog", "humulin", "ISMP"),
    _p("humalog", "novolog", "ISMP"),
    _p("novolin", "novolog", "ISMP"),
    _p("lantus", "lente", "ISMP"),
    _p("toradol", "tramadol", "ISMP"),
    _p("toradol", "torecan", "ISMP"),
    _p("retrovir", "ritonavir", "ISMP"),
    _p("lasix", "losec", "ISMP"),
    _p("adderall", "inderal", "ISMP"),
    _p("actos", "actonel", "ISMP"),
    _p("allegra", "viagra", "ISMP"),
    _p("zyban", "diovan", "ISMP"),
    _p("lunesta", "neulasta", "ISMP"),
    _p("neulasta", "neumega", "ISMP"),
    _p("femara", "femhrt", "ISMP"),
    _p("heparin", "hespan", "ISMP"),
    _p("levoxyl", "lanoxin", "ISMP"),
    _p("maalox", "marax", "ISMP"),
    _p("reminyl", "amaryl", "ISMP"),
    _p("sarafem", "serophene", "ISMP"),
    _p("tobradex", "tobrex", "ISMP"),
    _p("xeloda", "xenical", "ISMP"),
    _p("zestril", "zetia", "ISMP"),
    _p("klonopin", "clonidine", "ISMP"),
    _p("keppra", "kaletra", "ISMP"),
    _p("viracept", "viramune", "ISMP"),
    _p("wellbutrin", "wellcovorin", "ISMP"),
    _p("flomax", "volmax", "ISMP"),
    _p("plendil", "pletal", "ISMP"),
    _p("cardizem", "cardene", "ISMP"),
    _p("cardura", "coumadin", "ISMP"),
    _p("carafate", "cafergot", "ISMP"),
    _p("dilantin", "diflucan", "ISMP"),
    _p("edecrin", "eulexin", "ISMP"),
    _p("effexor", "effient", "ISMP"),
    _p("lipitor", "loniten", "ISMP"),
    _p("micronase", "microk", "ISMP"),
    _p("mirapex", "miralax", "ISMP"),
    _p("prevacid", "prinivil", "ISMP"),
    _p("provera", "premarin", "ISMP"),
    _p("risperdal", "requip", "ISMP"),
    _p("singulair", "sinequan", "ISMP"),
    _p("solu-medrol", "solu-cortef", "ISMP"),
    _p("valium", "valcyte", "ISMP"),
    _p("vesicare", "vesanoid", "ISMP"),
    _p("xanax", "zantac", "ISMP"),
    _p("zebeta", "diabeta", "ISMP"),
    _p("zyvox", "zovirax", "ISMP"),
]

ALL_PAIRS: list[LasaPair] = _TALL_MAN_PAIRS + _BRAND_PAIRS


def deduped_pairs() -> list[LasaPair]:
    """Unordered-unique pairs (a,b) == (b,a), self-pairs dropped."""
    seen: set[frozenset[str]] = set()
    out: list[LasaPair] = []
    for p in ALL_PAIRS:
        if p.a == p.b:
            continue
        key = frozenset((p.a, p.b))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def pair_lookup() -> dict[str, set[str]]:
    """name -> set of names it is officially confusable with (symmetric)."""
    idx: dict[str, set[str]] = {}
    for p in deduped_pairs():
        idx.setdefault(p.a, set()).add(p.b)
        idx.setdefault(p.b, set()).add(p.a)
    return idx


def names_in_pairs() -> set[str]:
    out: set[str] = set()
    for p in deduped_pairs():
        out.add(p.a)
        out.add(p.b)
    return out


def write_csv(path: Path) -> int:
    pairs = deduped_pairs()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["a", "b", "source", "a_tall_man", "b_tall_man", "high_alert"]
        )
        w.writeheader()
        for p in pairs:
            row = asdict(p)
            row["high_alert"] = int(p.high_alert())
            w.writerow(row)
    return len(pairs)


if __name__ == "__main__":
    from lasa.common.paths import LASA_PAIRS_CSV

    n = write_csv(LASA_PAIRS_CSV)
    ha = sum(p.high_alert() for p in deduped_pairs())
    print(f"wrote {n} unique LASA pairs -> {LASA_PAIRS_CSV}")
    print(f"  high-alert pairs: {ha}")
    print(f"  distinct drug names covered: {len(names_in_pairs())}")
