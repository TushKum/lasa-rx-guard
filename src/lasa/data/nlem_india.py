"""Indian National List of Essential Medicines — curated generic names.

Source: NLEM 2022 (Ministry of Health & Family Welfare, Govt. of India), which
lists 384 medicines across 27 therapeutic categories, plus a handful of
high-volume Indian retail generics that are not on NLEM but dominate real
prescriptions (and therefore matter for a pharmacy-counter demo).

This is a hand-curated transcription of *generic ingredient names only* — no
strengths, no dose forms, no brand names. Names are stored lowercase and
single-token wherever the INN is single-token.

Why this exists: RxNorm is US-centric. Without this, the corpus over-represents
US brands and under-represents what an Indian pharmacy counter actually sees,
which would make the demo's top-3 retrieval look better than it is in the field.
"""

from __future__ import annotations

NLEM_INDIA: tuple[str, ...] = (
    # --- analgesics, antipyretics, NSAIDs, antimigraine ---
    "paracetamol", "ibuprofen", "diclofenac", "aspirin", "naproxen",
    "ketorolac", "tramadol", "morphine", "fentanyl", "buprenorphine",
    "sumatriptan", "indomethacin", "aceclofenac", "nimesulide", "etoricoxib",
    # --- antiallergics / anaphylaxis ---
    "cetirizine", "chlorpheniramine", "loratadine", "levocetirizine",
    "fexofenadine", "hydrocortisone", "prednisolone", "dexamethasone",
    "adrenaline", "montelukast", "pheniramine",
    # --- antidotes ---
    "atropine", "naloxone", "flumazenil", "acetylcysteine", "pralidoxime",
    "deferoxamine", "penicillamine", "calcium gluconate", "activated charcoal",
    # --- anticonvulsants ---
    "phenytoin", "carbamazepine", "valproate", "phenobarbitone", "clobazam",
    "levetiracetam", "lamotrigine", "clonazepam", "diazepam", "lorazepam",
    "midazolam", "topiramate", "oxcarbazepine", "gabapentin", "pregabalin",
    # --- anti-infectives: anthelminthics ---
    "albendazole", "mebendazole", "ivermectin", "praziquantel",
    "diethylcarbamazine", "niclosamide",
    # --- antibacterials ---
    "amoxicillin", "ampicillin", "benzylpenicillin", "cloxacillin",
    "piperacillin", "cefazolin", "cefixime", "cefotaxime", "ceftriaxone",
    "cefuroxime", "ceftazidime", "cefepime", "meropenem", "imipenem",
    "azithromycin", "erythromycin", "clarithromycin", "clindamycin",
    "ciprofloxacin", "levofloxacin", "ofloxacin", "moxifloxacin",
    "doxycycline", "tetracycline", "gentamicin", "amikacin", "streptomycin",
    "vancomycin", "linezolid", "metronidazole", "nitrofurantoin",
    "trimethoprim", "sulfamethoxazole", "chloramphenicol", "colistin",
    "rifampicin", "isoniazid", "pyrazinamide", "ethambutol", "dapsone",
    "clofazimine", "bedaquiline", "delamanid", "ethionamide", "cycloserine",
    "capreomycin", "kanamycin", "faropenem", "cefoperazone", "sulbactam",
    "tazobactam", "teicoplanin", "aztreonam", "fosfomycin", "norfloxacin",
    # --- antifungals ---
    "fluconazole", "itraconazole", "voriconazole", "amphotericin",
    "griseofulvin", "nystatin", "clotrimazole", "ketoconazole", "terbinafine",
    "caspofungin", "flucytosine", "miconazole",
    # --- antivirals ---
    "acyclovir", "valacyclovir", "oseltamivir", "ganciclovir", "ribavirin",
    "zidovudine", "lamivudine", "tenofovir", "efavirenz", "nevirapine",
    "abacavir", "dolutegravir", "ritonavir", "lopinavir", "atazanavir",
    "raltegravir", "emtricitabine", "sofosbuvir", "daclatasvir", "entecavir",
    # --- antimalarials / antiprotozoals ---
    "chloroquine", "artesunate", "artemether", "lumefantrine", "primaquine",
    "quinine", "mefloquine", "sulfadoxine", "pyrimethamine", "proguanil",
    "miltefosine", "paromomycin", "tinidazole", "ornidazole", "secnidazole",
    # --- antineoplastics / immunosuppressants ---
    "methotrexate", "cyclophosphamide", "doxorubicin", "daunorubicin",
    "vincristine", "vinblastine", "cisplatin", "carboplatin", "oxaliplatin",
    "paclitaxel", "docetaxel", "etoposide", "fluorouracil", "cytarabine",
    "gemcitabine", "imatinib", "rituximab", "trastuzumab", "bleomycin",
    "mercaptopurine", "hydroxyurea", "tamoxifen", "letrozole", "anastrozole",
    "azathioprine", "cyclosporine", "tacrolimus", "mycophenolate",
    "leucovorin", "asparaginase", "ifosfamide", "irinotecan", "capecitabine",
    "bicalutamide", "flutamide", "thalidomide", "lenalidomide", "bortezomib",
    # --- antiparkinson / psychotherapeutic ---
    "levodopa", "carbidopa", "trihexyphenidyl", "amantadine", "pramipexole",
    "ropinirole", "haloperidol", "chlorpromazine", "olanzapine", "risperidone",
    "quetiapine", "clozapine", "aripiprazole", "amitriptyline", "imipramine",
    "fluoxetine", "sertraline", "escitalopram", "paroxetine", "venlafaxine",
    "duloxetine", "mirtazapine", "bupropion", "lithium", "buspirone",
    "alprazolam", "trazodone", "donepezil", "memantine", "sodium valproate",
    # --- cardiovascular ---
    "amlodipine", "nifedipine", "diltiazem", "verapamil", "atenolol",
    "metoprolol", "propranolol", "bisoprolol", "carvedilol", "labetalol",
    "enalapril", "ramipril", "lisinopril", "captopril", "losartan",
    "telmisartan", "valsartan", "olmesartan", "furosemide", "torsemide",
    "spironolactone", "hydrochlorothiazide", "chlorthalidone", "indapamide",
    "digoxin", "amiodarone", "adenosine", "dopamine", "dobutamine",
    "noradrenaline", "nitroglycerin", "isosorbide", "atorvastatin",
    "rosuvastatin", "simvastatin", "clopidogrel", "warfarin", "heparin",
    "enoxaparin", "streptokinase", "tenecteplase", "alteplase", "prazosin",
    "hydralazine", "methyldopa", "clonidine", "ivabradine", "ranolazine",
    "dabigatran", "rivaroxaban", "apixaban", "fenofibrate", "ezetimibe",
    # --- dermatological ---
    "betamethasone", "clobetasol", "mupirocin", "permethrin", "benzoyl",
    "salicylic acid", "tretinoin", "isotretinoin", "calamine", "podophyllin",
    # --- diagnostic agents / contrast ---
    "iohexol", "barium sulfate", "fluorescein",
    # --- gastrointestinal ---
    "omeprazole", "pantoprazole", "esomeprazole", "rabeprazole", "ranitidine",
    "famotidine", "ondansetron", "domperidone", "metoclopramide",
    "dicyclomine", "loperamide", "lactulose", "bisacodyl", "sucralfate",
    "mesalazine", "ursodeoxycholic acid", "rifaximin", "itopride",
    # --- hormones / endocrine ---
    "insulin", "metformin", "glimepiride", "glipizide", "glibenclamide",
    "gliclazide", "pioglitazone", "sitagliptin", "vildagliptin",
    "dapagliflozin", "empagliflozin", "levothyroxine", "carbimazole",
    "methimazole", "propylthiouracil", "testosterone", "estradiol",
    "progesterone", "medroxyprogesterone", "norethisterone", "oxytocin",
    "misoprostol", "mifepristone", "clomiphene", "bromocriptine",
    "cabergoline", "octreotide", "desmopressin", "fludrocortisone",
    # --- immunologicals / vaccines ---
    "tetanus", "immunoglobulin", "antivenom",
    # --- muscle relaxants / anaesthesia ---
    "ketamine", "propofol", "thiopentone", "halothane", "isoflurane",
    "sevoflurane", "lignocaine", "bupivacaine", "succinylcholine",
    "vecuronium", "atracurium", "rocuronium", "neostigmine", "glycopyrrolate",
    "baclofen", "tizanidine", "chlorzoxazone", "thiocolchicoside",
    # --- ophthalmological ---
    "timolol", "latanoprost", "pilocarpine", "acetazolamide", "brimonidine",
    "dorzolamide", "tropicamide", "moxifloxacin eye", "carboxymethylcellulose",
    # --- respiratory ---
    "salbutamol", "terbutaline", "ipratropium", "tiotropium", "budesonide",
    "fluticasone", "formoterol", "salmeterol", "theophylline", "doxofylline",
    "ambroxol", "guaifenesin", "dextromethorphan", "levosalbutamol",
    # --- vitamins / minerals / fluids ---
    "folic acid", "ferrous sulfate", "cyanocobalamin", "thiamine",
    "pyridoxine", "riboflavin", "niacinamide", "ascorbic acid",
    "cholecalciferol", "calcitriol", "retinol", "phytomenadione", "zinc",
    "calcium carbonate", "magnesium sulfate", "potassium chloride",
    "sodium bicarbonate", "oral rehydration salts",
    # --- urology / others ---
    "tamsulosin", "finasteride", "dutasteride", "sildenafil", "tadalafil",
    "oxybutynin", "solifenacin", "allopurinol", "febuxostat", "colchicine",
    "alendronate", "zoledronic acid", "teriparatide", "raloxifene",
    "erythropoietin", "filgrastim", "tranexamic acid", "etamsylate",
    "phenylephrine", "pseudoephedrine", "hydroxychloroquine", "sulfasalazine",
    "leflunomide", "adalimumab", "infliximab", "etanercept", "tofacitinib",
)


def nlem_names() -> list[str]:
    """Single-token generic names only (drops 'calcium gluconate' etc.).

    Multi-word entries are kept in the tuple above for completeness/citation but
    the OCR task is single-word recognition, so the corpus builder takes the
    first token when the entry is a compound with a well-known head word.
    """
    out: list[str] = []
    for n in NLEM_INDIA:
        n = n.strip().lower()
        if " " in n:
            continue
        out.append(n)
    return sorted(set(out))
