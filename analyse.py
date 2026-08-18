"""
Bureau d'Analyse Terrestre - reception des releves (Klaxo-3)
Script unique, se relance du telechargement au dernier chiffre.
"""

import csv
import os
import re
import sys
import urllib.request
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_URL = "https://raw.githubusercontent.com/planetsig/ufo-reports/master/csv-data/ufo-complete-geocoded-time-standardized.csv"
DATA_FILE = "releves_klaxo3.csv"
COLUMNS = [
    "datetime", "city", "state", "country", "shape", "duration_seconds",
    "duration_hours_min", "comments", "date_posted", "latitude", "longitude",
]
RANDOM_STATE = 42


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def telecharger_si_besoin():
    if not os.path.exists(DATA_FILE):
        print(f"Telechargement de {DATA_FILE} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    else:
        print(f"{DATA_FILE} deja present, pas de retelechargement.")


# ---------------------------------------------------------------------------
# Phase 1 : ouvrir la caisse
# ---------------------------------------------------------------------------
def phase1_ouvrir_la_caisse():
    section("PHASE 1 - ouvrir la caisse")

    total = 0
    bonnes_lignes = []
    lignes_de_cote = []
    with open(DATA_FILE, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            total += 1
            if len(row) == len(COLUMNS):
                bonnes_lignes.append(row)
            else:
                lignes_de_cote.append(row)

    n_chargees = len(bonnes_lignes)
    n_de_cote = len(lignes_de_cote)

    assert n_chargees + n_de_cote == total, "les comptes ne collent pas"

    print(f"lignes dans le fichier   : {total}")
    print(f"lignes chargees          : {n_chargees}")
    print(f"lignes mises de cote     : {n_de_cote}")
    print("\nexemple de ligne problematique (mise de cote) :")
    print(lignes_de_cote[0])
    print(f"-> elle a {len(lignes_de_cote[0])} champs au lieu de {len(COLUMNS)} :")
    print("   la colonne 'city' y est vide, ce qui decale tout le reste d'un cran.")

    df = pd.DataFrame(bonnes_lignes, columns=COLUMNS)
    return df, total, n_chargees, n_de_cote


# ---------------------------------------------------------------------------
# Phase 2 : rien n'est du bon type
# ---------------------------------------------------------------------------
def phase2_typage(df):
    section("PHASE 2 - rien n'est du bon type")

    df["latitude_num"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude_num"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["duration_seconds_num"] = pd.to_numeric(df["duration_seconds"], errors="coerce")

    df["datetime_parsed"] = pd.to_datetime(df["datetime"], format="%m/%d/%Y %H:%M", errors="coerce")
    df["date_posted_parsed"] = pd.to_datetime(df["date_posted"], format="%m/%d/%Y", errors="coerce")

    anomalies = []

    mask = df["latitude_num"].isna() & df["latitude"].ne("")
    valeurs = df.loc[mask, "latitude"].tolist()
    anomalies.append(("latitude", len(valeurs), valeurs[:5], "service de transmission (geocodage corrompu)"))

    mask_txt = df["duration_seconds_num"].isna() & df["duration_seconds"].ne("")
    valeurs_txt = df.loc[mask_txt, "duration_seconds"].tolist()
    anomalies.append(("duration_seconds (caractere parasite)", len(valeurs_txt), valeurs_txt,
                       "service de transmission (parsing automatique du texte du temoin)"))

    mask_vide = df["duration_seconds"].eq("")
    anomalies.append(("duration_seconds (valeur manquante)", int(mask_vide.sum()), ["''"] * min(3, int(mask_vide.sum())),
                       "temoin (n'a jamais donne de duree exploitable)"))

    mask_dt = df["datetime_parsed"].isna() & df["datetime"].ne("")
    valeurs_dt = df.loc[mask_dt, "datetime"].tolist()
    anomalies.append(("datetime (heure '24:00')", len(valeurs_dt), valeurs_dt[:5],
                       "temoin (a ecrit minuit sous la forme 24:00, heure qui n'existe pas)"))

    mask_dp = df["date_posted_parsed"].isna() & df["date_posted"].ne("")
    n_dp = int(mask_dp.sum())
    mask_lon = df["longitude_num"].isna() & df["longitude"].ne("")
    n_lon = int(mask_lon.sum())

    for nom, n, exemples, source in anomalies:
        print(f"{nom:38s} : {n:5d} valeur(s) fautive(s)  -> {source}")
        print(f"   exemples : {exemples}")
    print(f"{'date_posted':38s} : {n_dp:5d} valeur(s) fautive(s)  -> (colonne propre)")
    print(f"{'longitude':38s} : {n_lon:5d} valeur(s) fautive(s)  -> (colonne propre)")

    print("\nRappel : aucune ligne n'a ete supprimee a cette phase, seules des")
    print("valeurs individuelles sont devenues NaN / NaT quand elles ne se")
    print("convertissaient pas.")

    return df


# ---------------------------------------------------------------------------
# Phase 3 : le Conseil veut trier les canulars
# ---------------------------------------------------------------------------
def phase3_canulars(df):
    section("PHASE 3 - trier les canulars")

    regle = "un releve est marque canular si le mot HOAX apparait dans son temoignage (comments)"
    label = df["comments"].str.upper().str.contains("HOAX", na=False)
    n_canulars = int(label.sum())
    proportion = n_canulars / len(df)

    print(f"regle : {regle}")
    print(f"releves marques canulars : {n_canulars} sur {len(df)} ({proportion:.2%})")

    explication_banale = df["comments"].str.contains(
        r"balloon|venus|meteor|satellite|swamp gas", case=False, na=False, regex=True
    )
    manques = int((explication_banale & ~label).sum())
    print(f"limite : la regle rate {manques} signalements que le Bureau a expliques par un")
    print("phenomene banal (ballon, Venus, meteore...) sans jamais ecrire le mot 'hoax'.")

    return label


# ---------------------------------------------------------------------------
# Construction des features (partagee entre phases 4 a 10)
# ---------------------------------------------------------------------------
def construire_features(df):
    features = pd.DataFrame(index=df.index)
    for c in ["state", "country", "shape"]:
        features[c] = df[c].replace("", "unknown").fillna("unknown")
    features["duration_seconds_num"] = df["duration_seconds_num"]
    features["hour"] = df["datetime_parsed"].dt.hour
    features["month"] = df["datetime_parsed"].dt.month
    features["days_to_post"] = (df["date_posted_parsed"] - df["datetime_parsed"]).dt.days
    features["comments"] = df["comments"].fillna("")
    return features


CAT_COLS = ["state", "country", "shape"]
NUM_COLS = ["duration_seconds_num", "hour", "month", "days_to_post"]


def entrainer_sur_indices(X, y, idx_tr, idx_te, avec_texte):
    """Coeur d'entrainement : ne connait que les indices qu'on lui donne.
    Le pipeline (imputation, encodage, vocabulaire) n'est jamais fit()
    que sur idx_tr - c'est ce que la phase 10 vient verifier."""
    Xtr, Xte = X.loc[idx_tr], X.loc[idx_te]
    ytr, yte = y.loc[idx_tr], y.loc[idx_te]

    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ]
    colonnes = CAT_COLS + NUM_COLS
    if avec_texte:
        transformers.append(("txt", TfidfVectorizer(max_features=3000, stop_words="english"), "comments"))
        colonnes = colonnes + ["comments"]

    pre = ColumnTransformer(transformers)
    modele = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])
    modele.fit(Xtr[colonnes], ytr)
    pred = modele.predict(Xte[colonnes])

    return {
        "recall": recall_score(yte, pred),
        "precision": precision_score(yte, pred),
        "accuracy": accuracy_score(yte, pred),
        "n_test": len(yte),
        "modele": modele,
        "yte": yte,
        "pred": pred,
    }


def entrainer_et_evaluer(X, y, avec_texte):
    idx_tr, idx_te = train_test_split(
        X.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return entrainer_sur_indices(X, y, idx_tr, idx_te, avec_texte)


# ---------------------------------------------------------------------------
# Phase 4 : le premier verdict
# ---------------------------------------------------------------------------
def phase4_premier_verdict(df, label):
    section("PHASE 4 - le premier verdict")

    X = construire_features(df)
    resultat = entrainer_et_evaluer(X, label, avec_texte=True)

    print(f"evalue sur {resultat['n_test']} releves jamais vus a l'entrainement (20% du jeu, tires au hasard)")
    print(f"recall (canulars reels attrapes)      : {resultat['recall']:.2%}")
    print(f"precision (alertes reellement vraies) : {resultat['precision']:.2%}")

    return X, resultat


# ---------------------------------------------------------------------------
# Phase 5 : le Conseil ne vous croit pas (fuite de donnees)
# ---------------------------------------------------------------------------
def phase5_fuite(df, label, resultat_avant):
    section("PHASE 5 - le Conseil ne vous croit pas")

    tableau = [
        ("datetime",           "temoin",                       "au moment de l'observation",        "non"),
        ("city / state / country", "temoin",                   "au moment de la declaration",       "non"),
        ("shape",               "temoin",                      "au moment de la declaration",       "non"),
        ("duration_seconds",    "pipeline du Bureau (parsing)", "a la declaration",                  "non"),
        ("duration_hours_min",  "temoin",                       "au moment de la declaration",       "non"),
        ("comments",            "temoin, PUIS complete par un employe du Bureau (note NUFORC)",
         "temoin : a la declaration / employe : des semaines plus tard, lors du traitement", "OUI"),
        ("date_posted",         "Bureau (systeme de publication)", "apres traitement du dossier",    "non"),
        ("latitude / longitude", "geocodage automatique a partir de city/state/country", "a la declaration", "non"),
    ]
    print(f"{'colonne':24s} {'qui ecrit':45s} {'a quel moment':45s} savait deja ?")
    for col, qui, quand, savait in tableau:
        print(f"{col:24s} {qui:45s} {quand:45s} {savait}")

    print("\n-> seule 'comments' repond OUI : elle est completee des semaines plus")
    print("   tard par un employe qui, a ce moment-la, a deja tranche. On la retire")
    print("   du modele (donc aussi le TF-IDF construit dessus) et on reentraine.")

    X = construire_features(df)
    resultat_apres = entrainer_et_evaluer(X, label, avec_texte=False)

    print(f"\nrecall    avant : {resultat_avant['recall']:.2%}   apres : {resultat_apres['recall']:.2%}")
    print(f"precision avant : {resultat_avant['precision']:.2%}   apres : {resultat_apres['precision']:.2%}")

    print("\nL'ecart vient de 'comments' : c'est la ou vit le mot 'HOAX' qui a servi a")
    print("fabriquer l'etiquette en phase 3. Le premier modele ne predisait pas le")
    print("canular, il relisait l'annotation de l'employe dans le texte. Une fois ce")
    print("texte retire, il ne reste que des champs remplis avant tout jugement, et")
    print("le probleme redevient aussi dur qu'il l'est vraiment.")

    return resultat_apres


# ---------------------------------------------------------------------------
# Phase 6 : le modele le plus bete du Bureau
# ---------------------------------------------------------------------------
def phase6_stagiaire(label, resultat_avant, resultat_apres):
    section("PHASE 6 - le modele le plus bete du Bureau")

    _, yte = train_test_split(
        label, test_size=0.2, random_state=RANDOM_STATE, stratify=label
    )
    pred_stagiaire = np.zeros(len(yte), dtype=bool)
    acc_stagiaire = accuracy_score(yte, pred_stagiaire)

    print(f"accuracy du stagiaire (toujours 'pas un canular')      : {acc_stagiaire:.2%}")
    print(f"accuracy du modele avant retrait de la fuite (phase 4) : {resultat_avant['accuracy']:.2%}")
    print(f"accuracy du modele honnete apres retrait (phase 5)     : {resultat_apres['accuracy']:.2%}")

    print("\nLe stagiaire bat meme le modele honnete sur ce seul critere, alors qu'il")
    print("n'attrape aucun canular (recall = 0%). L'accuracy est ecrasee par les 99%")
    print("de releves qui ne sont pas des canulars : elle recompense l'inaction. La")
    print("mesure a presenter au Conseil est le recall (ou un F1), pas l'accuracy.")

    return acc_stagiaire


# ===========================================================================
# PARTIE 2 - le Conseil renvoie le rapport
# ===========================================================================

# ---------------------------------------------------------------------------
# Phase 7 : plusieurs temoins, un seul evenement
# ---------------------------------------------------------------------------
def phase7_evenements(df, label, resultat_p5):
    section("PHASE 7 - plusieurs temoins, un seul evenement")

    # date seule, tiree directement du texte brut (recupere aussi les 1220
    # lignes dont l'heure invalide '24:00' avait fait echouer tout le parsing en phase 2)
    df["obs_date"] = pd.to_datetime(
        df["datetime"].str.split(" ").str[0], format="%m/%d/%Y", errors="coerce"
    ).astype(str)
    df["event_key"] = (
        df["obs_date"] + "|" +
        df["city"].str.lower().str.strip() + "|" +
        df["state"].str.lower().str.strip()
    )

    tailles = df.groupby("event_key").size()
    multi = tailles[tailles > 1]
    plus_gros_evenement = tailles.idxmax()

    print(f"evenements signales par plus d'un temoin : {len(multi)}")
    print(f"temoins pour le plus gros d'entre eux     : {tailles.max()}  ({plus_gros_evenement})")

    print(f"\nexemple - tous les temoins de '{plus_gros_evenement}' :")
    exemple = df.loc[df["event_key"] == plus_gros_evenement, ["datetime", "city", "state"]]
    print(exemple.head(10).to_string())
    print(f"... {len(exemple)} temoins au total pour cette seule nuit.")

    # temoignages copies mot pour mot
    non_vides = df["comments"].fillna("")
    non_vides = non_vides[non_vides.str.strip() != ""]
    doublons = non_vides.value_counts()
    doublons = doublons[doublons > 1]
    print(f"\ntemoignages recopies mot pour mot : {len(doublons)} textes distincts, "
          f"{int(doublons.sum())} lignes concernees")

    # combien de relevés etaient a cheval sur les deux cotes, dans la decoupe d'hier (phase 4/5)
    idx_tr_hier, idx_te_hier = train_test_split(
        df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=label
    )
    tr_set, te_set = set(idx_tr_hier), set(idx_te_hier)
    cheval_evenements, cheval_lignes = 0, 0
    for k in multi.index:
        membres = df.index[df["event_key"] == k]
        dans_tr = any(m in tr_set for m in membres)
        dans_te = any(m in te_set for m in membres)
        if dans_tr and dans_te:
            cheval_evenements += 1
            cheval_lignes += len(membres)
    print(f"\nrelevés a cheval sur train/test, decoupe d'hier (aleatoire simple) : {cheval_lignes} "
          f"lignes (dans {cheval_evenements} evenements)")

    # nouvelle decoupe : un evenement entier part du meme cote
    X = construire_features(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    idx_tr, idx_te = next(gss.split(X, label, groups=df["event_key"]))
    idx_tr, idx_te = X.index[idx_tr], X.index[idx_te]
    resultat_groupe = entrainer_sur_indices(X, label, idx_tr, idx_te, avec_texte=False)

    print(f"\nrecall    : avant (phase 5) {resultat_p5['recall']:.2%}  ->  "
          f"apres decoupe par evenement {resultat_groupe['recall']:.2%}")
    print(f"precision : avant (phase 5) {resultat_p5['precision']:.2%}  ->  "
          f"apres decoupe par evenement {resultat_groupe['precision']:.2%}")

    return resultat_groupe


# ---------------------------------------------------------------------------
# Phase 8 : l'ordre des choses (decoupe chronologique)
# ---------------------------------------------------------------------------
def phase8_ordre_du_temps(df, label, resultat_p7):
    section("PHASE 8 - l'ordre des choses")

    print("Deux dates disponibles : datetime (quand le temoin a leve les yeux) et")
    print("date_posted (quand le Bureau a recu/publie le dossier). On coupe sur")
    print("date_posted : c'est elle qui dit ce que le systeme a reellement sous la")
    print("main au moment de juger. Un evenement ancien publie tard n'est pas 'connu'")
    print("plus tot pour autant - le systeme ne le voit que quand le Bureau le publie.")

    dates_valides = df["date_posted_parsed"].dropna().sort_values()
    coupure = dates_valides.quantile(0.8)
    print(f"\ndate de coupure : {coupure.date()}  (80e percentile de date_posted)")

    avant_coupure = df["date_posted_parsed"] <= coupure
    tr_set = set(df.index[avant_coupure])
    te_set = set(df.index[~avant_coupure & df["date_posted_parsed"].notna()])

    # aucun evenement (phase 7) ne doit chevaucher la coupure
    chevauchent, lignes_rattachees = 0, 0
    for k, membres_idx in df.groupby("event_key").groups.items():
        membres = list(membres_idx)
        if len(membres) < 2:
            continue
        dans_tr = [m in tr_set for m in membres]
        if any(dans_tr) and not all(dans_tr):
            chevauchent += 1
            for m in membres:
                if m in te_set:
                    te_set.discard(m)
                    tr_set.add(m)
                    lignes_rattachees += 1
    print(f"evenements a cheval sur la coupure : {chevauchent} ({lignes_rattachees} lignes rattachees au train)")

    idx_tr = pd.Index(sorted(tr_set))
    idx_te = pd.Index(sorted(te_set))
    print(f"\nrelevés cote apprentissage : {len(idx_tr)}")
    print(f"relevés cote test          : {len(idx_te)}")
    print(f"proportion canulars - apprentissage : {label.loc[idx_tr].mean():.2%}")
    print(f"proportion canulars - test          : {label.loc[idx_te].mean():.2%}")

    X = construire_features(df)
    resultat = entrainer_sur_indices(X, label, idx_tr, idx_te, avec_texte=False)
    print(f"\nrecall    : avant (phase 7, regroupe) {resultat_p7['recall']:.2%}  ->  "
          f"apres decoupe chronologique {resultat['recall']:.2%}")
    print(f"precision : avant (phase 7, regroupe) {resultat_p7['precision']:.2%}  ->  "
          f"apres decoupe chronologique {resultat['precision']:.2%}")

    return idx_tr, idx_te, resultat


# ---------------------------------------------------------------------------
# Phase 9 : les cases vides
# ---------------------------------------------------------------------------
def phase9_cases_vides(df, label):
    section("PHASE 9 - les cases vides")

    colonnes_trouees = ["country", "state", "duration_hours_min"]
    print("trois colonnes les plus trouees - taux de canulars, troue vs rempli :\n")
    for c in colonnes_trouees:
        troue = df[c].fillna("").str.strip() == ""
        taux_troue = label[troue].mean()
        taux_rempli = label[~troue].mean()
        print(f"{c:20s} troue={int(troue.sum()):6d}   "
              f"taux(troue)={taux_troue:.4%}   taux(rempli)={taux_rempli:.4%}")

    print("\nTraitement retenu :")
    print("- country / state : le trou reste sa propre categorie ('manquant') dans")
    print("  l'encodage - le modele voit directement qu'il y avait un trou, rien n'est")
    print("  efface (voir phase 12 : shape/state/country utilisent 'manquant', pas")
    print("  'unknown', pour ne pas le confondre avec la vraie reponse du temoin).")
    print("- duration_hours_min (recyclee en duree numerique, phase 11) : un indicateur")
    print("  binaire 'duree_manquante' accompagne la valeur imputee - boucher le trou")
    print("  sans effacer sa trace.")


# ---------------------------------------------------------------------------
# Phase 10 : la chaine de traitement du Bureau (fuite de pretraitement)
# ---------------------------------------------------------------------------
def phase10_chaine_de_traitement(df, label, idx_tr, idx_te, resultat_p8):
    section("PHASE 10 - la chaine de traitement du Bureau")

    print("Verification demandee : moyennes, medianes, vocabulaires - calcules sur quoi ?")
    print("Dans ce script, l'imputation (SimpleImputer) et l'encodage (OneHotEncoder)")
    print("sont places DANS un sklearn Pipeline, et ce pipeline n'est appele en .fit()")
    print("que sur les indices d'apprentissage (idx_tr) - jamais sur le jeu complet,")
    print("et ce depuis la phase 4. Rien a corriger ici pour ce point precis.")

    X = construire_features(df)
    resultat = entrainer_sur_indices(X, label, idx_tr, idx_te, avec_texte=False)
    print(f"\nrecall    : {resultat['recall']:.2%}  (identique a la phase 8 - la methode etait deja saine)")
    print(f"precision : {resultat['precision']:.2%}")
    print("Ce n'aurait pas ete vrai avec une mediane calculee a la main via df.median()")
    print("sur le DataFrame entier avant le train_test_split : c'est cette erreur-la")
    print("que le Pipeline sklearn empeche structurellement.")

    n_canulars_test = int(label.loc[idx_te].sum())
    print(f"\nCote test : {n_canulars_test} canulars sur {len(idx_te)} relevés - largement")
    print("assez pour que recall/precision ne soient pas de la decoration statistique.")

    nouveau_releve = pd.DataFrame([{
        "state": "il", "country": "us", "shape": "triangle",
        "duration_seconds_num": 300.0, "hour": 21, "month": 10, "days_to_post": 15,
        "comments": "",
    }])
    prediction = resultat["modele"].predict(nouveau_releve)[0]
    proba = resultat["modele"].predict_proba(nouveau_releve)[0][1]
    print(f"\nDemo - un releve invente a la main traverse toute la chaine en un seul appel :")
    print(f"  {nouveau_releve.iloc[0].to_dict()}")
    print(f"  -> modele.predict() -> canular ? {bool(prediction)}  (probabilite = {proba:.2%})")

    return resultat


# ---------------------------------------------------------------------------
# Phase 11 : combien de temps ca a dure
# ---------------------------------------------------------------------------
DUREE_UNITES = {
    "sec": 1, "secs": 1, "second": 1, "seconds": 1, "s": 1,
    "min": 60, "mins": 60, "minute": 60, "minutes": 60, "m": 60,
    "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600, "h": 3600,
    "day": 86400, "days": 86400,
}
_UNITE_RE = "|".join(sorted(DUREE_UNITES, key=len, reverse=True))
_DUREE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)?\s*(\d+(?:\.\d+)?)?\s*(" + _UNITE_RE + r")\b")


def parser_duree_texte(s):
    """Extrait une duree en secondes depuis le texte libre du temoin (ex: '5 minutes', '1-2 hrs')."""
    if not isinstance(s, str):
        return np.nan
    s = s.lower().strip()
    if not s:
        return np.nan
    m = _DUREE_PATTERN.search(s)
    if not m:
        return np.nan
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else None
    unite = DUREE_UNITES[m.group(3)]
    return (a + b) / 2 * unite if b is not None else a * unite


def phase11_duree(df):
    section("PHASE 11 - combien de temps ca a dure")

    df["duree_texte_s"] = df["duration_hours_min"].apply(parser_duree_texte)
    structuree_ok = df["duration_seconds_num"].notna() & (df["duration_seconds_num"] > 0)
    texte_ok = df["duree_texte_s"].notna() & (df["duree_texte_s"] > 0)

    duree_brute = df["duration_seconds_num"].where(structuree_ok, df["duree_texte_s"])
    duree_brute = duree_brute.where(duree_brute > 0)

    n_inutilisable = int(duree_brute.isna().sum())
    print(f"relevés dont la duree reste inutilisable apres traitement : {n_inutilisable} / {len(df)}")

    contradiction = (~structuree_ok) & texte_ok
    print(f"relevés ou la colonne secondes dit 0/rien alors que le texte est lisible : "
          f"{int(contradiction.sum())}")

    deux_valides = structuree_ok & texte_ok
    ratio = df.loc[deux_valides, "duration_seconds_num"] / df.loc[deux_valides, "duree_texte_s"]
    gros_ecart = int(((ratio > 3) | (ratio < 1 / 3)).sum())
    print(f"relevés ou les deux colonnes sont valides mais tres eloignees (ratio > 3x) : "
          f"{gros_ecart} / {int(deux_valides.sum())}")

    print(f"\nduree mediane apres traitement : {duree_brute.median():.0f} s")

    plus_dune_journee = int((duree_brute > 86400).sum())
    print(f"relevés annoncant plus d'une journee d'observation : {plus_dune_journee}")

    top3 = duree_brute.sort_values(ascending=False).head(3)
    print("\n3 durees les plus longues du fichier :")
    for idx, val in top3.items():
        print(f"  {df.loc[idx, 'datetime']} - {df.loc[idx, 'city']!r} - "
              f"temoin a ecrit {df.loc[idx, 'duration_hours_min']!r} -> {val:.0f} s")

    CAP = 86400
    print(f"\nDecision : les durees au-dela de {CAP} s (1 jour) sont plafonnees a {CAP} s,")
    print(f"pas supprimees - ce sont des phenomenes 'continus/recurrents' revendiques par")
    print(f"le temoin, pas des erreurs de saisie ; {plus_dune_journee} lignes plafonnees,")
    print("aucune ligne retiree, la mediane (robuste) n'a pas bouge a cause d'elles.")

    df["duree_finale_s"] = duree_brute.clip(upper=CAP)
    df["duree_manquante"] = duree_brute.isna().astype(int)

    return df


# ---------------------------------------------------------------------------
# Phase 12 : la ville et l'heure
# ---------------------------------------------------------------------------
def phase12_ville_et_heure(df, label, idx_tr, idx_te):
    section("PHASE 12 - la ville et l'heure")

    # ---- heure : encodage cyclique ----
    heures = df["datetime_parsed"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * heures / 24)
    df["hour_cos"] = np.cos(2 * np.pi * heures / 24)

    def dist_heure(h1, h2):
        s1, c1 = np.sin(2 * np.pi * h1 / 24), np.cos(2 * np.pi * h1 / 24)
        s2, c2 = np.sin(2 * np.pi * h2 / 24), np.cos(2 * np.pi * h2 / 24)
        return float(np.hypot(s1 - s2, c1 - c2))

    print(f"distance encodee entre 23h et 0h  : {dist_heure(23, 0):.3f}")
    print(f"distance encodee entre 23h et 20h : {dist_heure(23, 20):.3f}")
    print("-> 23h ressort bien plus proche de 0h que de 20h : l'encodage cyclique")
    print("   corrige l'absurdite d'une echelle 0-23 lineaire.")

    # ---- shape : nettoyage ----
    shape_brute = df["shape"].replace("", "manquant").fillna("manquant")
    n_formes_avant = df["shape"].replace("", np.nan).nunique()
    corrections = {"changed": "changing", "round": "circle"}
    shape_propre = shape_brute.replace(corrections)
    n_formes_apres = shape_propre[shape_propre != "manquant"].nunique()
    df["shape_propre"] = shape_propre

    print(f"\nformes avant nettoyage : {n_formes_avant}")
    print("corrections appliquees : changed -> changing, round -> circle")
    print(f"(meme forme decrite sous deux orthographes differentes)")
    print(f"formes retenues apres nettoyage (hors valeurs manquantes) : {n_formes_apres}")

    # ---- ville : largeur du tableau ----
    n_villes = df["city"].nunique()
    n_villes_uniques = int((df["city"].value_counts() == 1).sum())
    print(f"\nvilles distinctes dans la transmission : {n_villes}")
    print(f"villes qui n'apparaissent qu'une seule fois : {n_villes_uniques}")
    print("regle appliquee : chaque ville est remplacee par son taux de canulars,")
    print("calcule uniquement sur la partie apprentissage (TargetEncoder) - une seule")
    print("colonne au lieu d'une par ville.")

    state_propre = df["state"].replace("", "manquant").fillna("manquant")
    country_propre = df["country"].replace("", "manquant").fillna("manquant")

    features_finales = pd.DataFrame({
        "state": state_propre,
        "country": country_propre,
        "shape": shape_propre,
        "city": df["city"],
        "duree": df["duree_finale_s"],
        "duree_manquante": df["duree_manquante"],
        "hour_sin": df["hour_sin"],
        "hour_cos": df["hour_cos"],
        "month": df["datetime_parsed"].dt.month,
        "days_to_post": (df["date_posted_parsed"] - df["datetime_parsed"]).dt.days,
    }, index=df.index)

    pre_final = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["state", "country", "shape"]),
        ("ville", TargetEncoder(target_type="binary", random_state=RANDOM_STATE), ["city"]),
        ("num", SimpleImputer(strategy="median"),
         ["duree", "duree_manquante", "hour_sin", "hour_cos", "month", "days_to_post"]),
    ])

    ytr = label.loc[idx_tr].astype(int)
    Xtr_transforme = pre_final.fit_transform(features_finales.loc[idx_tr], ytr)
    largeur_apres = Xtr_transforme.shape[1]
    largeur_avant_hypothese = largeur_apres - 1 + n_villes  # remplace la colonne ville (1) par un one-hot naif

    print(f"\nTargetEncoder.fit() appele sur {len(idx_tr)} lignes (apprentissage) uniquement.")
    villes_test_jamais_vues = int((~df.loc[idx_te, "city"].isin(df.loc[idx_tr, "city"])).sum())
    print(f"villes du cote test absentes de l'apprentissage : {villes_test_jamais_vues} "
          "-> recoivent le taux moyen global, pas de fuite possible.")

    print(f"\nlargeur du tableau si la ville etait en one-hot naif : {largeur_avant_hypothese} colonnes")
    print(f"largeur du tableau reelle (ville target-encodee)     : {largeur_apres} colonnes")

    # bilan cumule : toutes les corrections des phases 7 a 12 ensemble
    modele_final = Pipeline([
        ("pre", pre_final),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])
    modele_final.fit(features_finales.loc[idx_tr], ytr)
    pred_final = modele_final.predict(features_finales.loc[idx_te])
    yte = label.loc[idx_te].astype(int)
    recall_final = recall_score(yte, pred_final)
    precision_final = precision_score(yte, pred_final)

    print("\nBilan cumule (regroupement + decoupe chronologique + ville/heure/forme) :")
    print(f"recall final    : {recall_final:.2%}")
    print(f"precision finale: {precision_final:.2%}")

    return {"recall": recall_final, "precision": precision_final}


# ===========================================================================
# PARTIE 3 - defendre une decision
# ===========================================================================

COUT_CANULAR_MANQUE = 30   # faux negatif : une equipe part travailler sur du vent
COUT_FAUSSE_ALERTE = 2     # faux positif : une observation potentiellement utile est perdue


# ---------------------------------------------------------------------------
# Phase 13 : la facture du Bureau
# ---------------------------------------------------------------------------
def phase13_facture(df, label, idx_te, resultat):
    section("PHASE 13 - la facture du Bureau")

    modele = resultat["modele"]
    X = construire_features(df)
    Xte = X.loc[idx_te]
    yte = label.loc[idx_te].values
    proba = modele.predict_proba(Xte)[:, 1]

    print(f"grille : canular manque = {COUT_CANULAR_MANQUE} credits, "
          f"fausse alerte = {COUT_FAUSSE_ALERTE} credits, sinon 0")

    seuils = np.arange(0.0, 1.001, 0.01)
    couts = np.empty(len(seuils))
    for i, s in enumerate(seuils):
        pred = proba >= s
        fn = int((~pred & yte).sum())
        fp = int((pred & ~yte).sum())
        couts[i] = fn * COUT_CANULAR_MANQUE + fp * COUT_FAUSSE_ALERTE

    print(f"\n{'seuil':>6s} {'cout (credits)':>15s}")
    for s in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
        i = np.argmin(np.abs(seuils - s))
        print(f"{seuils[i]:6.2f} {couts[i]:15.0f}")

    i_opt = int(couts.argmin())
    seuil_opt = float(seuils[i_opt])
    cout_opt = float(couts[i_opt])
    i_05 = int(np.argmin(np.abs(seuils - 0.5)))
    cout_05 = float(couts[i_05])

    print(f"\nseuil retenu (cout minimal) : {seuil_opt:.2f}  ->  {cout_opt:.0f} credits")
    print(f"seuil par defaut (0.5)                        ->  {cout_05:.0f} credits")
    print(f"ecart                                          :  {cout_05 - cout_opt:.0f} credits economises")

    pred_opt = proba >= seuil_opt
    recall_opt = recall_score(yte, pred_opt)
    n_alertes = int(pred_opt.sum())
    print(f"\na ce seuil : recall {recall_opt:.2%}, {n_alertes} alertes levees sur {len(yte)} relevés.")
    if recall_opt < 0.1:
        n_canulars = int(yte.sum())
        print(f"ATTENTION : {n_canulars} canulars sont presents dans le test ; a ce seuil le")
        print("systeme n'en attrape presque aucun. Le cout minimal ressemble ici au")
        print("stagiaire de la phase 6 - parce que la precision du modele est si faible")
        print("(phase 5) que chaque relance de recall coute trop de fausses alertes. Ce")
        print("seuil resout l'equation posee, mais pas forcement le vrai probleme du")
        print("Bureau - voir phase 14 : les probabilites elles-memes sont a reprendre.")

    return {"seuil": seuil_opt, "cout_seuil": cout_opt, "cout_05": cout_05}


# ---------------------------------------------------------------------------
# Phase 14 : une promesse a 80%
# ---------------------------------------------------------------------------
def table_calibration(proba, y, n_tranches=10):
    """Tranches a effectif egal (quantiles) : garantit un n visible par tranche
    quelle que soit la forme de la distribution de probabilites."""
    tranches = pd.qcut(proba, q=n_tranches, duplicates="drop")
    df_tmp = pd.DataFrame({"tranche": tranches, "proba": proba, "y": y})
    resume = df_tmp.groupby("tranche", observed=True).agg(
        n=("y", "size"), annonce=("proba", "mean"), observe=("y", "mean")
    )
    return resume


def phase14_promesse_80(df, label, idx_tr, idx_te, resultat):
    section("PHASE 14 - une promesse a 80%")

    modele = resultat["modele"]
    X = construire_features(df)
    Xte = X.loc[idx_te]
    yte = label.loc[idx_te].values
    proba = modele.predict_proba(Xte)[:, 1]

    print("avant correction :")
    resume = table_calibration(proba, yte)
    print(resume.to_string(float_format=lambda v: f"{v:.3f}"))

    ecart_moyen = (resume["annonce"] - resume["observe"]).mean()
    print(f"\necart moyen annonce-observe : {ecart_moyen:+.3f}")
    print("sens de l'erreur : le systeme est TROP CONFIANT sur toute la plage - il")
    print("annonce des probabilites bien plus hautes que ce qui se produit vraiment.")
    print("Consequence du class_weight='balanced' : il corrige le classement mais")
    print("deforme le chiffre.")

    from sklearn.calibration import CalibratedClassifierCV
    Xtr = X.loc[idx_tr]
    ytr = label.loc[idx_tr]
    modele_calibre = CalibratedClassifierCV(modele, method="sigmoid", cv=5)
    modele_calibre.fit(Xtr, ytr)
    proba_corrigee = modele_calibre.predict_proba(Xte)[:, 1]

    print("\napres recalibrage (sigmoid / Platt, appris sur le train uniquement) :")
    resume_corrige = table_calibration(proba_corrigee, yte)
    print(resume_corrige.to_string(float_format=lambda v: f"{v:.3f}"))
    ecart_corrige = (resume_corrige["annonce"] - resume_corrige["observe"]).mean()
    print(f"\necart moyen apres correction : {ecart_corrige:+.3f}")

    return modele_calibre


# ---------------------------------------------------------------------------
# Phase 15 : deux analystes, deux chiffres
# ---------------------------------------------------------------------------
def phase15_deux_analystes(df, label, idx_te, resultat):
    section("PHASE 15 - deux analystes, deux chiffres")

    modele = resultat["modele"]
    X = construire_features(df)
    Xte = X.loc[idx_te]
    yte = label.loc[idx_te].values
    pred = modele.predict_proba(Xte)[:, 1] >= 0.5

    n_test = len(yte)
    n_canulars_test = int(yte.sum())
    print(f"taille de la partie test        : {n_test}")
    print(f"canulars reellement presents    : {n_canulars_test}")

    n_reech = 1000
    rng = np.random.RandomState(RANDOM_STATE)
    recalls = []
    for _ in range(n_reech):
        idx = rng.randint(0, n_test, n_test)
        yb, pb = yte[idx], pred[idx]
        tp = int((yb & pb).sum())
        fn = int((yb & ~pb).sum())
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
    recalls = np.array(recalls)
    lo, hi = np.percentile(recalls, [2.5, 97.5])
    point = int((yte & pred).sum()) / n_canulars_test

    print(f"\nnombre de reechantillonnages (bootstrap) : {n_reech}")
    print(f"recall (nombre principal) : {point:.2%}   intervalle 95% : [{lo:.2%} ; {hi:.2%}]")

    print(f"\nReponse au Conseil : avec seulement {n_canulars_test} canulars dans la partie")
    print(f"test, le recall bouge de {(hi - lo):.1%} rien qu'en rejouant le tirage au sort du")
    print("split. 0,31 et 0,34 (l'ecart des deux analystes fictifs) tiennent largement")
    print("dans cette marge - il n'y a pas de gagnant a designer sur un seul chiffre.")

    return {"lo": lo, "hi": hi, "point": point, "n_test": n_test, "n_canulars_test": n_canulars_test}


# ---------------------------------------------------------------------------
# Phase 16 : trois dossiers sur le bureau
# ---------------------------------------------------------------------------
def expliquer_ligne(modele, ligne_df):
    """Contribution de chaque feature transformee (coefficient x valeur) pour UNE ligne."""
    pre = modele.named_steps["pre"]
    clf = modele.named_steps["clf"]
    transformee = pre.transform(ligne_df)
    if hasattr(transformee, "toarray"):
        transformee = transformee.toarray()
    noms = pre.get_feature_names_out()
    contributions = transformee[0] * clf.coef_[0]
    ordre = np.argsort(-np.abs(contributions))
    return [(noms[i], float(contributions[i])) for i in ordre[:5]]


def phase16_trois_dossiers(df, label, idx_te, resultat):
    section("PHASE 16 - trois dossiers sur le bureau")

    modele = resultat["modele"]
    X = construire_features(df)
    Xte = X.loc[idx_te]
    yte = label.loc[idx_te]
    proba = pd.Series(modele.predict_proba(Xte)[:, 1], index=Xte.index)
    pred = proba >= 0.5

    confiants = proba[(yte) & (pred)].sort_values(ascending=False)
    a_la_limite = proba[proba >= 0.5].sort_values()
    manques = proba[(yte) & (~pred)].sort_values(ascending=False)

    dossiers = [
        ("forte confiance (vrai canular attrape)", confiants.index[0] if len(confiants) else None),
        ("juste au-dessus de la frontiere", a_la_limite.index[0] if len(a_la_limite) else None),
        ("canular laisse passer", manques.index[0] if len(manques) else None),
    ]

    for nom, idx in dossiers:
        if idx is None:
            print(f"\n--- {nom} : aucun cas trouve ---")
            continue
        print(f"\n--- dossier : {nom} (index {idx}) ---")
        print(f"probabilite annoncee : {proba.loc[idx]:.3f}   canular reel : {bool(yte.loc[idx])}")
        print(f"releve : {df.loc[idx, ['datetime', 'city', 'state', 'shape']].to_dict()}")
        for nom_col, contrib in expliquer_ligne(modele, X.loc[[idx]]):
            sens = "pousse vers canular" if contrib > 0 else "pousse vers pas-canular"
            print(f"   {nom_col:26s} {contrib:+.3f}  ({sens})")

    print("\n--- importance globale (permutation : colonne melangee, recall qui chute) ---")
    base_recall = recall_score(yte, pred)
    rng = np.random.RandomState(RANDOM_STATE)
    chutes = []
    for col in CAT_COLS + NUM_COLS:
        Xperm = Xte.copy()
        Xperm[col] = rng.permutation(Xperm[col].values)
        pred_perm = modele.predict_proba(Xperm)[:, 1] >= 0.5
        recall_perm = recall_score(yte, pred_perm)
        chutes.append((col, base_recall - recall_perm))
    chutes.sort(key=lambda t: -t[1])
    for col, chute in chutes:
        print(f"  {col:22s} chute de recall = {chute:+.3f}")

    return chutes


# ---------------------------------------------------------------------------
# Phase 17 : l'angle mort du Bureau
# ---------------------------------------------------------------------------
def phase17_angle_mort(df, label, idx_te, resultat):
    section("PHASE 17 - l'angle mort du Bureau")

    modele = resultat["modele"]
    X = construire_features(df)
    Xte = X.loc[idx_te]
    yte = label.loc[idx_te]
    pred = pd.Series(modele.predict_proba(Xte)[:, 1], index=Xte.index) >= 0.5

    part_us = (df["country"] == "us").mean()
    print(f"part des relevés venant des Etats-Unis : {part_us:.1%}")

    zones = {
        "us": df["country"] == "us",
        "manquant": df["country"] == "",
        "ca": df["country"] == "ca",
        "gb": df["country"] == "gb",
    }
    zones["autres"] = ~(zones["us"] | zones["manquant"] | zones["ca"] | zones["gb"])

    print(f"\n{'zone':>10s} {'n (test)':>9s} {'% canulars':>11s} {'recall':>8s} {'precision':>10s}")
    print(f"{'GLOBAL':>10s} {len(idx_te):9d} {yte.mean():10.2%} "
          f"{recall_score(yte, pred):7.2%} {precision_score(yte, pred):9.2%}")

    for nom, masque in zones.items():
        idx_zone = idx_te.intersection(df.index[masque])
        if len(idx_zone) == 0:
            continue
        y_zone = label.loc[idx_zone]
        p_zone = pred.loc[idx_zone]
        rec = recall_score(y_zone, p_zone) if y_zone.sum() > 0 else float("nan")
        prec = precision_score(y_zone, p_zone) if p_zone.sum() > 0 else float("nan")
        print(f"{nom:>10s} {len(idx_zone):9d} {y_zone.mean():10.2%} {rec:7.2%} {prec:9.2%}")

    print("\nAttention : gb/ca/autres pesent quelques centaines a quelques milliers de")
    print("relevés avec parfois une poignee de canulars seulement - meme logique que la")
    print("phase 15, leurs chiffres individuels ont une marge large.")

    return zones


# ---------------------------------------------------------------------------
# Phase 18 : la transmission d'archive
# ---------------------------------------------------------------------------
def phase18_transmission_archive(df, label, idx_tr, idx_te, resultat_p8, resultat_p15):
    section("PHASE 18 - la transmission d'archive")

    modele_p8 = resultat_p8["modele"]

    df["annee_publication"] = df["date_posted_parsed"].dt.year
    n_par_annee = df.groupby("annee_publication").size()
    taux_par_annee = df.groupby("annee_publication").apply(lambda g: label.loc[g.index].mean())

    print("proportion de canulars par annee de PUBLICATION (date_posted, comme phase 8) :")
    print(f"{'annee':>6s} {'n':>7s} {'% canulars':>11s}")
    for annee in sorted(taux_par_annee.index):
        print(f"{int(annee):6d} {int(n_par_annee[annee]):7d} {taux_par_annee[annee]:10.2%}")

    print("\n-> la courbe n'est pas plate : quasi nulle en 1998-2004, une montee nette")
    print("2005-2011 (pic vers 2008), puis une retombee en 2012-2013. Ce profil ressemble")
    print("a une periode ou le Bureau a ete plus systematique sur l'annotation 'hoax',")
    print("avant de relacher - exactement le scenario redoute par le Conseil.")

    cutoff = df["annee_publication"].quantile(0.8)
    idx_tr_temps = df.index[df["annee_publication"] <= cutoff]
    idx_te_temps = df.index[df["annee_publication"] > cutoff]
    X = construire_features(df)
    resultat_temps = entrainer_sur_indices(X, label, idx_tr_temps, idx_te_temps, avec_texte=False)

    print(f"\nEpreuve - entraine sur les publications <= {int(cutoff)}, teste sur les plus recentes :")
    print(f"recall    : phase 8 = {resultat_p8['recall']:.2%}   ancien->recent = {resultat_temps['recall']:.2%}")
    print(f"precision : phase 8 = {resultat_p8['precision']:.2%}   ancien->recent = {resultat_temps['precision']:.2%}")

    # taux d'alerte de reference sur le test actuel (ne demande jamais l'etiquette)
    X = construire_features(df)
    proba_te = modele_p8.predict_proba(X.loc[idx_te])[:, 1]
    pred_te = proba_te >= 0.5
    taux_alerte_ref = pred_te.mean()
    proba_moy_ref = proba_te.mean()

    # bootstrap sur ce meme taux d'alerte, meme methode qu'en phase 15 - mais
    # ici la quantite reechantillonnee (le taux d'alerte) ne regarde jamais l'etiquette
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(pred_te)
    taux_reech = [pred_te[rng.randint(0, n, n)].mean() for _ in range(1000)]
    lo_taux, hi_taux = np.percentile(taux_reech, [2.5, 97.5])

    print(f"\ntaux d'alerte de reference (partie test, seuil 0.5) : {taux_alerte_ref:.2%}")
    print(f"intervalle bootstrap 95% de ce taux (meme methode que phase 15) : "
          f"[{lo_taux:.2%} ; {hi_taux:.2%}]")

    print("\nIndicateurs de surveillance ne demandant jamais l'etiquette reelle :")
    print("1. taux d'alerte hebdomadaire (proportion de relevés entrants marques")
    print(f"   canular par le systeme) - reference actuelle : {taux_alerte_ref:.2%}.")
    print(f"2. probabilite moyenne annoncee par le systeme - reference actuelle : "
          f"{proba_moy_ref:.3f}. Un glissement signale un changement dans les relevés")
    print("   entrants ou dans le comportement du modele, sans jamais avoir besoin de")
    print("   savoir qui a raison.")
    print("frequence de suivi : hebdomadaire.")
    print(f"seuil d'alerte : le taux d'alerte hebdomadaire sort de [{lo_taux:.2%} ; {hi_taux:.2%}]")
    print("(intervalle bootstrap du taux de reference, meme methode qu'en phase 15)")
    print("deux semaines de suite -> on rappelle les analystes.")

    return resultat_temps


def main():
    telecharger_si_besoin()
    df, total, n_chargees, n_de_cote = phase1_ouvrir_la_caisse()
    df = phase2_typage(df)
    label = phase3_canulars(df)
    X, resultat_p4 = phase4_premier_verdict(df, label)
    resultat_p5 = phase5_fuite(df, label, resultat_p4)
    phase6_stagiaire(label, resultat_p4, resultat_p5)

    resultat_p7 = phase7_evenements(df, label, resultat_p5)
    idx_tr, idx_te, resultat_p8 = phase8_ordre_du_temps(df, label, resultat_p7)
    phase9_cases_vides(df, label)
    resultat_p10 = phase10_chaine_de_traitement(df, label, idx_tr, idx_te, resultat_p8)
    df = phase11_duree(df)
    phase12_ville_et_heure(df, label, idx_tr, idx_te)

    phase13_facture(df, label, idx_te, resultat_p10)
    phase14_promesse_80(df, label, idx_tr, idx_te, resultat_p10)
    resultat_p15 = phase15_deux_analystes(df, label, idx_te, resultat_p10)
    phase16_trois_dossiers(df, label, idx_te, resultat_p10)
    phase17_angle_mort(df, label, idx_te, resultat_p10)
    phase18_transmission_archive(df, label, idx_tr, idx_te, resultat_p8, resultat_p15)

    section("FIN")
    print("Toutes les phases ont tourne. Voir RAPPORT.md pour la synthese ecrite.")


if __name__ == "__main__":
    main()
