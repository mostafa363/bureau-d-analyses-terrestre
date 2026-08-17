"""
Bureau d'Analyse Terrestre - reception des releves (Klaxo-3)
Script unique, se relance du telechargement au dernier chiffre.
"""

import csv
import os
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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

    # coordonnees et duree -> nombres
    df["latitude_num"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude_num"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["duration_seconds_num"] = pd.to_numeric(df["duration_seconds"], errors="coerce")

    # les deux dates -> dates
    df["datetime_parsed"] = pd.to_datetime(df["datetime"], format="%m/%d/%Y %H:%M", errors="coerce")
    df["date_posted_parsed"] = pd.to_datetime(df["date_posted"], format="%m/%d/%Y", errors="coerce")

    anomalies = []

    # latitude : valeurs non vides qui refusent la conversion
    mask = df["latitude_num"].isna() & df["latitude"].ne("")
    valeurs = df.loc[mask, "latitude"].tolist()
    anomalies.append(("latitude", len(valeurs), valeurs[:5], "service de transmission (geocodage corrompu)"))

    # duration_seconds : deux sous-anomalies de nature differente
    mask_txt = df["duration_seconds_num"].isna() & df["duration_seconds"].ne("")
    valeurs_txt = df.loc[mask_txt, "duration_seconds"].tolist()
    anomalies.append(("duration_seconds (caractere parasite)", len(valeurs_txt), valeurs_txt,
                       "service de transmission (parsing automatique du texte du temoin)"))

    mask_vide = df["duration_seconds"].eq("")
    anomalies.append(("duration_seconds (valeur manquante)", int(mask_vide.sum()), ["''"] * min(3, int(mask_vide.sum())),
                       "temoin (n'a jamais donne de duree exploitable)"))

    # datetime : toutes les valeurs invalides sont des "24:00"
    mask_dt = df["datetime_parsed"].isna() & df["datetime"].ne("")
    valeurs_dt = df.loc[mask_dt, "datetime"].tolist()
    anomalies.append(("datetime (heure '24:00')", len(valeurs_dt), valeurs_dt[:5],
                       "temoin (a ecrit minuit sous la forme 24:00, heure qui n'existe pas)"))

    # date_posted et longitude : verifiees, propres (mentionnees pour contraste)
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

    # limite connue : la regle rate les signalements expliques sans le mot "hoax"
    explication_banale = df["comments"].str.contains(
        r"balloon|venus|meteor|satellite|swamp gas", case=False, na=False, regex=True
    )
    manques = int((explication_banale & ~label).sum())
    print(f"limite : la regle rate {manques} signalements que le Bureau a expliques par un")
    print("phenomene banal (ballon, Venus, meteore...) sans jamais ecrire le mot 'hoax'.")

    return label


# ---------------------------------------------------------------------------
# Construction des features (partagee entre phase 4 et phase 5)
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


def entrainer_et_evaluer(X, y, avec_texte):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

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


def main():
    telecharger_si_besoin()
    df, total, n_chargees, n_de_cote = phase1_ouvrir_la_caisse()
    df = phase2_typage(df)
    label = phase3_canulars(df)
    X, resultat_avant = phase4_premier_verdict(df, label)
    resultat_apres = phase5_fuite(df, label, resultat_avant)
    phase6_stagiaire(label, resultat_avant, resultat_apres)

    section("FIN")
    print("Toutes les phases ont tourne. Voir RAPPORT.md pour la synthese ecrite.")


if __name__ == "__main__":
    main()
