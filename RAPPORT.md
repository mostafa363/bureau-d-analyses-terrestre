# Rapport — Bureau d'Analyse Terrestre

Notes brutes (chiffres + décisions). Les `[ ]` sont les endroits où tu
écris toi-même, en une ou deux phrases, avec tes mots.

## Phase 1 — ouvrir la caisse

- fichier : 88 875 lignes
- chargées : 88 679
- mises de côté : 196
- 88 679 + 196 = 88 875 → ça colle
- cause des 196 : elles ont 12 champs au lieu de 11
- exemple de ligne fautive :
  `['10/1/2006 12:00', '', '', '', '', '0', '', '', '((EDITORIAL COMMENT...))', '10/30/2006', '0', '0']`
- `city` est vide → tout le reste est décalé d'une case
- ce sont plutôt des commentaires généraux sur le phénomène ovni, pas de vraies observations
- [ ] pourquoi tu les as mises de côté plutôt que d'essayer de les réparer

## Phase 2 — rien n'est du bon type

- conversion : `latitude`, `longitude`, `duration_seconds` → nombres ; `datetime`, `date_posted` → dates
- aucune ligne supprimée, valeur fautive → NaN / NaT

| Champ | Valeurs fautives | Exemple |
|---|---|---|
| `latitude` | 1 | `33q.200088` |
| `duration_seconds` | 3 | `2\``, `8\``, `0.5\`` |
| `duration_seconds` | 2 | (valeur vide) |
| `datetime` | 1220 | `10/10/2005 24:00` |
| `date_posted` | 0 | — |
| `longitude` | 0 | — |

- fait notable : `latitude` a 88 679 valeurs, une seule (`33q.200088`) suffit à faire planter toute la colonne si on ne force pas le typage
- [ ] pour chacune des 4 anomalies : témoin, capteur, ou service de transmission — et pourquoi tu penses ça

## Phase 3 — trier les canulars

- règle utilisée : `comments` contient le mot "HOAX" → canular
- résultat : 802 / 88 679 = 0,90 %
- limite trouvée : 2016 signalements expliqués par un truc banal (ballon, Vénus, météore, satellite) mais sans le mot "hoax" → la règle ne les voit pas
- [ ] la règle en une phrase, avec tes mots
- [ ] pourquoi cette limite est un problème (ou pas)

## Phase 4 — le premier verdict

- modèle : régression logistique, `class_weight="balanced"`
- évalué sur 17 736 lignes jamais vues à l'entraînement (20 %, split stratifié, seed fixe)
- recall : 99,4 %
- precision : 98,8 %
- [ ] présente ces deux chiffres et ce qu'ils veulent dire concrètement

## Phase 5 — le Conseil ne vous croit pas

| Colonne | Qui écrit | Quand | Savait déjà si canular ? |
|---|---|---|---|
| `datetime` | témoin | le soir même | non |
| `city` / `state` / `country` | témoin | à la déclaration | non |
| `shape` | témoin | à la déclaration | non |
| `duration_seconds` | extrait auto du texte | à la déclaration | non |
| `duration_hours_min` | témoin | à la déclaration | non |
| `comments` | témoin, puis employé du Bureau | témoin : le soir même / employé : des semaines après | **oui** |
| `date_posted` | Bureau | après traitement | non |
| `latitude` / `longitude` | géocodage auto | à la déclaration | non |

- seule `comments` = "oui"
- preuve : le texte brut contient des notes du type `((NUFORC Note: Possible hoax?? PD))`, ajoutées après coup
- le mot "HOAX" de cette note = le même mot utilisé pour fabriquer l'étiquette en phase 3
- `comments` retiré (donc le TF-IDF dessus aussi) → réentraînement sur les mêmes autres colonnes

| | Avant | Après |
|---|---|---|
| Recall | 99,4 % | 58,1 % |
| Precision | 98,8 % | 1,3 % |

- [ ] explique l'écart en 3 lignes : pourquoi le chiffre "avant" n'avait pas le droit d'exister (pas juste "le modèle est devenu moins bon")

## Phase 6 — le modèle le plus bête du Bureau

- système du stagiaire : toujours répondre "pas un canular"
- accuracy stagiaire : 99,1 %
- accuracy modèle avant (phase 4) : 100,0 %
- accuracy modèle après (phase 5) : 60,5 %
- recall du stagiaire : 0 % (il n'attrape jamais rien)
- canulars = seulement 0,9 % des relevés → dire "non" tout le temps donne une accuracy très haute sans rien détecter
- [ ] quelle mesure tu présentes au Conseil pour prouver que ton travail vaut mieux que celui du stagiaire, et pourquoi (indice : accuracy vs recall)

---

# Partie 2 — le Conseil renvoie le rapport

## Phase 7 — plusieurs témoins, un seul événement

- clé d'événement utilisée : date de l'observation + ville + état (`date|city|state`)
- événements signalés par plus d'un témoin : 2433
- témoins pour le plus gros événement : 56 — nuit du 31/10/2004, Tinley Park (Illinois)
- témoignages recopiés mot pour mot : 251 textes distincts, 612 lignes concernées
- découpe d'hier (aléatoire simple, phase 4/5) : 2089 relevés à cheval entre train et test, répartis sur 833 événements
- nouvelle découpe : un événement entier part toujours du même côté (`GroupShuffleSplit`)

| | Avant (découpe aléatoire, phase 5) | Après (découpe par événement) |
|---|---|---|
| Recall | 58,1 % | 60,9 % |
| Precision | 1,3 % | 1,3 % |

- [ ] quelles colonnes tu utilises pour reconnaître un même événement, et pourquoi celles-là
- [ ] ce que tu fais des témoignages copiés mot pour mot (612 lignes) — gardés ? retirés ? pourquoi
- [ ] pourquoi le recall bouge (un peu) alors que la precision ne bouge presque pas

## Phase 8 — l'ordre des choses

- deux dates disponibles : `datetime` (le témoin) et `date_posted` (le Bureau)
- date de coupure choisie : **13/05/2012** (80ᵉ percentile de `date_posted`)
- 38 événements (39 relevés) chevauchaient la coupure → rattachés entièrement au côté apprentissage
- relevés côté apprentissage : 71 258
- relevés côté test : 17 421
- proportion de canulars — apprentissage : **0,94 %**
- proportion de canulars — test : **0,76 %**

| | Avant (phase 7, groupé) | Après (découpe chronologique) |
|---|---|---|
| Recall | 60,9 % | 68,2 % |
| Precision | 1,3 % | 1,4 % |

- [ ] laquelle des deux dates tu as utilisée pour couper, et pourquoi celle-là
- [ ] les deux proportions de canulars (0,94 % vs 0,76 %) ne sont pas égales — qu'est-ce que ça veut dire, en deux lignes

## Phase 9 — les cases vides

| Colonne | Trouée (nb) | Taux canular si trouée | Taux canular si remplie |
|---|---|---|---|
| `country` | 12 365 | 1,16 % | 0,86 % |
| `state` | 7 409 | 1,30 % | 0,87 % |
| `duration_hours_min` | 3 017 | 2,35 % | 0,85 % |

- traitement retenu : `country`/`state` gardent le trou comme catégorie à part (`manquant`, pas `unknown`) ; `duration_hours_min` (recyclée en durée numérique, phase 11) reçoit un indicateur binaire `duree_manquante` en plus de la valeur imputée
- [ ] pourquoi ce traitement ne détruit pas ce que tu viens de mesurer (le fait qu'un trou soit informatif)

## Phase 10 — la chaîne de traitement du Bureau

- vérifié : imputation (`SimpleImputer`) et encodage (`OneHotEncoder`) sont dans un `sklearn.Pipeline`, jamais fit ailleurs que sur les indices d'apprentissage — et ce depuis la phase 4
- recall / precision : **identiques** à la phase 8 (68,2 % / 1,4 %) — rien à corriger ici
- relevés canulars côté test : 132 sur 17 421 (pas un test vide)
- démo : un relevé inventé à la main passe par `modele.predict()` en un seul appel → prédiction rendue avec probabilité (51,9 %)
- [ ] pourquoi les chiffres ne bougent pas ici alors qu'ils ont bougé aux phases 7 et 8 (indice : c'est une bonne nouvelle, pas un bug)

## Phase 11 — combien de temps ça a duré

- durée inutilisable après traitement : 7 024 / 88 679
- colonnes en contradiction (secondes = 0/vide, texte lisible) : 8
- colonnes très éloignées l'une de l'autre (valides toutes les deux, ratio > 3x) : 504 / 75 162
- durée médiane après traitement : 180 s (3 minutes)
- relevés annonçant plus d'une journée d'observation : 189
- 3 durées les plus longues : "31 years" (Birmingham UK, 1983), "23000hrs" (Ottawa, 2010), "21 years" (Greenbrier, 1991)
- décision retenue : plafonner à 86 400 s (1 jour), aucune ligne supprimée

- [ ] tes deux natures d'aberration nommées avec leur compte (tu as au moins : contradictions=8, gros écarts=504)
- [ ] pourquoi plafonner plutôt que supprimer ces 3 lignes extrêmes — qu'est-ce que ça aurait fait à la médiane si tu les avais juste laissées telles quelles ou si tu les avais retirées

## Phase 12 — la ville et l'heure

- distance encodée 23h ↔ 0h : **0,261**
- distance encodée 23h ↔ 20h : **0,765**
- formes avant nettoyage : 29 — après (`changed→changing`, `round→circle`) : 27
- villes distinctes : 22 018 — dont 14 177 qui n'apparaissent qu'une seule fois
- règle ville : chaque ville remplacée par son taux de canulars, appris uniquement sur les 71 258 lignes d'apprentissage (`TargetEncoder`) ; 2 563 villes du test jamais vues à l'entraînement reçoivent le taux moyen global
- largeur du tableau si ville en one-hot naïf : 22 127 colonnes
- largeur réelle (ville target-encodée) : **110 colonnes**
- bilan cumulé (regroupement + chronologie + ville/heure/forme) : recall **59,1 %**, precision **1,4 %**

- [ ] la règle appliquée aux villes, en une phrase
- [ ] pourquoi 23h doit être proche de 0h et pas de 20h — en quoi l'ancien encodage (0 à 23) posait problème
- [ ] le bilan cumulé (59,1 %) est un peu plus bas que la phase 10 (68,2 %) — qu'est-ce que t'en dis au Conseil ? (indice : toutes les améliorations ne font pas forcément monter le même chiffre, et c'est OK tant que c'est honnête)
