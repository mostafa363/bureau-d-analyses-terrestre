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
