# DrainLab 360 — Projet universitaire ENSPY (Génie civil N4)

## Objet
Application Streamlit de pré-étude d'assainissement pluvial : description d'un petit bassin, pluies horaires de réanalyse ou importées, méthode rationnelle, capacités de sections par Manning, comparaison de variantes et Monte-Carlo par blocs. Utilise une démonstration fictive sans accès Internet. Les résultats sont **pédagogiques** et **ne constituent pas un dimensionnement réglementaire**.

## Structure
Seuls `app.py`, `requirements.txt` et `README.md` appartiennent au dossier livré. Le nom **requirements.txt** est utilisé directement pour assurer l'installation automatique sur Streamlit Community Cloud (contrairement à `requirement.text`, qui doit être renommé).

## Prérequis et installation Windows
Python **3.11** recommandé, VS Code (extension Python), accès Internet uniquement pour télécharger les bibliothèques et consulter les données externes.

1. Extraire l'archive, puis dans VS Code : **Fichier > Ouvrir un dossier > DrainLab360**.
2. Ouvrir **Terminal > Nouveau terminal**.
3. (Conseillé) créer un environnement virtuel :

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

4. Installer :

```powershell
python -m pip install -r requirements.txt
```

5. Exécuter :

```powershell
python -m streamlit run app.py
```

6. Ouvrir l'adresse locale affichée dans le terminal (en général `http://localhost:8501`). Ne pas taper `python app.py` : l'interface est une application Streamlit.

Si PowerShell bloque l'activation, utiliser directement `.venv\Scripts\python.exe -m pip install -r requirements.txt` et `.venv\Scripts\python.exe -m streamlit run app.py`.

## Navigation et tutoriel de soutenance
- **📊 Vue générale** : indicateurs Q, pluies, variantes, calendrier, PDF et JSON.
- **🗺️ Mon bassin** : créer et modifier des bassins, recherche facultative Open-Meteo Geocoding, position cartographique, altitude indicative SRTM (pas la pente de l'ouvrage).
- **🌧️ Pluviométrie** : récupérer des données horaires de réanalyse ERA5, ou importer CSV/Excel avec `date_heure` et `pluie_mm`. La date/heure interne est **locale Africa/Douala**, au **début** de l'heure d'accumulation. Les heures Open-Meteo représentent le cumul de l'heure **précédente** : le code décale leurs libellés d'une heure vers le passé et récupère le jour supplémentaire nécessaire à la dernière journée. Pour un CSV importé, fournir directement des libellés de début d'heure. Aucun manque n'est remplacé par une valeur inventée. Afficher les données brutes avant l'analyse.
- **📐 Dimensionnement** : méthode rationnelle, capacité Manning de sections rectangulaires et trapézoïdales, comparaison et exports.
- **🎲 Probabilités** : fréquence empirique des dépassements et simulations par blocs de journées historiques. L'événement simulé est `Q_rationnel >= Q_Manning` au moins une fois sur la fenêtre. Pas de date d'événement au-delà de l'horizon.
- **🎓 Comprendre et tester** : exemples et valeurs exactes, fonctions vérifiables à la calculatrice.

**Première démonstration hors connexion** : dans la rubrique « Comprendre et tester », cliquer « Charger l'exemple fictif ». Il ajoute une étude fictive clairement identifiée sans toucher aux autres. Retournez dans « Vue générale », puis « Dimensionnement », « Pluviométrie » et « Probabilités ». « Réinitialiser l'exemple fictif » supprime uniquement les études fictives.

### Résultats de référence (fictifs)
Dix jours de précipitations (mm/jour) : `0, 4, 12, 0, 8, 25, 1, 0, 16, 3` ; moyenne **6,9 mm/j** ; **3/10 = 30 %** des jours ont au moins 10 mm.

Bassin fictif : `A=2 ha`, `C=0,65`, `I=30 mm/h`, `durée=1 h`.

- `Q = C × I × A_ha / 360 = 0,108333 m³/s`.
- Caniveau rectangulaire `b=0,30 m`, `H=0,30 m`, `réserve=0 %`, `pente=0,5 %`, `n=0,015` : `A_mouillée=0,09 m²`, `P=0,90 m`, `R=0,10 m`, `Q_Manning≈0,091405 m³/s` ; ratio `Q_rational/Q_Manning≈1,185`.

Une troisième section plus compacte (20 × 20 cm) est également disponible pour illustrer les dépassements dans les simulations. Les dimensions, coefficients et intensités de cet exemple sont de **simples hypothèses pédagogiques**, pas des données mesurées ni des valeurs réglementaires camerounaises.

## Méthodes, unités et limites
### Méthode rationnelle
`Q(m³/s) = C × I(mm/h) × A(ha) / 360`. Convertit mm/h × ha en m³/s. Les valeurs `I` doivent correspondre à une durée assimilable au temps de concentration `tc`, validé par une méthode appropriée ; l'application ne détermine pas `tc`. La méthode est simplificatrice et adaptée principalement aux petits bassins sous hypothèses convenables. Ne pas extrapoler à des bassins complexes.

### Manning
`Q = (1/n) A R^(2/3) sqrt(S)`, `R=A/P`. Section rectangulaire : `A=b*y`, `P=b+2y`. Section trapézoïdale à talus `z` (H/V) : `A=y(b+z*y)`, `P=b+2y sqrt(1+z²)`. Pente `S` en **m/m** : la saisie `0,5 %` signifie `0,005`. La réserve de hauteur réduit `y`. Formule en régime uniforme et rugosité supposée constante. Ne couvre pas remous, pertes singulières, colmatage, chute, vitesse limite, stabilité, érosion, surcharge du réseau ou ouvrages de transition.

### Pluviométrie et probabilités
Source : https://open-meteo.com/en/docs/historical-weather-api ; coordonnées via https://open-meteo.com/en/docs/geocoding-api ; altitude via https://www.opentopodata.org/api/ . Les pluies horaires sont un **produit de réanalyse**, potentiellement peu représentatif d'orages convectifs ponctuels. Un cumul glissant de `d` heures (d entier 1–6 h) est converti en intensité moyenne mm/h. Les maximales sont affectées à la date locale de fin de fenêtre. Pour les fréquences journalières, seuls les jours avec **24 observations valides** sont retenus ; les valeurs manquantes ne sont pas imputées. L'échantillon de maxima journaliers n'est pas une série indépendante d'événements pluvieux ni une courbe IDF. **Aucune période de retour de dimensionnement ne peut être garantie** avec la seule série.

La simulation Monte-Carlo échantillonne avec remplacement des blocs contigus de maxima journaliers complets, potentiellement chevauchants. Elle concatène les blocs, puis tronque à la durée demandée. Le rééchantillonnage des journées d'intensité conserve certaines dépendances locales mais ignore saisons et dépendances longues. Les variables C et n sont tirées de lois normales tronquées physiquement, **indépendantes** des pluies et entre elles, avec coefficient de variation configurable. Dans un scénario, C et n sont constants sur l'horizon. Les scénarios sans dépassement sont comptés. Intervalle ±95 % ≈ `1,96 sqrt[p(1-p)/N]` : uniquement **erreur de simulation**, et non l'incertitude totale. Quantile 90 % du nombre de jours de dépassement calculé par quantile discret conservateur `higher`.

**Avant une étude de terrain réelle**, obtenir les courbes IDF locales validées, des levés topographiques, les caractéristiques de sol, le bassin exact, les débits amont/aval, la vérification des vitesses et des normes applicables. Ni la carte ni une altitude ponctuelle ne remplacent ces données.

## Formats et sauvegardes
- Import CSV/Excel : colonne `date_heure` (ISO 8601, une heure entière) et `pluie_mm` (cumul positif de l'heure) ; 100 000 lignes maximum, pas d'horodatages dupliqués. Les blancs deviennent valeurs manquantes, jamais des zéros fictifs.
- Export CSV (pluies, bilan, variantes), Excel multi-feuilles, rapport PDF avec police Vera intégrée (accents), JSON multi-projets.
- Stockage : `st.session_state` **par session de navigateur**. Aucun enregistrement sur disque serveur ; l'utilisateur doit **télécharger une sauvegarde JSON**. Une nouvelle session ou un redémarrage peut perdre les données non sauvegardées. Sans compte, ne pas y entrer de données confidentielles. Les requêtes météo transmettent les coordonnées aux services tiers selon leurs propres conditions.
- L'import de pluies réelles sur un exemple fictif crée une nouvelle étude séparée, sans remplacer la démonstration. L'import JSON valide les champs et remplace les études de la **session actuelle** seulement, pas celles d'autres utilisateurs.

## GitHub et Streamlit Community Cloud
1. Créer un dépôt GitHub privé ou public. Si public, ne pas déposer de données confidentielles.
2. À la racine du dépôt, ajouter les trois fichiers `app.py`, `requirements.txt`, `README.md`.
3. Dans le terminal du dossier projet :

```bash
git init
git add app.py requirements.txt README.md
git commit -m "Initialiser DrainLab 360"
git branch -M main
git remote add origin https://github.com/UTILISATEUR/DEPOT.git
git remote -v
git branch --show-current
git push -u origin main
```

Remplacer le remote d'exemple par l'URL **réelle** fournie par GitHub. En cas de remote existant, vérifier `git remote -v` avant d'en ajouter un.
4. Ouvrir https://share.streamlit.io/ , créer une application à partir du dépôt et choisir la branche `main` et le fichier `app.py`.
5. Streamlit installe automatiquement `requirements.txt` ; choisir Python 3.11 si disponible.
6. En cas d'échec, ouvrir « Manage app » → logs pour lire la cause. Ne jamais supposer que le déploiement est terminé sans l'avoir vérifié.

Mises à jour :

```bash
git status
git add .
git commit -m "Décrire la modification"
git push
```

## Dépannage
- `ModuleNotFoundError: plotly` : vérifier l'orthographe **requirements.txt**, sa position et les logs d'installation sur Streamlit Cloud ; redémarrer si nécessaire.
- `streamlit: command not found` : exécuter `python -m streamlit run app.py` avec le même Python que celui de l'installation.
- Erreur `requests`: tester la connexion, la localisation et la période ; importer un CSV hors connexion.
- `py` ou `python` non reconnu : installer Python 3.11 en cochant « Add Python to PATH », rouvrir VS Code.
- Dans VS Code, « Python: Select Interpreter » doit correspondre à `.venv` si l'environnement est activé.
- Les simulations ne changent pas après modification des paramètres : cliquer de nouveau « Lancer la simulation ».

## Vérifications
Tests des calculs, des validations, des imports, des séries manquantes, de la reproductibilité Monte-Carlo, des JSON, Excel et PDF effectués séparément du dossier livré. L'ouverture dans un navigateur Streamlit ainsi que la connexion live aux API doivent être revérifiées sur l'ordinateur ou le Cloud cible si l'environnement de livraison ne permet pas d'installer Streamlit ou d'accéder aux API.
