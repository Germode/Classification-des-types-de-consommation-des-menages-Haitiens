# Classification des types de consommation des ménages Haitien
  **Préparé et présenté par :** Saint Germain Emode / Darlens Damisca </br>
  **Email:** germodee12@gmail.com / bdamisca96@gmail.com

  
![Electricite d'Haiti](https://github.com/Germode/Classification_des_types_de_consommation/blob/main/Images/electricite.png)

#### Un projet de Data Science visant à classer les ménages selon leur niveau de consommation énergétique  (faible, moyenne ou forte)* à partir des données de compteurs électriques.

## 📖 Contexte

En Haïti, l’accès à l’électricité demeure irrégulier et inégal, notamment entre zones rurales et urbaines.  
Les ménages présentent des profils de consommation très variés, rendant difficile la planification énergétique nationale.

Grâce à l’exploitation de **données issues de compteurs intelligents (smart meters)**, ce projet propose une approche basée sur **l’intelligence artificielle** pour **analyser, comprendre et classer** les comportements de consommation des foyers haïtiens.

## 🎯 Objectif du projet
Développer un **modèle d’apprentissage automatique (Machine Learning)** capable de **classifier automatiquement les ménages haïtiens** selon leur **niveau de consommation énergétique moyenne (en kW)**.

### 🧩 Objectifs spécifiques
- Analyser les profils de consommation à partir des données collectées (ampérage, transactions, zones).
- Extraire et créer des **caractéristiques (features)** pertinentes.
- Gérer le **déséquilibre des classes** dans les données.
 
 ## Description du jeu de données
Dans le cadre de ce projet, nous avons eu accès à un jeu de données privé fourni par la société Sigora, contenant des informations détaillées sur les compteurs des clients situés dans plusieurs communes du Nord-Ouest d’Haïti, telles que Môle Saint-Nicolas, Jean Rabel, Bombardopolis et Môle Rouge.

Ce jeu de données couvre la période de janvier 2023 à septembre 2025 et comprend notamment :
- La consommation quotidienne de chaque client,
- Les transactions financières effectuées sur leurs compteurs.

## Compréhension des Données

Le jeu de données comprend **2 716 foyers** avec des relevés complets de compteurs intelligents incluant :

- **Données temporelles** : Enregistrements de consommation horodatés sur plusieurs mois
- **Métriques de consommation** : Relevés d'ampérage, agrégations quotidiennes, coûts énergétiques
- **Paramètres techniques** : Capacité de tension, force du signal WiFi, version du compteur
- **Métadonnées des ménages** : Zone géographique, type de maison, nombre de résidents
- **Historique de transactions** : Enregistrements de paiements (dépôts et retraits)
  
## Aplatir la structure JSON en DataFrame (une ligne = une mesure)
convertir la structure imbriquée (consommations pour chaque foyer) en un tableau plat exploitable.
Shape du DataFrame features : (2716, 24)
Lignes dans le DataFrame aplati : 6644210

  ![Data](https://github.com/Germode/Classification-des-types-de-consommation-des-menages-Haitiens/blob/main/Images/11.png)


   ## les données JSON brutes en indicateurs numériques exploitables
![Data](https://github.com/Germode/Classification-des-types-de-consommation-des-menages-Haitiens/blob/main/Images/Data.png)

## Analyse exploratoire (EDA) : statistiques et visualisations
Inspecter les distributions, détecter outliers, comprendre relations simples.

![Data](https://github.com/Germode/Classification-des-types-de-consommation-des-menages-Haitiens/blob/main/Images/t%C3%A9l%C3%A9chargement%20(22).png)

![Data](https://github.com/Germode/Classification-des-types-de-consommation-des-menages-Haitiens/blob/main/Images/t%C3%A9l%C3%A9chargement%20(23).png)

![Data](https://github.com/Germode/Classification-des-types-de-consommation-des-menages-Haitiens/blob/main/Images/t%C3%A9l%C3%A9chargement%20(24).png)

### Caractéristiques Principales Créées

| Caractéristique | Description | Importance |
|-----------------|-------------|------------|
| `avg_amperage_per_day` | Consommation moyenne d'ampérage quotidien | **Maximale** - Prédicteur principal |
| `avg_depense_per_day` | Dépense énergétique quotidienne moyenne | **Élevée** - Indicateur de coût |
| `ratio_depense_amperage` | Ratio d'efficacité des coûts | **Moyenne** - Modèle d'utilisation |
| `jours_observed` | Nombre de jours d'observation | **Moyenne** - Fiabilité des données |
| `nombre_personnes` | Taille du ménage | **Faible** - Facteur démographique |

La variable cible segmente les foyers en trois classes équilibrées :
- **Petit** (Petits consommateurs) : ≤33e percentile
- **Moyen** (Consommateurs moyens) : 33e-66e percentile  
- **Grand** (Grands consommateurs) : ≥66e percentile

## Modélisation et Évaluation

### Modèles Comparés

Trois algorithmes de classification ont été évalués avec optimisation des hyperparamètres :

1. **Random Forest Classifier** (n_estimators=200, max_depth=10)
2. **Régression Logistique** (C=100, penalty='l1', solver='liblinear')
3. **XGBoost Classifier** (learning_rate=0.05, max_depth=6, n_estimators=300)

Tous les modèles utilisent `class_weight='balanced'` pour gérer les légers déséquilibres de classes.

### Résultats de Performance

**Métriques Finales sur l'Ensemble de Test :**

| Modèle | Accuracy | Balanced Accuracy | F1-Score | Précision | Rappel |
|--------|----------|-------------------|----------|-----------|--------|
| **XGBoost (Meilleur)** | 99,63% | 99,63% | 99,63% | 99,63% | 99,63% |
| Random Forest | 99,82% | 99,82% | 99,81% | 99,82% | 99,82% |
| Régression Logistique | 91,54% | 91,55% | 91,56% | 91,95% | 91,54% |

![Metriques results](https://github.com/Germode/Classification-des-types-de-consommation-des-menages-Haitiens/blob/main/Images/download%20(3).png)

### Robustesse du Modèle

La validation croisée à 5 plis confirme la stabilité :
- XGBoost : 0,9982 ± 0,0017 F1-Score
- Random Forest : 0,9995 ± 0,0009 F1-Score  
- Régression Logistique : 0,9885 ± 0,0063 F1-Score

### Analyse des Erreurs

Sur 544 échantillons de test :
- **XGBoost** : 1 erreur de classification (taux d'erreur 0,18%)
- **Random Forest** : 1 erreur de classification (taux d'erreur 0,18%)
- **Régression Logistique** : 4 erreurs de classification (taux d'erreur 0,74%)

Les erreurs se produisent principalement entre classes adjacentes (ex: "moyen" vs "grand"), indiquant que la nature ordinale de la consommation est bien capturée.

## Conclusion

### Recommandations

**Pour les Fournisseurs d'Énergie :**
1. **Implémenter la segmentation automatique des clients** en utilisant le modèle XGBoost déployé pour classifier les 2 716 foyers avec une fiabilité de 99,8%
2. **Concevoir des stratégies de tarification échelonnée** adaptées à chaque segment de consommation, améliorant les revenus tout en maintenant l'accessibilité
3. **Cibler les programmes d'efficacité** sur les "grands" consommateurs (896 foyers identifiés) pour réduire la demande de pointe d'environ 15-20%
4. **Surveiller les modèles de consommation** mensuellement et réentraîner le modèle trimestriellement pour maintenir la précision à mesure que l'utilisation évolue

**Pour l'Utilisation Opérationnelle :**
- Déployer le modèle via API pour la classification en temps réel des nouvelles installations de compteurs
- Générer des alertes automatisées lorsque les foyers transitent entre les paliers de consommation
- Intégrer les prédictions dans les systèmes de facturation pour l'application dynamique des tarifs

### Déploiement du Modèle

Le modèle XGBoost entraîné, le scaler et les encodeurs sont sauvegardés dans `/sigor_model_artifacts/` et prêts pour le déploiement en production. Le pipeline traite les données brutes du compteur via l'ingénierie de caractéristiques, la mise à l'échelle et la classification en millisecondes par foyer.

### Limitations et Travaux Futurs

- **Fraîcheur des données** : Modèle entraîné sur des données historiques ; la performance peut diminuer avec les changements saisonniers
- **Expansion des caractéristiques** : Incorporer les données météorologiques et les modèles horaires pourrait améliorer la granularité
- **Explicabilité** : Implémenter les valeurs SHAP pour une prise de décision transparente dans les communications clients
- **Scalabilité** : Le modèle actuel gère 2 716 foyers ; évaluer le calcul distribué pour un déploiement national (50 000+ foyers)

## Navigation du Répertoire

```
Classification-des-types-de-consommation-des-menages-Haitiens/
│
├── README.md                          # Ce fichier
├── Notebook Final.ipynb                    # Notebook d'analyse complet (prêt pour production)
├── presentation.pdf                   # Diapositives du résumé exécutif
│
├── Donnees/
│   └── sigorahaitimetedatas.json     # Données brutes des compteurs intelligents (2 716 foyers)
│
├── models/
│   ├── best_model.joblib   # Classificateur XGBoost entraîné
│   ├── scaler.joblib                     # StandardScaler pour les caractéristiques
│   ├── label_encoder.joblib              # Encodeur ordinal pour les classes
│   └── performance_metrics.json          # Métriques d'évaluation du modèle
│
└── results/
    └── final_results_YYYYMMDD_HHMM.csv   # Prédictions avec probabilités
```

### Instructions de Reproduction

**Prérequis :**
- Python 3.8+
- Bibliothèques : pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

**Configuration :**
```bash
# Cloner le répertoire
git clone https://github.com/votreusername/Classification-des-types-de-consommation-des-menages-Haitiens.git
cd Classification-des-types-de-consommation-des-menages-Haitiens

# Installer les dépendances
pip install -r requirements.txt

# Exécuter le notebook d'analyse
jupyter notebook "Notebook Final.ipynb"
```

**Pour Google Colab :**
1. Télécharger `sigorahaitimetedatas.json` sur Google Drive
2. Ouvrir ` Notebook Final.ipynb` dans Colab
3. Mettre à jour `DATA_PATH` dans la cellule 3 vers votre emplacement Drive
4. Exécuter toutes les cellules séquentiellement (Runtime > Run all)

**Temps d'exécution attendu :** 15-20 minutes sur matériel standard (inclut l'optimisation GridSearchCV)

---

**Contact du Projet :** Darlens DAMISCA | Emode ST GERMAIN
**Dernière Mise à Jour :** Octobre 2025  
**Licence :** MIT
