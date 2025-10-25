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
- Construire et évaluer un modèle fiable pour prédire la catégorie d’un ménage :
  - **Faible consommation** (< 0.05 kW)
  - **Moyenne consommation** (0.05–0.5 kW)
  - **Forte consommation** (> 0.5 kW)
 
 ## Description du jeu de données
Dans le cadre de ce projet, nous avons eu accès à un jeu de données privé fourni par la société Sigora, contenant des informations détaillées sur les compteurs des clients situés dans plusieurs communes du Nord-Ouest d’Haïti, telles que Môle Saint-Nicolas, Jean Rabel, Bombardopolis et Môle Rouge.

Ce jeu de données couvre la période de janvier 2023 à septembre 2025 et comprend notamment :
- La consommation quotidienne de chaque client,
- Les transactions financières effectuées sur leurs compteurs.
Shape du DataFrame features : (2716, 24)
Lignes dans le DataFrame aplati : 6644210
  ![Data](https://github.com/Germode/Classification-des-types-de-consommation-des-menages-Haitiens/blob/main/Images/Data.png)

 # Analyse exploratoire des données (EDA)

Une analyse exploratoire a été menée pour comprendre la structure du jeu de données.
Elle a permis d’étudier la distribution des variables clés (consommation, ampérage, transactions) et d’identifier les corrélations entre les aspects énergétiques et financiers.
Des visualisations statistiques (histogrammes, scatter plots, heatmaps) ont été utilisées pour détecter les tendances et les valeurs atypiques.

![Visualisation](https://github.com/Germode/Classification-des-types-de-consommation-des-menages-Haitiens/blob/main/Images/visalusation.png)
  ![visalusation2](https://github.com/Germode/Classification-des-types-de-consommation-des-menages-Haitiens/blob/main/Images/visalusation2.png)

## Compréhension des Données

Le jeu de données comprend **2 716 foyers** avec des relevés complets de compteurs intelligents incluant :

- **Données temporelles** : Enregistrements de consommation horodatés sur plusieurs mois
- **Métriques de consommation** : Relevés d'ampérage, agrégations quotidiennes, coûts énergétiques
- **Paramètres techniques** : Capacité de tension, force du signal WiFi, version du compteur
- **Métadonnées des ménages** : Zone géographique, type de maison, nombre de résidents
- **Historique de transactions** : Enregistrements de paiements (dépôts et retraits)

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
| **XGBoost (Meilleur)** | 99,82% | 99,82% | 99,82% | 99,82% | 99,82% |
| Random Forest | 99,82% | 99,82% | 99,82% | 99,82% | 99,82% |
| Régression Logistique | 99,26% | 99,26% | 99,26% | 99,26% | 99,26% |

![Metriques results](https://github.com/Germode/Classification-des-types-de-consommation-des-menages-Haitiens/blob/main/Images/download%20(3).png)
