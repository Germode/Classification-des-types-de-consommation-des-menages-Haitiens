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
  ![Data](https://github.com/Germode/Classification-des-types-de-consommation-des-menages-Haitiens/blob/main/Images/Data.png)

