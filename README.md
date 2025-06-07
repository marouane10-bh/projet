# Pipeline de Prédiction QoS et Classification des Slices dans les Réseaux Cellulaires

## Présentation du Projet

Ce projet propose une chaîne complète de traitements basée sur le Machine Learning pour analyser les données de qualité de service (QoS) des réseaux cellulaires, prédire la génération réseau (2G/3G, 4G, 5G) et classifier les types de slices réseau. L’objectif est d’apporter des insights exploitables pour l’optimisation future des réseaux, notamment dans le contexte émergent des réseaux 6G.

---

## Motivation

Avec l’évolution rapide des technologies mobiles et la généralisation du slicing réseau en 5G et au-delà, la capacité à prédire précisément la génération du réseau et le comportement des slices à partir des indicateurs QoS est un enjeu majeur. Cela facilite une allocation proactive des ressources, améliore l’expérience utilisateur et soutient la vision des réseaux auto-organisés (SON).

---
## Jeux de données utilisées

Plusieurs jeux de données ont été exploités dans ce projet, chacun avec ses caractéristiques et son utilité spécifique :

- **2023_QoS_Metropole_data_habitations.csv** : C’est la véritable donnée brute principale du projet. Elle contient des mesures de QoS collectées en milieu métropolitain. Toutefois, cette base est relativement pauvre en variables explicatives pour la classification des slices réseau.

- **Data.csv** : Jeu de données prétraité issu du fichier précédent. Il a subi un nettoyage approfondi et une extraction de nouvelles caractéristiques afin d'améliorer sa qualité et sa pertinence pour la modélisation.

- **train_dataset.csv** : Une autre base, plus cohérente et riche en informations pertinentes pour la classification des types de slices. Ce jeu de données a été privilégié pour l'entraînement des modèles supervisés en raison de sa meilleure structuration.

- **6Gnetwork** : La premiere dataset exploré dans le cadre du projet, qui présentait une bonne clarté des données et un choix pertinent de colonnes normalisées. Cependant, malgré ces qualités apparentes, ce jeu de données s’est avéré peu utile. Les essais réalisés, notamment sur des données simulées, n’ont pas abouti à des résultats exploitables, probablement en raison d’un écart important avec la réalité observée dans les réseaux?

---

## Jeux de corrélation 
Après avoir généré les rapports de classification initiaux pour chaque modèle, j’ai expérimenté en augmentant le seuil du facteur de corrélation à 0.3 afin d’éliminer une  colonnes peu corrélée que les autres. Cette sélection a permis de simplifier le jeu de données tout en améliorant la qualité des prédictions.

Les résultats ont montré que pour les modèles K-Nearest Neighbors, Random Forest et Logistic Regression, cette suppression de variable  a conduit à une amélioration notable, avec un score F1 pondéré et une accuracy atteignant la valeur maximale de 1, indiquant une classification parfaite sur le jeu de test.

Cependant, le modèle SVM a présenté une légère diminution de ses performances, ce qui suggère que ce classifieur tirait bénéfice d’un ensemble de variables plus complet, et que la suppression  a pu réduire son efficacité.

---


## Contributions principales

- Nous avons comparé deux approches pour la prédiction de la génération réseau :  
  - Le clustering KMeans, qui est simple mais ne prend pas en compte la dimension temporelle ni les dépendances entre variables.  
  - Un modèle hybride combinant Hidden Markov Models (HMM), Réseaux Bayésiens (BN) et ARIMA, qui intègre explicitement la dépendance temporelle et les relations probabilistes complexes entre les métriques, produisant des clusters plus cohérents et interprétables.

- Pour la classification des slices réseau, nous avons testé trois modèles supervisés :  
  - Random Forest, qui offre la meilleure précision et une bonne robustesse.  
  - SVM, avec des performances modérées.  
  - Régression Logistique, moins performant que les précédents.

- Proposition d’un pipeline évolutif et modulaire capable de servir de base pour des systèmes en temps réel visant l’orchestration intelligente et l’optimisation des KPI réseau.

---

## Architecture du Pipeline

1. **Acquisition et Prétraitement des Données**  
   Collecte des métriques QoS et des paramètres signal, correction des données et extraction des features temporelles (sans normalisation à ce stade).

2. **Analyse Exploratoire des Données (EDA)**  
   Visualisations et analyses statistiques pour comprendre les distributions et corrélations des variables.

3. **Module de Prédiction de la Génération**  
   Présentation des deux modèles : KMeans et modèle hybride (HMM + BN + ARIMA), suivie d’une comparaison basée sur des critères de cohésion, stabilité et prise en compte temporelle.

4. **Module de Classification des Slices**  
   Évaluation comparative des performances des modèles Random Forest, Régression Logistique et SVM à partir de métriques classiques (accuracy, precision, recall, F1-score).

5. **Perspectives d’Optimisation**  
   Sur la base des résultats obtenus avec le meilleur classifieur de slices (Random Forest), l’idée est d’utiliser sa sortie comme feature additionnelle dans un modèle non supervisé. Ce modèle viserait à générer des clusters d’optimisation opérationnelle (par exemple : ajustement dynamique de la bande passante, politique de handover, contrôle de la puissance d’émission). Cependant, une contrainte temporelle importante a freiné cette exploration, liée au fait que la dataset 6Gnetwork, malgré une bonne clarté des données et un choix pertinent des colonnes avec valeurs normalisées, s’est révélée peu utile pour cet objectif. En effet, un premier essai réalisé sur des données simulées issues de cette base, très éloignées de la réalité, n’a pas donné de résultats exploitables.

---

## Technologies et Bibliothèques Utilisées

- Python  
- scikit-learn  
- hmmlearn  
- pgmpy (Réseaux Bayésiens)  
- statsmodels (ARIMA)  
- pandas, numpy, matplotlib, seaborn  

---


