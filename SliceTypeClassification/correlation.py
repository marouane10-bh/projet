import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement du jeu de données
df = pd.read_csv('train_dataset.csv')

# Séparation de la variable cible 'slice Type'
slice_type = df['slice Type']

# Suppression de la colonne cible pour la normalisation
df_features = df.drop('slice Type', axis=1)

# Initialisation du normaliseur Min-Max
scaler = MinMaxScaler()

# Application de la normalisation Min-Max sur les features
normalized_features = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns)

# Reconstruction du DataFrame final en réintégrant la colonne cible
normalized_data = pd.concat([normalized_features, slice_type], axis=1)

# Sauvegarde du dataset normalisé dans un fichier CSV
normalized_data.to_csv('train.csv', index=False)
print("Dataset normalisé sauvegardé sous 'train.csv'.")

# Affichage des premières lignes du DataFrame normalisé
print(normalized_data.head())

# Calcul de la matrice de corrélation sur le jeu de données normalisé
correlation_matrix = normalized_data.corr()

# Affichage de la matrice de corrélation
print(correlation_matrix)

# Visualisation de la matrice de corrélation
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True, linewidths=0.5)
plt.title("Matrice de corrélation des variables normalisées")
plt.show()

# Choix de la feature cible pour filtrage
feature_slice = 'slice Type'

# Seuil de corrélation
threshold = 0.3

# Sélection des colonnes dont la corrélation absolue dépasse le seuil, sauf la colonne cible elle-même
highly_correlated_features = {
    col for col in correlation_matrix.columns
    if col != feature_slice and abs(correlation_matrix.loc[feature_slice, col]) > threshold
}

# Affichage des features fortement corrélées
print(f"Features fortement corrélées avec '{feature_slice}' (|corr| > {threshold}) :")
print(highly_correlated_features)
