import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("Data.csv", sep=';', encoding='utf-8', low_memory=False)

# Colonnes utilisées, ajout de 'adresse' si nécessaire
features = ['rsrp', 'longitude', 'latitude', 'page_chargée_moins_10s', 
            'page_chargée_moins_5s', 'temps_en_secondes', 'url', 'terminal']

df_clust = df[features].dropna()

cat_features = ['url', 'terminal']
num_features = [f for f in features if f not in cat_features]

# Nettoyage : convertir ',' en '.' et en float pour colonnes numériques
for col in num_features:
    if df_clust[col].dtype == object:
        df_clust[col] = df_clust[col].str.replace(',', '.')
    df_clust[col] = df_clust[col].astype(float)

# Préparation pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

X = preprocessor.fit_transform(df_clust)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
silhouette_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    kmeans = KMeans(n_clusters=3, random_state=42)  # Changement ici : k=3
    kmeans.fit(X_train)
    labels = kmeans.predict(X_test)
    score = silhouette_score(X_test, labels)
    silhouette_scores.append(score)

print("Silhouette scores per fold:", silhouette_scores)
print("Mean Silhouette score: {:.4f}".format(np.mean(silhouette_scores)))

plt.figure(figsize=(8,5))
plt.plot(range(1, 6), silhouette_scores, marker='o')
plt.title('Silhouette Score per Fold (K=3)')
plt.xlabel('Fold')
plt.ylabel('Silhouette Score')
plt.ylim(0, 1)
plt.grid(True)
plt.show()
