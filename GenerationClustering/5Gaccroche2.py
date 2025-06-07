import numpy as np
import pandas as pd
from hmmlearn import hmm
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler

# Exemple de données factices
# Remplace par ton vrai df avec les bonnes colonnes et prétraitements
df = pd.read_csv("Data.csv", sep=';', encoding='utf-8', low_memory=False)

# Supposons que les features numériques sont ['rsrp', 'longitude', 'latitude', ...]
features = ['rsrp', 'longitude', 'latitude', 'page_chargée_moins_10s', 
            'page_chargée_moins_5s', 'temps_en_secondes']  # sans colonnes INSEE, etc.

df_clust = df[features].dropna()
X = df_clust.values

# --- Normalisation des données ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Étape 1 : Entraîner un HMM ---
model_hmm = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=200)
model_hmm.fit(X_scaled)
hidden_states = model_hmm.predict(X_scaled)

# --- Étape 2 : Entraîner un BN par état ---
bns = {}
for state in range(3):
    data_state = pd.DataFrame(X[hidden_states == state], columns=features)
    # Attention : ici on crée un BN vide pour exemple, en vrai il faut définir la structure
    bn = BayesianNetwork()  
    # Exemple : on suppose pas de structure, on peut faire apprentissage structurel si besoin
    # Ici on l'entraîne juste avec MaximumLikelihoodEstimator sur un graphe vide
    from pgmpy.estimators import MaximumLikelihoodEstimator
    bn.fit(data_state, estimator=MaximumLikelihoodEstimator)
    bns[state] = bn

# --- Étape 3 : Entraîner un ARIMA par état sur la variable temporelle ---
# Supposons qu'on utilise 'rsrp' comme série temporelle pour ARIMA
arima_models = {}
for state in range(3):
    rsrp_series = pd.Series(X[hidden_states == state, 0])  # colonne rsrp (non normalisée)
    if len(rsrp_series) > 10:  # ARIMA nécessite assez de points
        arima_model = ARIMA(rsrp_series, order=(1,0,0)).fit()
        arima_models[state] = arima_model
    else:
        arima_models[state] = None

# --- Fonction pour combiner probas d’émission (simplifiée) ---
def combine_emission_prob(x, state, bn_model, arima_model):
    # BN prob simplifiée : on prend la log prob de chaque feature conditionnelle
    # Ici approximation arbitraire car pas d'inférence directe sans structure
    bn_log_prob = -np.sum(np.abs(x))  # simple proxy

    if arima_model is not None:
        pred = arima_model.predict(start=0, end=0)
        arima_error = np.abs(pred[0] - x[0])  # x[0] = rsrp
        arima_log_prob = -arima_error
    else:
        arima_log_prob = 0

    return bn_log_prob + arima_log_prob

# --- Étape 4 : Boucle itérative de raffinement (simplifiée) ---
for iteration in range(3):
    new_states = []
    for i, x in enumerate(X):
        scores = []
        for state in range(3):
            score = combine_emission_prob(x, state, bns[state], arima_models[state])
            scores.append(score)
        new_states.append(np.argmax(scores))
    hidden_states = np.array(new_states)
    print(f"Iteration {iteration+1}, cluster counts: {np.bincount(hidden_states)}")

# Affichage final
print("Final clustering states distribution:", np.bincount(hidden_states))

# --- Visualisation ---
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
scatter = plt.scatter(X[:, 1], X[:, 2], c=hidden_states, cmap='viridis', alpha=0.6)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Visualisation des clusters après HMM + BN + ARIMA')
plt.colorbar(scatter, label='Cluster')
plt.show()
