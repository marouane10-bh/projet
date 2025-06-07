import pandas as pd

# 1. Lecture du fichier CSV
df = pd.read_csv('2023_QoS_Metropole_data_habitations.csv', sep=';', encoding='latin1')

# 2. Suppression des colonnes avec + de 50% de valeurs manquantes
threshold = 0.5  # seuil
df_cleaned = df.loc[:, df.isnull().mean() <= threshold]

# 3. Sauvegarde du fichier nettoyé en UTF-8
df_cleaned.to_csv('Data.csv', index=False, sep=';', encoding='utf-8')

print("✅ Fichier nettoyé sauvegardé sous le nom 'Data.csv'")
