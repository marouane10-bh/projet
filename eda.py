import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Load cleaned dataset ----
df = pd.read_csv("Data.csv", sep=';', encoding='utf-8')

# ---- Keep only first 19 columns for EDA ----
eda_df = df.iloc[:, :19]

# ---- Convert date and time columns ----
eda_df['date'] = pd.to_datetime(eda_df['date'], errors='coerce')
eda_df['heure'] = pd.to_datetime(eda_df['heure'], format='%H:%M:%S', errors='coerce').dt.time

# ---- Convert 'temps_en_secondes' from string to float ----
if 'temps_en_secondes' in eda_df.columns:
    eda_df['temps_en_secondes'] = eda_df['temps_en_secondes'].str.replace(',', '.', regex=False).astype(float)

# ---- Select only numeric columns for visualizations ----
numeric_cols = eda_df.select_dtypes(include=['number']).columns

# ---- Pairplot (Zoom réduit) ----
sns.pairplot(eda_df[numeric_cols], height=1.5)
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.tight_layout()
plt.show()

# ---- Boxplot ----
plt.figure(figsize=(12, 6))
sns.boxplot(data=eda_df[numeric_cols])
plt.title("Boxplot of Numeric Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---- Correlation Matrix ----
corr = eda_df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix of Numeric Features")
plt.show()
