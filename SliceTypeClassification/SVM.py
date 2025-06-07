import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

# Lecture du dataset
data = pd.read_csv('train_dataset.csv')

# Suppression de la feature AR/VR/Gaming
data = data.drop(columns=['AR/VR/Gaming'])

# Features utilisées
features = ['Packet delay', 'IoT', 'LTE/5G', 'GBR', 'Non-GBR',
            'Healthcare', 'Industry 4.0', 'Public Safety', 'Smart Transportation', 'Smartphone']

X = data[features]
y = data['slice Type']

# Modèle SVM
clf = SVC()

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entraînement
clf.fit(X_train, y_train)

# Prédiction
y_pred = clf.predict(X_test)

# Affichage des métriques
print("\nClassifier: SVM")
print(f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
print("Confusion Matrix:")
print(conf_matrix)

# Visualisation matrice de confusion
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Rapport de classification
print("Classification Report:")
print(classification_report(y_test, y_pred, labels=[0, 1, 2, 3]))
