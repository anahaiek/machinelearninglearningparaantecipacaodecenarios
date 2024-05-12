# Instale as bibliotecas necessárias
!pip install pandas scikit-learn

# Importe as bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

from google.colab import files
uploaded = files.upload()

# Carregue os dados
data = pd.read_csv("wine_dataset.csv")

# Converte a coluna 'style' em valores numéricos
data['style'] = data['style'].map({'red': 0, 'white': 1})

# Divida os dados em recursos (X) e rótulos (y)
X = data.drop('style', axis=1)  # Recursos
y = data['style']  # Rótulos

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicialize o classificador Gaussian Naive Bayes
model = GaussianNB()

# Treine o modelo
model.fit(X_train, y_train)

# Faça previsões nos dados de teste
y_pred = model.predict(X_test)

# Calcule a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print("A precisão do modelo Gaussian Naive Bayes é:", accuracy)

# Exiba as previsões e os rótulos verdadeiros lado a lado
results = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
print(results)

# Exibe o relatório de classificação
print(classification_report(y_test, y_pred))