import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Carregar os dados
file_path = r'C:\Users\laisa\Downloads\PID_Aut_Inteligente\DadosTratados.xlsx'
data = pd.read_excel(file_path)

# Separar X e y
X = data.drop(columns=['Timestamp', 'PT_4A', 'PT_3B','PT_6A', 'FT_2B', 'FT_4B', 'FT_5A', 'CV_4A_READ', 'PT_1A', 'FT_2A'])
y = data['PT_4A']

# Dividir os dados em conjunto de treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Carregar o modelo treinado
model = load_model('Modelo_AI_v1.h5')

# Fazer as previsões para o conjunto de teste
y_pred = model.predict(X_test).flatten()

# Ordenar os valores reais e as previsões com base nos valores reais
sorted_indices = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# Plotar o gráfico de valores reais vs previsões (ordenados)
plt.figure(figsize=(10,6))
plt.plot(y_test_sorted, label='Real Values', color='blue', linestyle='--')
plt.plot(y_pred_sorted, label='Predicted Values', color='red')
plt.title('Prediction vs Real Value (Sorted by Real Values)')
plt.xlabel('Sample Index (Sorted)')
plt.ylabel('PT_4A Value')
plt.legend()
plt.show()