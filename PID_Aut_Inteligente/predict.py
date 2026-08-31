import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Carregar o modelo treinado
model = load_model('Modelo_AI_v1.h5')

# Novo input para fazer a previsão
new_input = np.array([30, 5.377099, 3.30916, 3.199237, 4.277863, 3.606107, 
                      0.064122, 3.471756, 1.206405, 2.47063, 0.365964, 
                      2.220835, 0.352547])

# Carregar os dados novamente para aplicar o mesmo escalonamento
file_path = r'C:\Users\55839\Desktop\AI\DadosTratados.xlsx'
data = pd.read_excel(file_path)

# Selecionar as colunas utilizadas no treinamento
X = data.drop(columns=['Timestamp', 'PT_4A', 'PT_3B','PT_6A', 'FT_2B', 'FT_4B', 'FT_5A', 'CV_4A_READ', 'PT_1A', 'FT_2A'])

# Aplicar o mesmo escalonamento que foi utilizado no treinamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Escalonar o novo input
new_input_scaled = scaler.transform([new_input])

# Fazer a previsão
prediction = model.predict(new_input_scaled)

# Mostrar o resultado da previsão
print(f'Previsão para o novo input: {prediction[0][0]}')
