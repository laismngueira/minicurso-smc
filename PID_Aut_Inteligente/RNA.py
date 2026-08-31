import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

# Carregar os dados
file_path = r'C:\Users\laisa\Downloads\PID_Aut_Inteligente\DadosTratados.xlsx'
data = pd.read_excel(file_path)

X = data.drop(columns=['Timestamp', 'PT_4A', 'PT_3B','PT_6A', 'FT_2B', 'FT_4B', 'FT_5A', 'CV_4A_READ', 'PT_1A', 'FT_2A'])
y = data['PT_4A']

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Callback para parar treinamento baseado no MAPE
class MAPEStopper(Callback):
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(X_test)
        non_zero_indices = y_test != 0
        y_test_non_zero = y_test[non_zero_indices]
        mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test_non_zero)) * 100
        if mape < 5:
            self.model.stop_training = True
            print(f"\nEpoch {epoch+1}: MAPE abaixo de 5% - parando treinamento. MAPE: {mape:.2f}%")

# Definição do modelo usando a API funcional
inputs = Input(shape=(X_train.shape[1],))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=10, verbose=1, mode='min', restore_best_weights=True)
mape_stopper = MAPEStopper()

# Treinamento do modelo
history = model.fit(X_train, y_train, epochs=350, validation_split=0.2, batch_size=32, verbose=1, callbacks=[mape_stopper])

# Avaliação do modelo
loss = model.evaluate(X_test, y_test)
print(f'MSE: {loss}')

y_pred = model.predict(X_test)
non_zero_indices = y_test != 0
y_test_non_zero = y_test[non_zero_indices]
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test_non_zero)) * 100
print(f'MAPE: {mape}')

# Plot dos resultados
plt.plot(history.history['loss'], label='Trainamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('Treinamento e Validação')
plt.legend()
plt.show()

# Salvar o modelo
model.save('Modelo_AI_v1.h5')
