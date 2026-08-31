import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Definição da Classe Controladora PID
class PID:
    def __init__(self, Kp, Ki, Kd, setpoint):
        """
        Inicializa o controlador PID com os parâmetros fornecidos.

        :param Kp: Ganho Proporcional
        :param Ki: Ganho Integral
        :param Kd: Ganho Derivativo
        :param setpoint: Valor desejado para a saída
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.erro_anterior = 0
        self.integral = 0

    def calcular(self, medida, dt):
        """
        Calcula a ação de controle com base na medida atual.

        :param medida: Valor atual da saída medida
        :param dt: Intervalo de tempo
        :return: Ação de controle
        """
        erro = self.setpoint - medida
        self.integral += erro * dt
        derivada = (erro - self.erro_anterior) / dt if dt > 0 else 0
        self.erro_anterior = erro
        return self.Kp * erro + self.Ki * self.integral + self.Kd * derivada

def main():
    # 1. Carregar o modelo treinado
    modelo_path = 'Modelo_AI_v1.h5'  # Altere para o caminho correto se necessário
    try:
        model = load_model(modelo_path)
        print(f'Modelo carregado com sucesso de {modelo_path}.')
    except Exception as e:
        print(f'Erro ao carregar o modelo: {e}')
        return

    # 2. Carregar os dados para o escalonamento
    file_path = r'DadosTratados.xlsx'  # Altere para o caminho correto
    try:
        data = pd.read_excel(file_path)
        print(f'Dados carregados com sucesso de {file_path}.')
    except Exception as e:
        print(f'Erro ao carregar os dados: {e}')
        return

    # 3. Selecionar as colunas utilizadas no treinamento
    colunas_para_entrada = ['Coluna1', 'Coluna2', 'Coluna3', 'Coluna4', 'Coluna5',
                            'Coluna6', 'Coluna7', 'Coluna8', 'Coluna9', 'Coluna10',
                            'Coluna11', 'Coluna12', 'Coluna13']  # Substitua pelos nomes reais das colunas

    # Ajuste conforme as colunas que você realmente usou no treinamento
    X = data.drop(columns=['Timestamp', 'PT_4A', 'PT_3B', 'PT_6A', 'FT_2B',
                            'FT_4B', 'FT_5A', 'CV_4A_READ', 'PT_1A', 'FT_2A'])

    # Verifique se as colunas selecionadas correspondem às esperadas
    if X.shape[1] != 13:
        print(f'Número de colunas selecionadas para entrada é {X.shape[1]}, esperado 13.')
        return

    # 4. Aplicar o escalonamento que foi utilizado no treinamento
    scaler = StandardScaler()
    scaler.fit(X)  # Ajusta o scaler com os dados de treinamento
    print('Escalonador ajustado com os dados de treinamento.')

    # 5. Definir o setpoint
    setpoint = 5.0  # Defina o valor desejado para a saída
    print(f'Setpoint definido para: {setpoint}')

    # 6. Inicializar o controlador PID
    # Ajuste os valores de Kp, Ki e Kd conforme necessário para o seu sistema
    pid = PID(Kp=1.2, Ki=0.3, Kd=0.1, setpoint=setpoint)
    print('Controlador PID inicializado.')

    # 7. Configurar os parâmetros da simulação
    dt = 1.0  # Intervalo de tempo (em segundos, por exemplo)
    num_steps = 1200  # Número de passos de simulação
    historico_setpoint = []
    historico_saida = []
    historico_controle = []

    # 8. Inicializar a entrada atual
    # Aqui, usamos a primeira amostra dos dados de treinamento como ponto de partida
    entrada_atual = X.iloc[0].values.copy()
    print('Entrada inicial definida a partir dos dados de treinamento.')

    # 9. Executar a simulação em malha fechada
    for passo in range(num_steps):

        if passo == 400 : 
            setpoint = 6.0
            pid = PID(Kp=1.2, Ki=0.3, Kd=0.1, setpoint=setpoint)

        if passo == 800 : 
            setpoint = 4.0
            pid = PID(Kp=1.2, Ki=0.3, Kd=0.1, setpoint=setpoint)

        # Escalonar a entrada atual
        entrada_scaled = scaler.transform([entrada_atual])

        # Fazer a previsão da saída com a RNA
        saida_predita = model.predict(entrada_scaled)[0][0]

        # Registrar histórico
        historico_setpoint.append(setpoint)
        historico_saida.append(saida_predita)

        # Calcular a ação de controle
        controle = pid.calcular(saida_predita, dt)
        historico_controle.append(controle)

        # Atualizar a variável de controle
        
        entrada_atual[0] += controle  # Ajuste conforme a dinâmica do sistema
        
        # Limitar a variável de controle dentro de certos limites
        entrada_atual[0] = np.clip(entrada_atual[0], 0, 100)  # Exemplo de limites

        # (Opcional) Atualizar outras entradas se necessário
        # Por exemplo, adicionar perturbações ou atualizar variáveis externas

        # Debugging: Imprimir informações a cada 10 passos
        if (passo + 1) % 10 == 0:
            print(f'Passo {passo + 1}/{num_steps} - Saída Predita: {saida_predita:.4f}, Controle: {controle:.4f}, Entrada Ajustada: {entrada_atual[0]:.4f}')

    # 10. Plotar os resultados
    plt.figure(figsize=(12, 6))
    plt.plot(historico_setpoint, label='Setpoint', linestyle='--')
    plt.plot(historico_saida, label='Saída Predita', marker='o')
    plt.xlabel('Passo de Tempo')
    plt.ylabel('Valor')
    plt.title('Simulação de Sistema em Malha Fechada')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # (Opcional) Plotar a ação de controle ao longo do tempo
    plt.figure(figsize=(12, 4))
    plt.plot(historico_controle, label='Ação de Controle', color='orange')
    plt.xlabel('Passo de Tempo')
    plt.ylabel('Controle')
    plt.title('Ação de Controle ao Longo do Tempo')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
