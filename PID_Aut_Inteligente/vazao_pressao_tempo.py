import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados coletados
file_path = 'DadosTratados.xlsx'
data = pd.read_excel(file_path)

# Colunas de interesse: vazão (FT_4A) e pressão (PT_4A) no ponto 4A
tempo = data['Timestamp']
vazao = data['FT_4A']
pressao = data['PT_4A']

# Plotar vazão e pressão em subplots separados, com o mesmo eixo de tempo
fig, (ax_vazao, ax_pressao) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax_vazao.plot(tempo, vazao, color='#1B6BB0', linewidth=1.5)
ax_vazao.set_ylabel('Vazão - FT_4A')
ax_vazao.set_title('Vazão ao longo do tempo')
ax_vazao.grid(True, alpha=0.3)

ax_pressao.plot(tempo, pressao, color='#D1495B', linewidth=1.5)
ax_pressao.set_ylabel('Pressão - PT_4A')
ax_pressao.set_title('Pressão ao longo do tempo')
ax_pressao.set_xlabel('Tempo')
ax_pressao.grid(True, alpha=0.3)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()
