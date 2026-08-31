import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados coletados
file_path = 'DadosTratados.xlsx'
data = pd.read_excel(file_path)

# A coleta tem pausas reais entre sessões (reconfiguração do inversor, troca de
# dia), que não carregam informação sobre a dinâmica da planta. Para visualizar a
# variação completa da frequência (30 -> 35 -> 40 -> 45 -> 50 Hz) em um único
# gráfico, sem que essas pausas distorçam a escala de tempo, cada sessão contínua
# (ensaio) é reposicionada em um eixo de "tempo experimental" acumulado, com um
# intervalo fixo pequeno entre sessões apenas para deixar a quebra visível.
gap = data['Timestamp'].diff()
ensaio_id = (gap > pd.Timedelta(minutes=10)).cumsum()

intervalo_entre_ensaios_min = 2.0
tempo_experimental = pd.Series(index=data.index, dtype=float)
offset = 0.0
limites_ensaio = []

for eid, idx in data.groupby(ensaio_id).groups.items():
    t0 = data.loc[idx, 'Timestamp'].iloc[0]
    decorrido_min = (data.loc[idx, 'Timestamp'] - t0).dt.total_seconds() / 60.0
    tempo_experimental.loc[idx] = decorrido_min + offset
    limites_ensaio.append(offset)
    offset += decorrido_min.max() + intervalo_entre_ensaios_min

data['tempo_experimental'] = tempo_experimental

freq = data['Inversor']
vazao = data['FT_4A']
pressao = data['PT_4A']

# Transições de frequência (mudanças de patamar) para marcar no gráfico
transicoes = data['tempo_experimental'][freq.diff().fillna(0) != 0]

fig, (ax_freq, ax_vazao, ax_pressao) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

for ax in (ax_freq, ax_vazao, ax_pressao):
    for t in limites_ensaio[1:]:
        ax.axvline(t, color='gray', linestyle='-', linewidth=1.0, alpha=0.6)
    for t in transicoes:
        ax.axvline(t, color='gray', linestyle='--', linewidth=0.7, alpha=0.4)
    ax.grid(True, alpha=0.3)

ax_freq.plot(data['tempo_experimental'], freq, color='#4B4B4B', linewidth=1.5)
ax_freq.set_ylabel('Frequência (Hz)')
ax_freq.set_title('Frequência do inversor, vazão e pressão em função da variação da frequência')

ax_vazao.plot(data['tempo_experimental'], vazao, color='#1B6BB0', linewidth=1.5)
ax_vazao.set_ylabel('Vazão - FT_4A')

ax_pressao.plot(data['tempo_experimental'], pressao, color='#D1495B', linewidth=1.5)
ax_pressao.set_ylabel('Pressão - PT_4A')
ax_pressao.set_xlabel('Tempo experimental acumulado (min) - pausas entre ensaios comprimidas')

# Anotar cada patamar de frequência com o valor em Hz, na primeira sessão em
# que ele aparece
patamares_anotados = set()
for eid, idx in data.groupby(ensaio_id).groups.items():
    sub = data.loc[idx]
    for f, bloco in sub.groupby('Inversor'):
        if f == 0 or f in patamares_anotados:
            continue
        patamares_anotados.add(f)
        x_meio = bloco['tempo_experimental'].median()
        ax_freq.annotate(f'{int(f)} Hz', xy=(x_meio, f), xytext=(0, 8),
                          textcoords='offset points', ha='center', fontsize=9)

plt.tight_layout()
fig.savefig('variacao_freq_pressao_vazao.png', dpi=150)
plt.show()
