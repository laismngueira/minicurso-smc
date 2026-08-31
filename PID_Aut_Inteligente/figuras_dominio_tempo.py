import pandas as pd
import matplotlib.pyplot as plt

# Carregar os dados coletados na planta do LENHS/UFPB
file_path = 'DadosTratados.xlsx'
data = pd.read_excel(file_path)

# A coleta tem pausas de reconfiguração do inversor entre patamares de frequência
# (a maior delas de ~16h, de um dia para o outro, e uma de ~34 min entre 30 e 35 Hz).
# Essas pausas são identificadas automaticamente e usadas para separar os dados em
# "ensaios" contínuos, evitando que o matplotlib desenhe uma reta interpolada entre
# trechos sem coleta (o que sugeriria, de forma enganosa, uma variação gradual de
# pressão/vazão quando na verdade a bomba estava parada/sendo reconfigurada).
gap = data['Timestamp'].diff()
ensaio_id = (gap > pd.Timedelta(minutes=10)).cumsum()

for ensaio, grupo in data.groupby(ensaio_id):
    tempo = grupo['Timestamp']
    freq = grupo['Inversor']       # frequência do inversor (atuador - velocidade da bomba)
    vazao = grupo['FT_4A']         # vazão medida no ponto 4A
    pressao = grupo['PT_4A']       # pressão medida no ponto 4A (variável controlada)

    # Instantes onde a frequência do inversor muda de patamar dentro do ensaio
    transicoes = tempo[freq.diff().fillna(0) != 0]

    fig, (ax_freq, ax_vazao, ax_pressao) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax in (ax_freq, ax_vazao, ax_pressao):
        for t in transicoes:
            ax.axvline(t, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.grid(True, alpha=0.3)

    ax_freq.plot(tempo, freq, color='#4B4B4B', linewidth=1.5)
    ax_freq.set_ylabel('Frequência (Hz)')
    ax_freq.set_title(f'Ensaio {ensaio + 1} - Frequência do inversor (atuador) ao longo do tempo')

    ax_vazao.plot(tempo, vazao, color='#1B6BB0', linewidth=1.5)
    ax_vazao.set_ylabel('Vazão - FT_4A')
    ax_vazao.set_title(f'Ensaio {ensaio + 1} - Vazão ao longo do tempo')

    ax_pressao.plot(tempo, pressao, color='#D1495B', linewidth=1.5)
    ax_pressao.set_ylabel('Pressão - PT_4A')
    ax_pressao.set_title(f'Ensaio {ensaio + 1} - Pressão ao longo do tempo')
    ax_pressao.set_xlabel('Tempo')

    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(f'ensaio_{ensaio + 1}_dominio_tempo.png', dpi=150)

plt.show()

# ---------------------------------------------------------------------------
# Aspectos físicos da planta observados nos dados (para uso no relatório)
# ---------------------------------------------------------------------------
resumo = data[data['Inversor'] > 0].groupby('Inversor')[['FT_4A', 'PT_4A']].max()

print('\nResumo por patamar de frequência (valores máximos observados):')
print(resumo.to_string())

print("""
Comentários sobre o comportamento físico da planta:

1. Cada patamar de frequência do inversor (30, 35, 40, 45, 50 Hz) define um novo
   ponto de operação da bomba centrífuga. Dentro de cada patamar, a válvula CV_4A
   é fechada progressivamente, o que desloca o ponto de operação ao longo da
   curva característica da bomba: com a válvula aberta (CV_4A ~ 0°) a vazão é
   máxima e a pressão é mínima; com a válvula fechada (CV_4A ~ 30°) a vazão tende
   a zero e a pressão atinge o valor de "shutoff head" (pressão máxima com vazão
   nula), efeito clássico de estrangulamento (throttling) em sistemas hidráulicos.

2. O aumento da frequência do inversor eleva tanto a vazão máxima quanto a
   pressão de shutoff (ver tabela acima), de forma consistente com as leis de
   afinidade de bombas centrífugas: a vazão varia aproximadamente de forma linear
   com a rotação (Q proporcional a N) e a pressão/altura manométrica varia com o
   quadrado da rotação (H proporcional a N^2). Isso explica por que a pressão
   cresce proporcionalmente mais rápido que a vazão à medida que a frequência
   aumenta.

3. Observa-se um atraso (tempo morto) entre a mudança de frequência/abertura da
   válvula e a estabilização da vazão e da pressão, associado à inércia hidráulica
   do fluido nas tubulações e ao tempo de resposta do inversor/bomba - relevante
   para o projeto do controlador PI, pois limita a velocidade de resposta em
   malha fechada sem gerar sobressinal excessivo.

4. No intervalo sem coleta contínua (entre ~18h de 17/07 e ~10h de 18/07, quando
   Inversor = 0), tanto a vazão quanto a pressão caem a valores próximos de zero,
   evidenciando que, com a bomba desligada, não há energia hidráulica no sistema
   (sem carga estática residual relevante nesse ponto de medição).
""")
