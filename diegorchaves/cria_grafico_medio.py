import csv
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Caminho do CSV geral
csv_path = "graficos/csvs/todas_imagens.csv"

# Verifica se o arquivo existe
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

# Dicionário: {(lambda, wsmse): [ (bpp, wspsnr), ... ]}
data = defaultdict(list)

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        lmbda = float(row["lambda"])
        wsmse = int(row["wsmse"])
        bpp = float(row["bpp"])
        wspsnr = float(row["wspsnr"])
        data[(lmbda, wsmse)].append((bpp, wspsnr))

# Agrupar e calcular médias
avg_results = defaultdict(lambda: {"bpp": [], "wspsnr": []})

for (lmbda, wsmse), values in data.items():
    bpps, wspsnrs = zip(*values)
    avg_bpp = np.mean(bpps)
    avg_wspsnr = np.mean(wspsnrs)
    avg_results[wsmse]["bpp"].append((lmbda, avg_bpp))
    avg_results[wsmse]["wspsnr"].append((lmbda, avg_wspsnr))

# Ordenar por lambda (para garantir ordem no plot)
for wsmse in [0, 1]:
    avg_results[wsmse]["bpp"].sort()
    avg_results[wsmse]["wspsnr"].sort()

# Extrair valores
bpp_mse = [b for _, b in avg_results[0]["bpp"]]
wspsnr_mse = [w for _, w in avg_results[0]["wspsnr"]]

bpp_wsmse = [b for _, b in avg_results[1]["bpp"]]
wspsnr_wsmse = [w for _, w in avg_results[1]["wspsnr"]]

# Plotar gráfico
plt.figure(figsize=(8, 6))
plt.plot(bpp_mse, wspsnr_mse, 'o-', label="MSE", color='tab:orange')
plt.plot(bpp_wsmse, wspsnr_wsmse, 's-', label="WS-MSE", color='tab:blue')

plt.xlabel("Média bpp")
plt.ylabel("Média WS-PSNR (dB)")
plt.title("RD Médio - Todas as Imagens")
plt.grid(True)
plt.legend()
# Adicionar anotações com lambda em notação científica
for (lmbda, b), (_, w) in zip(avg_results[0]["bpp"], avg_results[0]["wspsnr"]):
    plt.text(b, w, f"{lmbda:.0e}", fontsize=8, ha='left', va='bottom', color='tab:orange')

for (lmbda, b), (_, w) in zip(avg_results[1]["bpp"], avg_results[1]["wspsnr"]):
    plt.text(b, w, f"{lmbda:.0e}", fontsize=8, ha='right', va='top', color='tab:blue')

plt.tight_layout()

# Salvar o gráfico
output_path = "graficos/rd_medio.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Gráfico RD médio salvo em: {output_path}")
