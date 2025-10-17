import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import csv

from diegorchaves.wspsnr import WSPSNR
import imageio.v2 as imageio


def run_experiment(images_dir: str, lambdas, mode: str = "wsmse"):
    assert mode in ["wsmse", "mse"], "Modo inválido. Use 'wsmse' ou 'mse'."
    script_path = "coolchic/encode.py" if mode == "wsmse" else "coolchic_mse/encode.py"
    metric_flag = 1 if mode == "wsmse" else 0

    all_bpps, all_times, all_psnrs, all_wspsnrs = [], [], [], []

    # Listar imagens
    images_info = [
        {"path": os.path.join(images_dir, fname)}
        for fname in os.listdir(images_dir)
        if fname.endswith(".png")
    ]

    for img_info in images_info:
        image_path = img_info["path"]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\nProcessando imagem: {image_path} | Modo: {mode}")

        bpps, times, psnrs, wspsnrs = [], [], [], []

        for lmbda in lambdas:
            workdir = f"./workdirs/{image_name}_lmbda_{lmbda:.0e}_{mode}/"
            if os.path.exists(workdir):
                shutil.rmtree(workdir)
            os.makedirs(workdir, exist_ok=True)

            # Codificação
            cmd = [
                "python",
                script_path,
                f"--input={image_path}",
                f"--output=./bitstream.cool",
                f"--workdir={workdir}",
                "--enc_cfg=cfg/enc/intra/fast_10k.cfg",
                "--dec_cfg_residue=cfg/dec/intra_residue/hop.cfg",
                f"--lmbda={lmbda}",
            ]
            _ = subprocess.run(cmd, check=True)

            # Leitura dos resultados
            results_path = os.path.join(workdir, "0000-results_best.tsv")
            with open(results_path) as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    parts = lines[1].split()
                    psnr = float(parts[2])
                    bpp = float(parts[4])
                    time = parts[6]
                    psnrs.append(psnr)
                    bpps.append(bpp)
                    times.append(time)

            # Calcular WS-PSNR
            decoded_path = os.path.join(workdir, f"0000-decoded-{image_name}.png")
            img1 = imageio.imread(image_path).astype(float)
            img2 = imageio.imread(decoded_path).astype(float)
            wspsnr = WSPSNR(img1, img2)
            wspsnrs.append(wspsnr)

        # Salvar CSV individual por imagem
        os.makedirs("graficos/csvs", exist_ok=True)
        csv_image_path = f"graficos/csvs/{image_name}.csv"
        write_header = not os.path.exists(
            csv_image_path
        )  # Se não existe, escreve o cabeçalho
        with open(csv_image_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["lambda", "psnr", "image_name", "bpp", "wsmse", "wspsnr", "time"]
                )
            for l, p, b, t, ws in zip(lambdas, psnrs, bpps, times, wspsnrs):
                writer.writerow([l, p, f"{image_name}.png", b, metric_flag, ws, t])

        all_bpps.append(bpps)
        all_times.append(times)
        all_psnrs.append(psnrs)
        all_wspsnrs.append(wspsnrs)

    # Salvar CSV geral
    with open("graficos/csvs/todas_imagens.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if os.stat("graficos/csvs/todas_imagens.csv").st_size == 0:
            writer.writerow(
                ["lambda", "psnr", "image_name", "bpp", "wsmse", "wspsnr", "time"]
            )
        for i, img_info in enumerate(images_info):
            image_name = os.path.basename(img_info["path"])
            for l, p, b, t, ws in zip(
                lambdas, all_psnrs[i], all_bpps[i], all_times[i], all_wspsnrs[i]
            ):
                writer.writerow([l, p, image_name, b, metric_flag, ws, t])


# ======= EXECUÇÃO =======

lmbdas = np.logspace(-4, -2, num=10)
print("Lambdas to test:", lmbdas)

images_dir = "/home/diego/Cool-Chic/imagens_4k/512 + png"

# Limpar CSV geral antes de rodar
if os.path.exists("graficos/csvs/todas_imagens.csv"):
    os.remove("graficos/csvs/todas_imagens.csv")

# Executar para ambos os modos
run_experiment(images_dir, lmbdas, mode="wsmse")
run_experiment(images_dir, lmbdas, mode="mse")
