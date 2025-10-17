import sys
import argparse
from numpy import *
from skimage.metrics import peak_signal_noise_ratio as PSNR
import imageio.v2 as imageio


def WSPSNR(img1, img2, max=255.0):
    def __weights(height, width):
        phis = arange(height + 1) * pi / height
        deltaTheta = 2 * pi / width
        column = asarray(
            [deltaTheta * (-cos(phis[j + 1]) + cos(phis[j])) for j in range(height)]
        )
        return repeat(column[:, newaxis], width, 1)

    height, width, _ = img1.shape
    w = __weights(height, width)
    w = stack((w,) * 3, axis=-1)
    mse = mean(((img1 - img2) ** 2 * w), axis=2)
    wmse = sum(mse) / (4 * pi)

    print(f"WMSE = {wmse}")
    print(f"WSPSNR = {10 * log10(max**2 / wmse)}")

    return 10 * log10(max**2 / wmse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True, help="Caminho da imagem original")
    parser.add_argument("--dec", required=True, help="Caminho da imagem decodificada")
    args = parser.parse_args()

    img1 = imageio.imread(args.ref).astype(float)
    img2 = imageio.imread(args.dec).astype(float)

    print(img1.max(), img1.min(), img2.max(), img2.min())
    max_val = 255.0 if img1.max() > 1 else 1.0
    wspsnr = WSPSNR(img1, img2, max_val)
    psnr = PSNR(img1 / 255, img2 / 255)
    print(wspsnr, psnr)
    return wspsnr, psnr


if __name__ == "__main__":
    main()
