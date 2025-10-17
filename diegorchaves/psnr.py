import skimage.io
import skimage.metrics
import argparse

def calcular_psnr(path_ref: str, path_test: str) -> float:
    """Calcula o PSNR entre duas imagens."""
    img_truth = skimage.io.imread(path_ref)
    img_test = skimage.io.imread(path_test)
    psnr = skimage.metrics.peak_signal_noise_ratio(img_truth, img_test)
    return psnr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Calcular PSNR entre duas imagens.')
    parser.add_argument('--ref', required=True, type=str, help=
                        'Caminho da imagem de referência (truth)')
    parser.add_argument('--test', required=True, type=str, help=
                        'Caminho da imagem testada (decoded)')
    args = parser.parse_args()

    psnr_value = calcular_psnr(args.ref, args.test)
    print(f'PSNR: {psnr_value}')
