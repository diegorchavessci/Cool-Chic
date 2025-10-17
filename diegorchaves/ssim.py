import skimage.io
import skimage.metrics
import argparse

def calcular_ssim(path_ref: str, path_test: str) -> float:
    """Calcula o SSIM entre duas imagens."""
    img_truth = skimage.io.imread(path_ref)
    img_test = skimage.io.imread(path_test)
    ssim = skimage.metrics.structural_similarity(img_truth, img_test, 
                                                 channel_axis=2)
    return ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Calcular SSIM entre duas imagens.')
    parser.add_argument('--ref', required=True, type=str, help=
                        'Caminho da imagem de referência (truth)')
    parser.add_argument('--test', required=True, type=str, help=
                        'Caminho da imagem testada (decoded)')
    args = parser.parse_args()

    ssim_value = calcular_ssim(args.ref, args.test)
    print(f'SSIM: {ssim_value}')
