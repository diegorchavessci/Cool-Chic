import argparse
from skimage import io, transform, img_as_ubyte

def main():
    parser = argparse.ArgumentParser(
        description="Redimensionar imagem com skimage.")
    parser.add_argument("-i", "--input", required=True, 
                        help="Caminho da imagem de entrada")
    parser.add_argument("-o", "--output", required=True, 
                        help="Caminho para salvar a imagem redimensionada")
    parser.add_argument("-w", "--width", required=True, type=int, 
                        help="Nova largura")
    parser.add_argument("-H", "--height", required=True, type=int, 
                        help="Nova altura")
    args = parser.parse_args()

    # Carregar imagem
    img = io.imread(args.input)

    # Redimensionar (mantém valores entre 0 e 1)
    resized = transform.resize(img, (args.height, args.width), 
                               anti_aliasing=True)

    # Converter de volta para uint8 (0–255)
    resized_uint8 = img_as_ubyte(resized)

    # Salvar
    io.imsave(args.output, resized_uint8)
    print(f"Imagem salva em: {args.output}")

if __name__ == "__main__":
    main()
