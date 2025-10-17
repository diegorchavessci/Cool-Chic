from PIL import Image
import os

def convert_bmp_to_png(input_dir, output_dir, size=(512, 256)):
    """
    Converts a BMP image to PNG format.

    Args:
        input_path (str): The path to the input JPG file.
        output_path (str): The path where the output PNG file will be saved.
    """
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # Verifica se é um arquivo e termina com .bmp
        if os.path.isfile(input_path) and filename.lower().endswith('.bmp'):
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_dir, output_filename)
            try:
                with Image.open(input_path) as img:
                    resized_img = img.resize(size, Image.Resampling.LANCZOS)
                    resized_img.save(output_path, format='PNG')
                    print(f"Convertido e redimensionado: '{input_path}' → '{output_path}'")
            except Exception as e:
                print(f"Erro ao processar '{input_path}': {e}")

# Example usage:
input_file = "/home/diego/Cool-Chic/imagens_4k/originais"  # Replace with your JPG file name
output_file = "/home/diego/Cool-Chic/imagens_4k/512 + png" # Desired output PNG file name

convert_bmp_to_png(input_file, output_file)