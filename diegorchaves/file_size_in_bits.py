import argparse
import os

parser = argparse.ArgumentParser('Calcular o tamanho em bits de um arquivo.')
parser.add_argument('-i', required=True, type=str, 
                    help='Caminho da do arquivo de referencia.')

args = parser.parse_args()

try:
    size_in_bytes = os.path.getsize(args.i)
    size_in_bits = size_in_bytes * 8
    print(f'Tamanho em bytes: ', size_in_bytes)
    print(f'Tamanho em bits: ', size_in_bits)

except FileNotFoundError:
    print(f"Error: File not found at '{args.i}'")

except OSError as e:
        print(f"Error accessing file '{args.i}': {e}")