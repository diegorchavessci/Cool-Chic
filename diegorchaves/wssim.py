from numpy import *
from scipy import signal
import imageio.v2 as imageio
import argparse

def WSSSIM(img1, img2, K1 = .01, K2 = .03, L = 255):

    def __fspecial_gauss(size, sigma):
        x, y = mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    def __weights(height, width):
        deltaTheta = 2*pi/width 
        column = asarray([cos( deltaTheta * (j - height/2.+0.5)) for j in range(height)])
        return repeat(column[:, newaxis], width, 1)

    img1 = float64(img1)
    img2 = float64(img2)

    k = 11
    sigma = 1.5
    window = __fspecial_gauss(k, sigma)
    window2 = zeros_like(window); window2[k//2,k//2] = 1
 
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    # ====== window.shape = [11, 11], img1.shape = [256, 512, 3] =======
    print(window.shape)
    print(img1.shape)

    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')
    
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    
    sigma1_sq = signal.convolve2d(img1*img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2*img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1*img2, window, 'valid') - mu1_mu2
   
    W = __weights(*img1.shape)
    Wi = signal.convolve2d(W, window2, 'valid')

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)) * Wi
    mssim = sum(ssim_map)/sum(Wi)

    return mssim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True, help='Caminho da imagem original')
    parser.add_argument('--dec', required=True, help='Caminho da imagem decodificada')
    args = parser.parse_args()

    img1 = imageio.imread(args.ref).astype(float)
    img2 = imageio.imread(args.dec).astype(float)
    
    result = WSSSIM(img1, img2)
    print(result)

if __name__ == "__main__":
    main()