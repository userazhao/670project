import numpy as np
from skimage import io
import random
import scipy.ndimage as sp
import time

def synthRandomPatch(img, tileSize, numTiles, outSize):
    h,w = img.shape[0:2]
    out = np.zeros((outSize, outSize), dtype=img.dtype)
    for i in range(0,numTiles):
        for j in range(0,numTiles):
            x,y = random.randrange(h-tileSize), random.randrange(w-tileSize)
            out[j*tileSize:(j+1)*tileSize,i*tileSize:(i+1)*tileSize] = img[x:x+tileSize,y:y+tileSize]
    return out

def synthEfrosLeung(img, winsize, outSize):
    ErrThreshold = 0.1
    r = winsize // 2
    c = outSize // 2
    h,w = img.shape[0:2]
    out = -np.ones((outSize, outSize))
    x,y = random.randrange(h-3), random.randrange(w-3)
    out[c-1:c+2,c-1:c+2] = img[x:x+3,y:y+3]
    padimg = np.pad(img, r, mode="symmetric")
    windows = np.array([[padimg[i:i+winsize,j:j+winsize] for j in range(0,w)] for i in range(0,h)])
    counter = 9
    done = False
    while not done:
        mask_im = (out >= 0).astype(int)
        window = np.ones((winsize, winsize))
        dilation = sp.binary_dilation(mask_im, structure=window) - mask_im
        pixelList = [(i//outSize,i%outSize) for i in range(0,dilation.size) if dilation[i//outSize,i%outSize]]
        sizes = sp.convolve(mask_im, window, mode="constant")
        sizes = [sizes[p] for p in pixelList]
        indices = np.argsort(sizes)
        pixelList = np.array(pixelList)[indices[::-1]]
        px, py = pixelList[0]
        def maskCheck(i, j):
            if i < 0 or j < 0 or i >= outSize or j >= outSize:
                return 0
            return mask_im[i,j]
        ValidMask = np.array([[maskCheck(i,j) for j in range(py-r,py+r+1)] for i in range(px-r,px+r+1)])
        padout = np.pad(out, r, mode="symmetric")
        hood = padout[px:px+winsize,py:py+winsize]
        ssds = np.sum((windows - hood) ** 2 * ValidMask, axis=(2,3))
        minSSD = np.min(ssds)
        BestMatches = [(i//w, i%w) for i in range(0,ssds.size) if ssds[i//w,i%w] <= minSSD*(1+ErrThreshold)]
        out[px,py] = img[random.choice(BestMatches)]
        done = len(pixelList) == 1
        counter += 1
        print("pixel", str(counter) + "/" + str(outSize**2))
    return out.astype(np.uint8)

names = ["D20.png", "Texture2.bmp", "english.jpg"]
tileSizes = [15, 20, 30, 40]
numTiles = 5
winsizes = [5, 7, 11, 15]
runtime = {}
for name in names:
    img = io.imread('../data/texture/' + name)[:,:,0]
    outSize = 70
    for winsize in winsizes:
        if winsize == 5 or winsize == 15:
            start = time.time()
        im_synth = synthEfrosLeung(img, winsize, outSize)
        outname = "../efrosleung/" + str(winsize) + "w" + name
        print(outname, "done")
        io.imsave(outname, im_synth)
        if winsize == 5 or winsize == 15:
            end = time.time()
            runtime[(name, winsize)] = end - start
    for tileSize in tileSizes:
        outSize = numTiles * tileSize # calculate output image size
        im_patch = synthRandomPatch(img, tileSize, numTiles, outSize)
        outname = "../random/" + str(tileSize) + "t" + name
        print(outname, "done")
        io.imsave(outname, im_patch)
print("runtime", runtime)