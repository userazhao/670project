import numpy as np
import scipy.ndimage as sp
import sys
from PIL import Image
import os

def synthEfrosLeung(img, refs, winsize=7):
    ErrThreshold = 0.1
    r = winsize // 2
    h,w = img.shape[:2]
    out = img
    refcache = [np.lib.stride_tricks.sliding_window_view(np.pad(ref, ((r,r),(r,r),(0,0)), mode="symmetric"), (winsize, winsize), axis=(0,1)) for ref in refs]
    refcache.insert(0, np.lib.stride_tricks.sliding_window_view(np.pad(img, ((r,r),(r,r),(0,0)), mode="symmetric"), (winsize, winsize), axis=(0,1)))
    counter = 0
    done = False
    while not done:
        mask_im = (out[:,:,3] == 255).astype(int)
        window = np.ones((winsize, winsize))
        dilation = sp.binary_dilation(mask_im, structure=window) - mask_im
        pixelList = []
        for i in range(0,h):
            for j in range(0,w):
                if dilation[i,j]:
                    pixelList.append((i,j))
        sizes = sp.convolve(mask_im, window, mode="constant")
        sizes = [sizes[p] for p in pixelList]
        indices = np.argsort(sizes)
        pixelList = np.array(pixelList)[indices[::-1]]
        px, py = pixelList[0]
        def maskCheck(i, j):
            if i < 0 or j < 0 or i >= h or j >= w:
                return 0
            return mask_im[i,j]
        ValidMask = np.array([[maskCheck(i,j) for j in range(py-r,py+r+1)] for i in range(px-r,px+r+1)])
        padout = np.pad(out, ((r,r),(r,r),(0,0)), mode="symmetric")
        hood = np.moveaxis(padout[px:px+winsize,py:py+winsize], 2, 0)
        ssds = [np.sum((windows - hood) ** 2 * ValidMask, axis=(2,3,4)) + 256*winsize**2*(windows[:,:,3,r,r] != 255) for windows in refcache]
        minSSD = np.min(ssds)
        BestMatches = []
        for n in range(0,len(ssds)):
            for i in range(0,h):
                for j in range(0,w):
                    if ssds[n][i,j] <= minSSD*(1+ErrThreshold):
                        BestMatches.append((n,i,j))
        ind = BestMatches[np.random.choice(len(BestMatches))]
        if ind[0] == 0:
            out[px,py] = img[ind[1:]]
        else:
            out[px,py] = refs[ind[0]-1][ind[1:]]
        done = len(pixelList) == 1
        counter += 1
        print(counter, len(pixelList))
    outimg = Image.fromarray(out.astype(np.uint8))
    outimg.save("../data/efOut.png")

if __name__ == "__main__":
    img = np.array(Image.open(sys.argv[1]))
    dir = sys.argv[2]
    refs = []
    for file in os.listdir(dir):
        refs.append(np.array(Image.open(os.path.join(dir, file))))
    if len(sys.argv) > 3:
        winsize = sys.argv[3]
        if not winsize % 2:
            winsize += 1
    else:
        winsize = 7
    synthEfrosLeung(img, refs, winsize)