import numpy as np
from PIL import Image
import sys

iters = 5

def synthPatchMatch(img, ref, winsize=7):
    r = winsize//2
    h, w = img.shape[:2]
    rsmax = max(h,w)
    n = h*w
    nnf = np.array([[np.random.randint(h)-i//w, np.random.randint(w)-i%w] for i in range(0,n)])

    padimg = np.pad(img, ((r,r),(r,r),(0,0)), mode="symmetric")
    padref = np.pad(ref, ((r,r),(r,r),(0,0)), mode="symmetric")
    def ssd(i, vec):
        x = i//w
        y = i%w
        tWin = padimg[x:x+winsize,y:y+winsize]
        x += vec[0]
        y += vec[1]
        sWin = padref[x:x+winsize,y:y+winsize]
        return np.sum((tWin-sWin)**2)
    nnd = np.array([ssd(i, nnf[i]) for i in range(0,n)])

    def improve(i, vec):
        x = i//w+vec[0]
        y = i%w+vec[1]
        if x >= 0 and x < h and y >= 0 and y < w:
            candD = ssd(i, vec)
            if nnd[i] > candD:
                nnd[i] = candD
                nnf[i] = vec

    indlist = np.arange(0,n)
    for i in range(0,iters):
        if i % 2:
            s = 1
        else:
            s = -1
        if i != 0:
            indlist = indlist[::-1]
        for j in indlist:
            # propagate
            nInd = j + s
            if nInd >= 0 and nInd < n:
                improve(j, nnf[nInd])

            nInd = min(max(j + s*w, 0), n-1)
            if nInd >= 0 and nInd < n:
                improve(j, nnf[nInd])
            # search
            x = j//w
            y = j%w
            rs = rsmax
            while rs > 1:
                xmin = max(-rs, -x)
                ymin = max(-rs, -y)
                xmax = min(rs, h-x-1)
                ymax = min(rs, w-y-1)
                xc = np.random.randint(xmin, xmax)
                yc = np.random.randint(ymin, ymax)
                improve(j, [xc,yc])
                rs //= 2
            print("pixel", str(j) + "/" + str(n), "iter", str(i+1) + "/" + str(iters))

    subimg = np.zeros(img.shape)
    for i in range(0,n):
        x = i//w+nnf[i,0]
        y = i%w+nnf[i,1]
        subimg[i//w,i%w] = ref[x,y]
    outimg = Image.fromarray(subimg.astype(np.uint8))
    outimg.save("../data/basicOut.png")

if __name__ == "__main__":
    img = np.array(Image.open(sys.argv[1]))
    ref = np.array(Image.open(sys.argv[2]))
    if len(sys.argv) > 3:
        winsize = sys.argv[3]
        if not winsize % 2:
            winsize += 1
    else:
        winsize = 7
    synthPatchMatch(img, ref, winsize)