import numpy as np
from PIL import Image
import sys
import os

iters = 5

def synthPatchMatch(img, refs, winsize=7):
    r = winsize//2
    h, w = img.shape[:2]
    rsmax = max(h,w)
    n = h*w
    refn = refs.shape[0]

    def refRandVec(i, ref):
        return [ref, np.random.randint(h)-i//w, np.random.randint(w)-i%w]
    def randVec(i):
        return refRandVec(i, np.random.randint(refn))
    nnf = np.array([randVec(i) for i in range(0,n)])

    padimg = np.pad(img, ((r,r),(r,r),(0,0)), mode="symmetric")
    padref = np.pad(refs, ((0,0),(r,r),(r,r),(0,0)), mode="symmetric")
    def ssd(i, vec):
        x = i//w
        y = i%w
        tWin = padimg[x:x+winsize,y:y+winsize]
        x += vec[1]
        y += vec[2]
        sWin = padref[vec[0],x:x+winsize,y:y+winsize]
        return np.sum((tWin-sWin)**2)
    nnd = np.array([ssd(i, nnf[i]) for i in range(0,n)])

    def improve(i, vec):
        x = i//w+vec[1]
        y = i%w+vec[2]
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
            # planeshift
            for ref in range(0,refn):
                improve(j, refRandVec(j, ref))
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
                improve(j, [nnf[j,0],xc,yc])
                rs //= 2
            print("pixel", str(j) + "/" + str(n), "iter", str(i+1) + "/" + str(iters))

    subimg = np.zeros(img.shape)
    for i in range(0,n):
        x = i//w+nnf[i,1]
        y = i%w+nnf[i,2]
        subimg[i//w,i%w] = refs[nnf[i,0],x,y]
    outimg = Image.fromarray(subimg.astype(np.uint8))
    outimg.save("../data/refsOut.png")

if __name__ == "__main__":
    img = np.array(Image.open(sys.argv[1]))
    dir = sys.argv[2]
    refs = np.array([np.array(Image.open(os.path.join(dir, file))) for file in os.listdir(dir)])
    if len(sys.argv) > 3:
        winsize = int(sys.argv[3])
        if not winsize % 2:
            winsize += 1
    else:
        winsize = 7
    synthPatchMatch(img, refs, winsize)