import numpy as np
from PIL import Image
import sys
import os

iters = 5

def synthPatchMatch(img, refs, rsmax, winsize=7):
    r = winsize // 2
    padded = [np.pad(ref, ((r,r),(r,r),(0,0)), mode="symmetric") for ref in refs]
    padded.insert(0, np.pad(img, ((r,r),(r,r),(0,0)), mode="symmetric"))

    def fill(i):
        padded[0][holes[i]] = padded[nnf[i][0]][nnf[i][1:]]
    def ssd(holePos, patchPos):
        fImg = padded[0][holePos[0]-r:holePos[0]+r, holePos[1]-r:holePos[1]+r,:2]
        sImg = padded[patchPos[0]][patchPos[1]-r:patchPos[1]+r,patchPos[2]-r:patchPos[2]+r,:2]
        return np.sum((fImg-sImg)**2)

    samples = [] # list of indices of valid sample patches
    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            if img[x,y,3] != 0:
                samples.append((0, x, y))
    for i in range(0,len(refs)):
        ref = refs[i]
        for x in range(0,ref.shape[0]):
            for y in range(0,ref.shape[1]):
                samples.append((i+1, x, y))
    holes = np.nonzero(img[:,:,3] != 255) # list of transparent pixels
    n = holes.shape[0]
    nnf = []
    for i in range(0,n): # randomize
        nnf.append(np.random.choice(samples))
        fill(i)
    nnd = []
    for i in range(0,n):
        nnd.append(ssd(holes[i], nnf[i]))

    def improve(hInd, patchPos):
        candD = ssd(holes[hInd], padded[patchPos[0]][patchPos[1:]])
        if nnd[hInd] > candD:
            nnf[hInd] = patchPos
            fill(hInd)

    for i in range(0,iters):
        if iter % 2:
            s = -1
        else:
            s = 1
        if i != 0:
            nnf = nnf[::-1]
            nnd = nnd[::-1]
        # propagate
        for j in range(0,n):
            x = holes[i][0]
            y = holes[i][1]
            nInd = np.nonzero(holes == (x+s, y))
            if nInd.size > 0:
                improve(j, (nnf[nInd[0]][0], nnf[nInd[0]][1]-s, nnf[nInd[0]][2]))

            nInd = np.nonzero(holes == (x, y+s))
            if nInd.size > 0:
                improve(j, (nnf[nInd[0]][0], nnf[nInd[0]][1], nnf[nInd[0]][2]-s))
        # search
            rs = rsmax
            while rs > 1:
                xmin = max(x-rs, 0)
                ymin = max(y-rs, 0)
                xmax = min(x+rs, samples[nnf[j][0]].shape[0]-r-r)
                ymax = min(y+rs, samples[nnf[j][0]].shape[1]-r-r)
                xc = np.random.rand(np.range(xmin, xmax))
                yc = np.random.rand(np.range(ymin, ymax))
                improve(j, (nnf[j][0], xc, yc))
                rs //= 2
            print("pixel", str(j) + "/" + str(n))
    out = Image.fromarray(padded[0][r:-r,r:-r])
    out.save("output.png")

if __name__ == "__main__":
    img = np.array(Image.open(sys.argv[1]))
    dir = sys.argv[2]
    refs = []
    for file in os.listdir(dir):
        refs.append(np.array(Image.open(os.path.join(dir, file))))
    rsmax = sys.argv[3]
    if len(sys.argv) > 4:
        winsize = sys.argv[4]
        if not winsize % 2:
            winsize += 1
    else:
        winsize = 7
    synthPatchMatch(img, refs, rsmax, winsize)