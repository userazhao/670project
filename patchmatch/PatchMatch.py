import numpy as np
from PIL import Image
import sys
import os

iters = 5

def synthPatchMatch(img, refs, rsmax, winsize=17):
    r = winsize // 2
    padded = [np.pad(ref, ((r,r),(r,r),(0,0)), mode="symmetric") for ref in refs]
    padded.insert(0, np.pad(img, ((r,r),(r,r),(0,0)), mode="symmetric"))
    mask = padded[0][:,:,3] != 255

    def indToPos(ind):
        return [ind // h + minbx + r, ind % h + minby + r]
    def fill(i):
        holePos = indToPos(i)
        if mask[holePos[0],holePos[1]]:
            padded[0][holePos[0],holePos[1]] = padded[nnf[i][0]][nnf[i][1],nnf[i][2]]
    def ssd(hInd, patchPos):
        holePos = indToPos(hInd)
        fImg = padded[0][holePos[0]:holePos[0]+r+r, holePos[1]:holePos[1]+r+r,:3]
        sImg = padded[patchPos[0]][patchPos[1]:patchPos[1]+r+r,patchPos[2]:patchPos[2]+r+r,:3]
        return np.sum((fImg-sImg)**2)

    holes = np.transpose(np.nonzero(img[:,:,3] != 255)) # list of transparent pixels
    minbx = np.min(holes[:,0]) - r
    maxbx = np.max(holes[:,0]) + r
    minby = np.min(holes[:,1]) - r
    maxby = np.max(holes[:,1]) + r
    print(minbx, maxbx, minby, maxby) # debug
    samples = [] # list of indices of valid sample patches
    for x in range(0,img.shape[0]):
        for y in range(0,img.shape[1]):
            if (x < minbx or x > maxbx) and (y < minby or y > maxby):
                samples.append((0, x, y))
    for i in range(0,len(refs)):
        ref = refs[i]
        for x in range(0,ref.shape[0]):
            for y in range(0,ref.shape[1]):
                samples.append((i+1, x, y))
    h = maxbx-minbx
    w = maxby-minby
    n = h*w
    print(h, w)
    print(n)
    nnf = []
    for i in range(0,n): # randomize
        nnf.append(samples[np.random.choice(len(samples))])
        fill(i)
    nnf = np.array(nnf)
    nnd = []
    for i in range(0,n):
        nnd.append(ssd(i, nnf[i]))

    def improve(hInd, patchPos):
        candD = ssd(hInd, patchPos)
        if nnd[hInd] > candD:
            nnd[hInd] = candD
            nnf[hInd] = patchPos
            fill(hInd)

    for i in range(0,iters):
        if i % 2:
            s = -1
        else:
            s = 1
        if i != 0:
            nnf = nnf[::-1]
            nnd = nnd[::-1]
        # propagate
        for j in range(0,n):
            x, y = indToPos(i)
            nInd = np.transpose(np.nonzero(np.all(holes == [x+s,y], axis=1))) #fix this
            if nInd.size > 0:
                nInd = nInd[0,0]
                improve(j, (nnf[nInd,0], nnf[nInd,1]-s, nnf[nInd,2]))

            nInd = np.transpose(np.nonzero(np.all(holes == [x,y+s], axis=1)))
            if nInd.size > 0:
                nInd = nInd[0,0]
                improve(j, (nnf[nInd][0], nnf[nInd,1], nnf[nInd,2]-s))
        # search
            rs = rsmax
            while rs > 1:
                xmin = max(x-rs, 0)
                ymin = max(y-rs, 0)
                xmax = min(x+rs, padded[nnf[j,0]].shape[0]-1-r-r)
                ymax = min(y+rs, padded[nnf[j,0]].shape[1]-1-r-r)
                xc = np.random.randint(xmin, xmax)
                yc = np.random.randint(ymin, ymax)
                improve(j, (nnf[j,0], xc, yc))
                rs //= 2
            print("pixel", str(j+1) + "/" + str(n), "iter", str(i+1) + "/" + str(5))
    out = Image.fromarray(padded[0][r:-r,r:-r])
    out.save("../data/patchOut.png")

if __name__ == "__main__":
    img = np.array(Image.open(sys.argv[1]))
    dir = sys.argv[2]
    refs = []
    for file in os.listdir(dir):
        refs.append(np.array(Image.open(os.path.join(dir, file))))
    rsmax = int(sys.argv[3])
    if len(sys.argv) > 4:
        winsize = sys.argv[4]
        if not winsize % 2:
            winsize += 1
    else:
        winsize = 7
    synthPatchMatch(img, refs, rsmax, winsize)