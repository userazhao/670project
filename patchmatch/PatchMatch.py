import numpy as np
from PIL import Image
import sys
import os

iters = 5

def synthPatchMatch(img, refs, rsmax, winsize=7):
    r = winsize // 2
    padded = [np.pad(ref, ((r,r),(r,r),(0,0)), mode="symmetric") for ref in refs]
    # padded.insert(0, np.pad(img, ((r,r),(r,r),(0,0)), mode="symmetric"))
    padded.insert(0, np.pad(refs[0], ((r,r),(r,r),(0,0)), mode="symmetric"))
    mask = img[:,:,3] != 255

    def subFill(i):
        subimg[i//w,i%w] = padded[nnf[i][0]][nnf[i][1],nnf[i][2]]
    def indToPos(ind):
        return [ind // w + minbx + r, ind % w + minby + r]
    def fill(i):
        holePos = [i // w + minbx, i % w + minby]
        if mask[holePos[0],holePos[1]]:
            img[holePos[0],holePos[1]] = subimg[i//w,i%w]
    def ssd(hInd, patchPos):
        pX, pY = patchPos[1:]
        holePos = indToPos(hInd)
        fImg = padded[0][holePos[0]:holePos[0]+r+r, holePos[1]:holePos[1]+r+r,:3]
        sImg = padded[patchPos[0]][pX:pX+r+r,pY:pY+r+r,:3]
        return np.sum((fImg-sImg)**2)

    holes = np.transpose(np.nonzero(img[:,:,3] != 255)) # list of transparent pixels
    minbx = np.min(holes[:,0]) - r
    maxbx = np.max(holes[:,0]) + r
    minby = np.min(holes[:,1]) - r
    maxby = np.max(holes[:,1]) + r
    samples = [] # list of indices of valid sample patches
    for x in range(0,img.shape[0]):
        if x < minbx or x > maxbx:
            for y in range(0,img.shape[1]):
                if y < minby or y > maxby:
                    samples.append((0, x, y))
    for i in range(0,len(refs)):
        ref = refs[i]
        for x in range(0,ref.shape[0]):
            for y in range(0,ref.shape[1]):
                samples.append((i+1, x, y))
    h = maxbx-minbx
    w = maxby-minby
    n = h*w
    nnf = []
    for i in range(0,n): # randomize
        nnf.append(samples[np.random.choice(len(samples))])
    nnf = np.array(nnf)
    subimg = np.zeros((h,w,4))
    for i in range(0,n):
        subFill(i)
    nnd = []
    for i in range(0,n):
        nnd.append(ssd(i, nnf[i]))

    def improve(hInd, patchPos):
        iNum = patchPos[0]
        if iNum == 0:
            xMax, yMax = img.shape[:-1]
        else:
            xMax, yMax = refs[iNum-1].shape[:-1]
        pX = min(max(patchPos[1],0),xMax-1)
        pY = min(max(patchPos[2],0),yMax-1)
        patchPos = [iNum, pX, pY]
        candD = ssd(hInd, patchPos)
        if nnd[hInd] > candD:
            nnd[hInd] = candD
            nnf[hInd] = patchPos
            subFill(hInd)

    for i in range(0,iters):
        if i % 2:
            s = 1
        else:
            s = -1
        if i != 0:
            nnf = nnf[::-1]
            nnd = nnd[::-1]
        for j in range(0,n):
            # propagate
            nInd = min(max(j + s, 0), n-1)
            improve(j, (nnf[nInd,0], nnf[nInd,1]-s, nnf[nInd,2]))

            nInd = min(max(j + s*w, 0), n-1)
            improve(j, (nnf[nInd][0], nnf[nInd,1], nnf[nInd,2]-s))
            # search
            x, y = indToPos(j)
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
            print("pixel", str(j+1) + "/" + str(n), "iter", str(i+1) + "/" + str(iters))

    for i in range(0,n):
        fill(i)
    out = Image.fromarray(img)
    out.save("../data/patchOut.png")
    debug = Image.fromarray(subimg.astype(np.uint8))
    debug.save("../data/debug.png")

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