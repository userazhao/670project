import numpy as np
from PIL import Image
import sys
import os

iters = 5

def initPatchMatch(img, refs):
    h, w = img.shape[:2]
    rsmax = max(h,w)
    n = h*w
    refn = refs.shape[0]

    holes = np.transpose(np.nonzero(img[:,:,3] != 255))
    winsize = min(np.max(holes[:,0])-np.min(holes[:,0]),np.max(holes[:,1])-np.min(holes[:,1])) // 2
    if not winsize % 2:
        winsize += 1
    r = winsize//2
    minbx = max(np.min(holes[:,0]) - r,0)
    maxbx = min(np.max(holes[:,0]) + r + 1,h)
    minby = max(np.min(holes[:,1]) - r,0)
    maxby = min(np.max(holes[:,1]) + r + 1,w)

    def inBounds(x,y):
        return x >= 0 and x < h and y >= 0 and y < w
    def inHole(i):
        return img[i//w,i%w,3] != 255
    def inBbox(ref, x, y):
        return ref==0 and x > minbx and x <= maxbx and y > minby and y <= maxby
    def inTarget(i):
        return not inHole(i) and inBbox(0, i//w, i%w)
    def refRandVec(i, ref):
        x = np.random.randint(h)-i//w
        y = np.random.randint(w)-i%w
        while inBbox(ref,x,y):
            x = np.random.randint(h)-i//w
            y = np.random.randint(w)-i%w
        return [ref, x, y]
    def randVec(i):
        if not inTarget(i):
            return [0,0,0]
        return refRandVec(i, np.random.randint(refn))
    nnf = np.array([randVec(i) for i in range(0,n)])

    padimg = np.pad(img, ((r,r),(r,r),(0,0)), mode="symmetric")
    padref = np.pad(refs, ((0,0),(r,r),(r,r),(0,0)), mode="symmetric")
    def ssd(i, vec):
        if not inTarget(i):
            return -1
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
        if not inBbox(vec[0],x,y) and inBounds(x,y):
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
            if not inTarget(j):
                continue
            # propagate
            nInd = j + s
            if inTarget(nInd):
                improve(j, nnf[nInd])

            nInd = min(max(j + s*w, 0), n-1)
            if inTarget(nInd):
                improve(j, nnf[nInd])
            # planeshift
            for ref in range(0,refn):
                improve(j, refRandVec(j, ref))
            # search
            rs = rsmax
            while rs > 1:
                x = j//w + nnf[j,1]
                y = j%w + nnf[j,2]
                xmin = max(-rs, -x)
                ymin = max(-rs, -y)
                xmax = min(rs, h-x-1)
                ymax = min(rs, w-y-1)
                xc = np.random.randint(xmin, xmax)
                yc = np.random.randint(ymin, ymax)
                while inBbox(nnf[j,0],xc,yc):
                    xc = np.random.randint(xmin, xmax)
                    yc = np.random.randint(ymin, ymax)
                improve(j, [nnf[j,0],xc,yc])
                rs //= 2
            print("pixel", str(j) + "/" + str(n), "iter", str(i+1) + "/" + str(iters))

    # idea: check nnd for nonnegative values and use those corresponding in nnf as votes
    subimg = np.zeros(img.shape)
    votes = np.zeros((h,w,1))
    def getVotes(i):
        x0 = i//w
        y0 = i%w
        for xs in range(-winsize,winsize+1):
            for ys in range(-winsize,winsize+1):
                i1 = i+xs*w+ys
                if i1 >= 0 and i1 < n and nnd[i1] != -1:
                    x = x0 + nnf[i1,1]
                    y = y0 + nnf[i1,2]
                    ref = nnf[i1,0]
                    if not inBbox(ref, x, y) and inBounds(x,y):
                        subimg[x0,y0] += refs[ref,x,y]
                        votes[x0,y0] += 1
    for i in range(0,n):
        if inHole(i):
            getVotes(i)
            print("vote", str(i) + "/" + str(n))
    
    fill = subimg / votes
    out = Image.fromarray(fill.astype(np.uint8))
    out.save("../data/holeOut.png")

if __name__ == "__main__":
    img = np.array(Image.open(sys.argv[1]))
    dir = sys.argv[2]
    refs = [np.array(Image.open(os.path.join(dir, file))) for file in os.listdir(dir)]
    refs.insert(0, img)
    refs = np.array(refs)
    initPatchMatch(img, refs)