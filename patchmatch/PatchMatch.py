import numpy as np
from PIL import Image
import sys
import os

iters = 5

def synthPatchMatch(img, refs, winsize=7):
    r = winsize // 2
    padded = [np.pad(ref, r, mode="symmetric") for ref in refs]
    padded.insert(0, np.pad(img, r, mode="symmetric"))
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
    holes = np.nonzero(img[:,:,3] == 0) # list of transparent pixels with nontransparent pixels within window
    n = holes.shape[0]
    nnf = []
    for i in range(0,n): # randomize
        nnf.append(np.random.choice(samples))
    def fill():
        
    nnd = []
    def ssd(holePos, patchPos):
        fImg = padded[0][holePos[0]-r:holePos[0]+r, holePos[1]-r:holePos[1]+r,:2]
        sImg = padded[patchPos[0]][patchPos[1]-r:patchPos[1]+r,patchPos[2]-r:patchPos[2]+r,:2]
        return np.sum((fImg-sImg)**2)
    for i in range(0,n):
        nnd.append(ssd(holes[i], nnf[i]))
    for i in range(0,iters):
        if iter % 2:
            shift = -1
        else:
            shift = 1
        nnf = nnf[::-1]
        nnd = nnd[::-1]
        # propgate
        # search
        


if __name__ == "__main__":
    img = Image.open(sys.argv[1])
    dir = sys.argv[2]
    refs = []
    for file in os.listdir(dir):
        refs.append(Image.open(os.path.join(dir, file)))
    if len(sys.argv) > 3:
        winsize = sys.argv[3]
        if not winsize % 2:
            winsize += 1
    else:
        winsize = 7