from PIL import Image
import numpy as np

spongebasic = np.array(Image.open("data/basicOut.png"))
spongerefs = np.array(Image.open("data/refsOut.png"))
groundtruth = np.array(Image.open("data/spongeinput.jpg"))

n = spongebasic.size
print(20*np.log(255) - 10*np.log(1/n * np.sum((spongebasic - groundtruth)**2)))
print(20*np.log(255) - 10*np.log(1/n * np.sum((spongerefs - groundtruth)**2)))

spongebasic = np.array(Image.open("data/basicOut1.png"))
spongerefs = np.array(Image.open("data/refsOut3.png"))
groundtruth = np.array(Image.open("data/patchinput.png"))

n = spongebasic.size
print(20*np.log(255) - 10*np.log(1/n * np.sum((spongebasic - groundtruth)**2)))
print(20*np.log(255) - 10*np.log(1/n * np.sum((spongerefs - groundtruth)**2)))