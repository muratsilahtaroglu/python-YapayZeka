import numpy as np
from PIL import Image
import os

# l = []
# for i in range(50):
#     l.append(np.asarray([[x for x in range(256)] for y in range(256)]))
#
# t = np.asarray(l)
#
# print(t)
# print(type(t))

l = []
for (dirpath, dirnames, filenames) in os.walk(os.path.join("Image_Classification_Deneme", "DATA", "train")):
    for file in filenames:
        image = Image.open(os.path.join(dirpath, file))
        l.append(np.asarray(image))


t = np.asarray(l)

print(t)
print(type(t))
