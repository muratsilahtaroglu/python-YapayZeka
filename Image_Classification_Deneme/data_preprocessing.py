from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt#plot olarak ekrana bastırır
import os

class_name = []

def load_data():
    train_images = []
    train_labels = []
    for (dirpath, dirnames, filenames) in os.walk(os.path.join("DATA", "train")):
        for file in filenames:
            image = Image.open(os.path.join(dirpath, file))
            train_images.append(np.asarray(image))#matamatiksel işlem yapabilmek için image i matrise dönüştürdük
            # image = image.resize((256))
            #print([x for x in image.__dir__() if str(x).__contains__("size")])
            #print(image.size)
            label=file[:file.index("-")]
            if not class_name.__contains__(label):
                class_name.append(label)
                train_labels.append(len(class_name)-1)
            else:
                index = class_name.index(label)
                train_labels.append(index)


    test_images = []
    test_labels=[]
    for (dirpath, dirnames, filenames) in os.walk(os.path.join("DATA", "test")):
        for file in filenames:
            try:
                image = Image.open(os.path.join(dirpath, file))
                test_images.append(np.asarray(image))
                index = class_name.index(file[:file.index("-")])
                test_labels.append(index)
                # test_labels.append(file[:file.index("-")])
            except:
                print("hata")
    print(type(train_images))
    IMGSIZE = 255
    return (np.array(train_images).reshape(-1, IMGSIZE, IMGSIZE, 1), train_labels), (np.array(test_images).reshape(-1, IMGSIZE, IMGSIZE, 1), test_labels)

(train_images, train_labels), (test_images, test_labels)=load_data()



# def grafikshow(data):
#     plt.figure()
#     plt.imshow(data)
#     plt.colorbar()
#     plt.grid(False)
#     plt.show()
# x=5
# grafikshow(train_images[x])
# print(class_name[train_labels[x]])
# grafikshow(test_images[x])
# print(class_name[test_labels[x]])
print(class_name)
