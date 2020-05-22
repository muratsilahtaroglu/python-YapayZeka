# TensorFlow and tf.keras
import tensorflow as tf# eğitimleri yapar
from tensorflow import keras

# Helper libraries
import numpy as np# matris dizisi tutar ve bazı matamatiksel işleri yapar
import matplotlib.pyplot as plt#plot olarak ekrana bastırır

# print(tf.__version__)
# print(np.__version__)
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()#resim datalarını yükler
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#resimler yüklendimi diye bakılabilir
print(train_images.shape)
plt.figure()
plt.imshow(train_images[2])
plt.colorbar()
plt.grid(False)
plt.show()
#herbir resmin herbir pikselini 0-255 aralığından 0-1 arasına çeker
train_images = train_images / 255.0
test_images = test_images / 255.0
print(type(train_images))
plt.figure(figsize=(10,10))#resimlerin arsındaki boşluk ayarlar

for i in range(25):#25 tane resim basar
    plt.subplot(5,5,i+1)#5*5 basar
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
#Neuranları oluşturur burda 3 katmanlı nöron vardır
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),#28*28 tane nöron var
    keras.layers.Dense(128, activation='relu'),#relu fonsiyonu kullanılımış sigmoid de kullanılabilr.
    keras.layers.Dense(10)#enson 10 çeşit sınıfa eşleme yapar
])
#geri yayılım(back propagation) işlemini yapar
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)#kaç defa eğitim yapılacağı seçilir.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)#verbose raporlama yapar. 0,1 veya 2 değeri alır 2 önerilir.

print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

i =100
print(predictions[i])

print("Forecast result:"+class_names[np.argmax(predictions[i])])

print("Orginal:"+class_names[test_labels[i]])


plt.figure()
plt.imshow(test_images[i])
plt.colorbar()
plt.grid(False)
plt.show()
