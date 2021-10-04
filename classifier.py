import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
DATA_PATH = 'data2'
path_labels = 'labels.csv'
imgs = []
img_id = []
test_ratio = 0.2
val_ratio = 0.2
img_dimensions = (32, 32, 3)
dir_list = os.listdir(DATA_PATH)
print(f'Detectare pentru {len(dir_list)} cifre')
number_of_numbers = len(dir_list)
for x in range(0, number_of_numbers):
 picture_list = os.listdir(f'{DATA_PATH}/{x}')
 for y in picture_list:
 current_image = cv2.imread(f'{DATA_PATH}/{x}/{y}')
 current_image = cv2.resize(current_image, (img_dimensions[0],
img_dimensions[1]))
 imgs.append(current_image)
 img_id.append(x)
 print(x, end=" ")
print(" ")

imgs = np.array(imgs)
img_id = np.array(img_id)
X_train, X_test, Y_train, Y_test = train_test_split(imgs, img_id,
test_size=test_ratio)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train,
test_size=val_ratio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)
number_of_samples = []
for x in range(0, number_of_numbers):
 print(len(np.where(Y_train == x)[0]))
 number_of_samples.append(len(np.where(Y_train == 0)[0]))
print(number_of_samples)
plt.figure(figsize=(10, 5))
plt.bar(range(0, number_of_numbers), number_of_samples)
plt.title("Numarul de imagini in fiecare director")
plt.xlabel("Cifra")
plt.ylabel("Numarul de imagini")
plt.show()
def preProcessing(img):
 img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 img = cv2.equalizeHist(img)
 img = img / 255
 return img
# img speficica
img_nou = preProcessing(X_train[30])
img_nou = cv2.resize(img_nou, (300, 300))
cv2.imshow("PreProcessed", img_nou)
cv2.waitKey(0)
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1],
X_validation.shape[2], 1)
data_generator = ImageDataGenerator(width_shift_range=0.1,
height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
data_generator.fit(X_train)
Y_train = to_categorical(Y_train, number_of_numbers)
Y_test = to_categorical(Y_test, number_of_numbers)
Y_validation = to_categorical(Y_validation, number_of_numbers)
def modelul():
 number_of_filters = 60
 size_of_filter1 = (5, 5)
 size_of_filter2 = (3, 3)
 size_of_pool = (2, 2)
 number_of_node = 500
 model = Sequential()
 model.add((Conv2D(number_of_filters, size_of_filter1,
input_shape=(img_dimensions[0], img_dimensions[1], 1),
 activation="relu")))
 model.add((Conv2D(number_of_filters, size_of_filter1, activation="relu")))
 model.add(MaxPooling2D(pool_size=size_of_pool))
 model.add((Conv2D(number_of_filters // 2, size_of_filter2, activation="relu")))
 model.add((Conv2D(number_of_filters // 2, size_of_filter2, activation="relu")))
 model.add(MaxPooling2D(pool_size=size_of_pool))
 model.add(Dropout(0.5))
 model.add(Flatten())
 model.add(Dense(number_of_node, activation="relu"))
 model.add(Dropout(0.5))
 model.add(Dense(number_of_numbers, activation="softmax"))
 model.compile(Adam(lr=0.001), loss="categorical_crossentropy",
metrics=['accuracy'])
 return model
model = modelul()
print(model.summary())
batch_size_val = 32
epoch_num = 10
steps_per_epoch = len(X_train)//batch_size_val
history = model.fit(data_generator.flow(X_train, Y_train, batch_size=
batch_size_val), steps_per_epoch=steps_per_epoch, epochs=epoch_num,
validation_data=(X_validation, Y_validation), shuffle=1)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['antrenare', 'validare'])
plt.title('Functia de pierdere - Loss')
plt.ylabel('Loss')
plt.xlabel('Iteratie')
plt.grid()
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history["val_accuracy"])
plt.legend(['antrenare', 'validare'])
plt.title('Precizia modelului')
plt.xlabel('Iteratie')
plt.ylabel('Precizie')
plt.grid()
plt.show()
score = model.evaluate(X_test, Y_test, verbose=0)
print("Test Score = ", score[0])
print("Test Accuracy = ", score[1])
model.save("data2.h5")
del model
