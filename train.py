import numpy as np
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models
import os
import random

classes = ['ok', 'good', 'bad', 'v', 'hand', 'none']

inputs_dict = {}

for classe in classes:
    inputs_dict[classe] = []
    for file in os.listdir('./data/{}/'.format(classe)):
        input = cv2.imread('./data/{}/{}'.format(classe, file)) / 255.0
        inputs_dict[classe].append(input)

inputs = []
outputs = []

for i, classe in enumerate(classes):
    out = []
    for a in range(len(classes)):
        out.append(float(a == i))
    for a in inputs_dict[classe]:
        inputs.append(a)
        outputs.append(np.array(out))

inputs = np.array(inputs)
outputs = np.array(outputs)

train_x = []
train_y = []
test_x = []
test_y = []

for i in range(len(inputs)):
    if (random.random() < 0.95):
        train_x.append(inputs[i])
        train_y.append(outputs[i])
    else:
        test_x.append(inputs[i])
        test_y.append(outputs[i])

train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(120, 180, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=6)

model.save('model')

guess = model.predict(np.array([inputs[1]]))
print(guess)

plt.figure()
plt.plot(history.history['accuracy'], color='red', label='Accuracy')
plt.plot(history.history['val_accuracy'], color='blue', label='Val Accuracy')
plt.legend()

plt.figure()
plt.plot(history.history['loss'], color='red', label='Loss')
plt.plot(history.history['val_loss'], color='blue', label='Val Loss')
plt.legend()
plt.show()
