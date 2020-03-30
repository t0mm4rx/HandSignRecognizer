import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import os

classes = ['ok', 'good', 'bad', 'v', 'hand']

inputs_dict = {}

for classe in classes:
    inputs_dict[classe] = []
    for file in os.listdir('./data/{}/'.format(classe)):
        inputs_dict[classe].append(cv2.imread('./data/{}/{}'.format(classe, file)) / 255.0)

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

print(inputs.shape)
print(outputs.shape)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(120, 180, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(inputs, outputs, epochs=10)
