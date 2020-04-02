import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('model')
model.summary()

classes = ['ok', 'good', 'bad', 'v', 'hand', 'none']
cap = cv2.VideoCapture(0)

def predict(image):
    image = image / 255.0
    prediction = model.predict(np.array([image]))
    prediction = prediction[0].tolist()
    index = prediction.index(max(prediction))
    return classes[index]

while(True):
    ret, frame = cap.read()

    source = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    input = cv2.resize(source, (int(720 * 0.25), int(480 * 0.25)))
    img = np.stack((input,)*3, axis=-1)
    source = cv2.imshow('frame', source)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(predict(img))
