import numpy as np
import cv2
import time
import random

cap = cv2.VideoCapture(0)

state = 0
states = ['', 'ok', '', 'good', '', 'bad', '', 'v', '', 'hand', '', 'none']

inputs = {
    'ok': [],
    'good': [],
    'bad': [],
    'v': [],
    'hand': [],
    'none': []
}

while(True):
    ret, frame = cap.read()

    source = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    input = cv2.resize(source, (int(720 * 0.25), int(480 * 0.25)))
    source = cv2.imshow('frame', source)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        state += 1
        if (state >= 10):
            break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if (states[state] in inputs):
        inputs[states[state]].append(input)
        print("{} sign has {} frames".format(states[state], len(inputs[states[state]])))

cap.release()
cv2.destroyAllWindows()

if (state >= 12):
    for sign in inputs:
        for image in inputs[sign]:
            url = "./data/{}/{}-{}.jpg".format(sign, time.time(), random.random())
            cv2.imwrite(url, image)
print("Done")
