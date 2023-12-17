import csv
import os
import numpy as np
import cv2
from keras.models import load_model
import imageio

model = load_model('Autopilot.h5')


def keras_predict(model_, image):
    processed = keras_process_image(image)
    steering_angle_ = float(model_.predict(processed, batch_size=1))

    steering_angle_ = steering_angle_ * 100
    return steering_angle_


def keras_process_image(img):
    image_x = 40
    image_y = 40
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


out = open("./data/night_result.csv", 'a', newline='')
csv_writer = csv.writer(out)
smoothed_angle = 0

i = 1
while i <= len(os.listdir("./data/night")):
    frame = imageio.imread("./data/night/night_" + str(i) + ".jpg", pilmode="RGB")
    gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
    steering_angle = keras_predict(model, gray)

    outputDict = [
        "night_" + str(i),
        str(steering_angle),
    ]
    csv_writer.writerow(outputDict)
    i = i + 1

out.close()
cv2.destroyAllWindows()
