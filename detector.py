import numpy as np
import cv2
from keras.models import load_model
from .train import preProcessing
width = 640
height = 480
threshold = 0.65
camera_port = 0
capture = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
capture.set(3, width)
capture.set(4, height)
model = load_model("data2.h5")
while True:
 success, imgOriginal = capture.read()
 img = np.asarray(imgOriginal)
 img = cv2.resize(img, (32, 32))
 img = preProcessing(img)
 cv2.imshow("Preprocessed Image", img)
 img = img.reshape(1, 32, 32, 1)
 classIndex = int(model.predict_classes(img))
 predictions = model.predict(img)
 probVal = np.amax(predictions)
 if probVal > threshold:
 cv2.putText(imgOriginal, f'{classIndex}: {probVal * 100}%', (50, 50),
cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
 cv2.imshow("Original Image", imgOriginal)
 if cv2.waitKey(20) & 0xFF == ord('q'):
 break
capture.release()
cv2.destroyAllWindows()
