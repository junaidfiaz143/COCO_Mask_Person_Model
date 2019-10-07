from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# load model
model = load_model("models/coco_person_mask_model.h5")

os.system("cls")

# input_name = "normal.jpeg"
input_name = "input_3.jpg"

image_size = 256

# load input image
input_image = cv2.imread(input_name)
# resize input image on which model is trained
input_image = cv2.resize(input_image, (image_size, image_size))

# expand input dimension to add dimension for batch_size
input_image = np.expand_dims(input_image, axis=0)

print("Input Image Name:", input_name)
print("Input Image Dimension: ", input_image.shape)

# get predictions from model
predictions = model.predict(input_image)[0]

cv2.imwrite("output_3.png", predictions*255)

plt.imshow(np.reshape(predictions, (image_size, image_size)), cmap="gray")
plt.show()

print("OK")