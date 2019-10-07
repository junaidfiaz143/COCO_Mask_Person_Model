import os
import random
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

seed = 2019
random.seed = seed

class DataGen(keras.utils.Sequence):
  def __init__(self, ids, images_path, masks_path, batch_size=8, image_size=256):
    self.ids = ids
    self.images_path = images_path
    self.masks_path = masks_path
    self.batch_size = batch_size
    self.image_size = image_size

  def __load__(self, id_name):
    image_path = os.path.join(self.images_path, id_name)
    mask_path = os.path.join(self.masks_path, id_name)

    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (self.image_size, self.image_size))

    mask = cv2.imread(mask_path, -1)
    mask = cv2.resize(mask, (self.image_size, self.image_size))

    mask = np.expand_dims(mask, axis=-1)

    image = image/255.0
    mask = mask/255.0

    return image, mask

  def __getitem__(self, index):
    if (index+1)*self.batch_size > len(self.ids):
      self.batch_size = len(self.ids) - index*self.batch_size

    files_batch = self.ids[index*self.batch_size : (index+1)*batch_size]

    image = []
    mask = []

    for id_name in files_batch:
      _img, _mask = self.__load__(id_name)
      image.append(_img)
      mask.append(_mask)

    image = np.array(image)
    mask = np.array(mask)

    return image, mask

  def __len__(self):
    return int(np.ceil(len(self.ids)/float(self.batch_size)))

image_size = 256
batch_size = 8
epochs = 15

path = "./dataset_coco_person_mask/coco/"

images_path = os.path.join(path, "img/")
masks_path = os.path.join(path, "seg/")

train_ids = next(os.walk(images_path))[2]

# print(train_ids)

valid_data_size = 500

valid_ids = train_ids[: valid_data_size]
train_ids = train_ids[valid_data_size: ]

gen = DataGen(train_ids, images_path, masks_path, batch_size=batch_size, image_size=image_size)

image_batch, mask_batch = gen.__getitem__(0)
print(image_batch.shape, mask_batch.shape)

r =  random.randint(0, batch_size-1)

print("random numbers: ", r)

print("length of x: ", len(image_batch))

# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)

# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(image_batch[r])

# ax = fig.add_subplot(1, 2, 2)
# ax.imshow(mask_batch[r], cmap="gray")

# plt.savefig("mask.png")

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    p = keras.lyers.Dropout(0.5)(p)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = keras.lyers.Dropout(0.5)(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    c = keras.lyers.Dropout(0.5)(c)
    return c

def MaskModel():
    inputs = keras.layers.Input(shape=(image_size, image_size, 3))
    
    c1, p1 = down_block(inputs, 16)
    c2, p2 = down_block(p1, 32) 
    c3, p3 = down_block(p2, 64) 
    c4, p4 = down_block(p3, 128) 
    c5, p5 = down_block(p3, 256)
    
    bn = bottleneck(p4, 512)
    
    u1 = up_block(bn, c5, 256)
    u2 = up_block(u1, c4, 128)
    u3 = up_block(u2, c3, 64)
    u4 = up_block(u3, c2, 32)
    u5 = up_block(u4, c1, 16)
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u5)

    model = keras.models.Model(inputs, outputs)
    return model

model = MaskModel()

lrate = 0.0001
decay = lrate/epochs

adam = keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay, amsgrad=False)

model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])

print(model.summary())

train_gen = DataGen(train_ids, images_path, masks_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, images_path, masks_path, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs)

# model.save("coco_person_mask_model.h5")