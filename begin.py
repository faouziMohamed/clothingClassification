#!/usr/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


fashion_mnist = keras.datasets.fashion_mnist

#Each image is mapped to a single label. Since the class names are not included with the dataset, store them here to use later when plotting the images:
#class_names_en = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names_fr = ['T-shirt/Haut', 'Pantalon', 'Tricot', 'Robe', 'Manteau',
               'Sandale', 'Chemise', 'Basket', 'Sac', 'Bottine']

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Normalisation of data every pixel will be equal 0<pixel<1
train_images = train_images/255.0
test_images  = test_images/255.0  

#Divide and Reshape of the dataset
# - Train    data
# - Validate data
# - Test     data

##Validate data shape : ()             --> (10 000, 28, 28)
##Train    data shape : (60000, 28, 28)--> (50 000, 28, 28)
train_images, validate_images = train_images[10000:],train_images[:10000]
train_labels, validate_labels = train_labels[10000:],train_labels[:10000]

## Reshape data to add explicitly...
##...the shannel of color (Important for the convolutioning step):
#Train    images :(55000, 28, 28) --> (55000, 28, 28,1)
#Validate images :( 5000, 28, 28) --> ( 5000, 28, 28,1)
#test     images :(10000, 28, 28) --> (10000, 28, 28,1)
images_width  = train_images.shape[1]
images_height = train_images.shape[1]

train_images = train_images.reshape(train_images.shape[0], images_width, images_height, 1)
validate_images = validate_images.reshape(validate_images.shape[0], images_width, images_height, 1)
test_images = test_images.reshape(test_images.shape[0], images_width, images_height, 1)



#Build the model with 2 convD2 layers empiled
###Set up the layers
model = models.Sequential(
[
	Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',input_shape=(images_width,images_height,1)),
	MaxPooling2D(2,2),
	Dropout(0.3),

	Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'),
	MaxPooling2D(2,2),
	Dropout(0.3),

	Flatten(),
	Dense(256,activation='relu'),
	Dense(10,activation='softmax')
])

#Compile the model
model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Feed the model
model.fit(train_images,
		 train_labels,
		 batch_size=64,
		 epochs=10,
		 validation_data=(validate_images,validate_labels)
		 )

model.reset_metrics()
# Save the model
model.save('model_PFE_mnist.h5')
model.save('model_PFE_mnist_tf.keras_python.h5', save_format='tf')

classifications = model.predict(test_images)

label = 8452
name = test_labels[label]

#Getting the predicted value 
print(f'Probabities of the {label}th element : \n{classifications[label]}')
name_predicted = classifications[label].argmax()

#Test 
print(f'\nThis {label}th element is probably : {class_names_fr[name_predicted]}')
print(f'This prediction is {class_names_fr[name_predicted] == class_names_fr[name]}') #Shows if the prediction is true or false

test_images = test_images.reshape(test_images.shape[0], images_width, images_height)
plt.figure(figsize=(6,6))
plt.subplot(1,2,1)
plt.imshow(test_images[label])
plt.title(class_names_fr[name])

plt.subplot(1,2,2)
plt.imshow(test_images[label])

if class_names_fr[name_predicted] == class_names_fr[name]:
  plt.title(class_names_fr[name_predicted],c='blue')
else:
  plt.title(class_names_fr[name_predicted],c='red')

plt.show()
























"""
#Graph this to look at the full set of 10 class predictions.
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#Verify predictions
#With the model trained, you can use it to make predictions about some images.
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()



#Use the trained model
###Grab an image from the test dataset.
img = test_images[1]
print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

#Now predict the correct label for this image:
predictions_single = probability_model.predict(img)
print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
np.argmax(predictions_single[0])
"""
