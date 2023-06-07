import tensorflow as tf
import numpy as np

# load images from directory
TEST_IMAGE = "/home/aorus01/projects/VMMR/TEST_MITSUBISHI_FUSO/29_0_40_23.jpg"
img = tf.keras.utils.load_img(TEST_IMAGE, target_size=(224, 224))


# load model
MODEL = "/home/aorus01/projects/VMMR/ResNet/my_model/"
model = tf.keras.models.load_model(MODEL)

# prediction
img_array = tf.keras.utils.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0) # Create a batch
img_array = np.array([img_array])

predictions = model.predict(img_array)
print(predictions)
score = tf.nn.softmax(predictions[0])
print(np.argmax(score))
print(score)


