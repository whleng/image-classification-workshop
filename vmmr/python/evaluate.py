import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report

# load images from directory
TEST_FOLDER = "/media/wenhui/SSD_02/VMMR_WenHui/all_data/data_split/make/TEST/"

# load model
MODEL = '/home/wenhui/vmmr-project/vmmr/local/keras_image_classifier/test_resnet_make_2'
model = tf.keras.models.load_model(MODEL)
        
# Pred
preds = []
actual = []

# all_classes = sorted(['VOLVO', 'HINO', 'SINOTRUK_A7', 'HINO_SH1E', 'MITSUBISHI'])
all_classes = sorted(['VOLVO', 'HINO', 'NISSAN', 'MITSUBISHI'])

for i in range(len(all_classes)):
    category = all_classes[i]
    class_folder = os.path.join(TEST_FOLDER, category)

    img_array = []

    for file in os.listdir(class_folder):
        img_path = os.path.join(TEST_FOLDER, class_folder, file)

        img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        new_img = tf.keras.utils.img_to_array(img)

        if len(img_array) == 0:
            img_array = np.array([new_img])
        else:
            img_array = np.vstack((img_array, [new_img]))
        actual.append(i)
    

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions)
    pred_class = np.argmax(score, axis=1)
    preds += pred_class.flatten().tolist()


print(classification_report(actual, preds, target_names=all_classes))
