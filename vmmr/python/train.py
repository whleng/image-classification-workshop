import tensorflow as tf

# Load image from directory
root = '' # Insert path of root folder here
TRAINING_FOLDER = root + "/TRAIN/"
VALIDATION_FOLDER = root + "/VAL/"
NUMBER_OF_CLASSES = 4

###########################################################################
# Training parameters to change
epochs = 10
learning_rate = 0.0005
###########################################################################

# Preparation of dataset

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=TRAINING_FOLDER,
    seed=123,
    image_size=(224, 224),
    batch_size=2,
    label_mode='categorical'
)
print(train_ds.class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=VALIDATION_FOLDER,
    seed=123,
    image_size=(224, 224),
    batch_size=2,
    label_mode='categorical'
)

###########################################################################
# Define model
# Refer here for other parameters to be added: 
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(224, 224, 3),
    weights='imagenet'
)
base_model.trainable = False

###########################################################################

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=tf.keras.metrics.categorical_accuracy)
print(model.summary())


# Training
history = model.fit(train_ds,
          epochs=epochs,
          validation_data=val_ds)

# Output model
model.save("my_model")

