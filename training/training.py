from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import os
from os.path import join

img_height, img_width = 229, 229
batch_size = 32
epochs = 3


def train_model(train_dir, val_dir, results_dir):
    # Count number of train and val samples
    train_num_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
    val_num_samples = sum([len(files) for r, d, files in os.walk(val_dir)])

    # Create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # We would like to train only the newly added layers and freeze all lower layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # Define the data augmenters
    train_aug = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=30,
                                   horizontal_flip=True,
                                   vertical_flip=True)

    validation_aug = ImageDataGenerator(rescale=1. / 255)

    # Define the data generators
    train_generator = train_aug.flow_from_directory(train_dir,
                                                    target_size=[img_width, img_height],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

    validation_generator = validation_aug.flow_from_directory(val_dir,
                                                              target_size=[img_width, img_height],
                                                              batch_size=batch_size,
                                                              class_mode='categorical')

    # Train the model
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_num_samples // batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=val_num_samples // batch_size)

    model.save(join(results_dir, 'Models'))
