from data.dataset import Dataset

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import numpy as np
import os
from os.path import join
batch_size = 32
epochs = 100


# # Count number of train and val samples
# train_num_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
# val_num_samples = sum([len(files) for r, d, files in os.walk(val_dir)])

# Create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# We would like to train only the newly added layers and freeze all lower layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# data
data_dir = join(os.pardir, 'Data', 'suites')
train = Dataset(join(data_dir, 'train'))
val = Dataset(join(data_dir, 'val'))
test = Dataset(join(data_dir, 'test'))

train_iter = train.dataset.shuffle(1000).batch(batch_size).repeat().prefetch(batch_size)
val_iter = val.dataset.shuffle(1000).batch(batch_size).repeat().prefetch(batch_size)
test_iter = test.dataset.batch(batch_size, drop_remainder=True).prefetch(batch_size)

# Training configurations
class_weights = class_weight.compute_class_weight('balanced', np.unique(train.labels), train.labels)
callbacks = [EarlyStopping(patience=10), ReduceLROnPlateau()]

# Train the model
model.fit(train_iter, validation_data=val_iter, epochs=epochs,
          steps_per_epoch=train.size // batch_size, validation_steps=val.size // batch_size,
          class_weight=class_weights,
          callbacks=callbacks
          )

# Evaluate the model
preds = model.predict(test_iter, steps=test.size // batch_size)
test_labels = test.labels
test_labesl = test_labels[:(test.size // batch_size) * batch_size] # truncate to the number of predictions made
preds = np.argmax(preds, axis=1)
print(classification_report(test_labels, preds))