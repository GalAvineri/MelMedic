from training.training import train_model
from auxiliries import io

import os
from os.path import join

data_dir = join(os.pardir, 'Data')
results_dir = join(os.pardir, 'Results')
io.create_if_none(results_dir)

train_model(train_dir=join(data_dir, 'train'),
            val_dir=join(data_dir, 'val'),
            results_dir=results_dir)
