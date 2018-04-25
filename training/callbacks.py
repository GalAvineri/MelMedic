from os.path import join
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow import Summary


def define_callbacks(results_dir, overfit=False, verbose=0, early_stopper_patience=30, lr_reducer_patience=10):
    """
    Defines callbacks to use during the training process
    :param results_dir: Directory to keep reports about the training process,
        such as the best model so far and history of measurements
    :return A list of callbacks
    """

    monitor = 'loss' if overfit else 'val_loss'
    mode = 'min'

    model_checkpoint_dir = join(results_dir, 'Model')
    tensorboard_events_dir = join(results_dir, 'Tensorboard')

    model_checkpointer = ModelCheckpoint(filepath=model_checkpoint_dir, monitor=monitor, verbose=verbose,
                                         save_best_only=True, mode=mode)

    early_stopper = EarlyStopping(monitor=monitor, min_delta=0, patience=early_stopper_patience, verbose=verbose, mode=mode)

    lr_reducer = ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=lr_reducer_patience, verbose=verbose, mode=mode, epsilon=1e-4,
                                   cooldown=0, min_lr=0)

    tensorboard = TensorBoardBatched(log_dir=tensorboard_events_dir, write_graph=True, log_every_batch=True)

    return [model_checkpointer, early_stopper, lr_reducer, tensorboard]


class TensorBoardBatched(TensorBoard):
    def __init__(self, log_every_batch=False, **kwargs):
        super().__init__(**kwargs)
        self.log_every_batch = log_every_batch
        self.batch_counter = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        if self.log_every_batch:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.batch_counter)
            self.writer.flush()

        super().on_batch_end(batch, logs)

