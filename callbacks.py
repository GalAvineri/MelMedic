from tensorflow.keras.callbacks import TensorBoard
from tensorflow import Summary


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

