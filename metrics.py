from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from operator import truediv


class Recall(Layer):
    '''Compute recall over all batches.
    # Arguments
        name: String, name for the metric.
        class_ind: Integer, class index.
    '''
    def __init__(self, name='recall', class_ind=1):
        super(Recall, self).__init__(name=name)
        self.true_positives = K.variable(value=0, dtype='float32')
        self.total_positives = K.variable(value=0, dtype='float32')
        self.class_ind = class_ind

    def reset_states(self):
        K.set_value(self.true_positives, 0.0)
        K.set_value(self.total_positives, 0.0)

    def __call__(self, y_true, y_pred):
        '''Update recall computation.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            Overall recall for the epoch at the completion of the batch.
        '''
        # Batch
        y_true, y_pred = _slice_by_class(y_true, y_pred, self.class_ind)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        total_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        # Current
        current_true_positives = self.true_positives * 1
        current_total_positives = self.total_positives * 1
        # Updates
        updates = [K.update_add(self.true_positives, true_positives),
                   K.update_add(self.total_positives, total_positives)]
        self.add_update(updates, inputs=[y_true, y_pred])
        # Compute recall
        return (current_true_positives + true_positives) / \
               (current_total_positives + total_positives + K.epsilon())


class Precision(Layer):
    '''Compute precision over all batches.
    # Arguments
        name: String, name for the metric.
        class_ind: Integer, class index.
    '''
    def __init__(self, name='precision', class_ind=1):
        super(Precision, self).__init__(name=name)
        self.true_positives = K.variable(value=0, dtype='float32')
        self.pred_positives = K.variable(value=0, dtype='float32')
        self.class_ind = class_ind

    def reset_states(self):
        K.set_value(self.true_positives, 0.0)
        K.set_value(self.pred_positives, 0.0)

    def __call__(self, y_true, y_pred):
        '''Update precision computation.
        # Arguments
            y_true: Tensor, batch_wise labels
            y_pred: Tensor, batch_wise predictions
        # Returns
            Overall precision for the epoch at the completion of the batch.
        '''
        # Batch
        y_true, y_pred = _slice_by_class(y_true, y_pred, self.class_ind)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        pred_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        # Current
        current_true_positives = self.true_positives * 1
        current_pred_positives = self.pred_positives * 1
        # Updates
        updates = [K.update_add(self.true_positives, true_positives),
                   K.update_add(self.pred_positives, pred_positives)]
        self.add_update(updates, inputs=[y_true, y_pred])
        # Compute recall
        return (current_true_positives + true_positives) / \
               (current_pred_positives + pred_positives + K.epsilon())


class F1(Layer):
    """Create a metric for the model's F1 score calculation.
    The F1 score is the harmonic mean of precision and recall.
    """

    def __init__(self, name='f1', class_ind=1):
        super().__init__(name=name)
        self.recall = Recall(class_ind=class_ind)
        self.precision = Precision(class_ind=class_ind)
        self.class_ind = class_ind

    def reset_states(self):
        """Reset the state of the metrics."""
        self.precision.reset_states()
        self.recall.reset_states()

    def __call__(self, y_true, y_pred):
        pr = self.precision(y_true, y_pred)
        rec = self.recall(y_true, y_pred)
        return 2 * truediv(pr * rec, pr + rec + K.epsilon())


def _slice_by_class(y_true, y_pred, class_ind):
    ''' Slice the batch predictions and labels with respect to a given class
    that is encoded by a categorical or binary label.
    #  Arguments:
        y_true: Tensor, batch_wise labels.
        y_pred: Tensor, batch_wise predictions.
        class_ind: Integer, class index.
    # Returns:
        y_slice_true: Tensor, batch_wise label slice.
        y_slice_pred: Tensor,  batch_wise predictions, slice.
    '''
    # Binary encoded
    if y_pred.shape[-1] == 1:
        y_slice_true, y_slice_pred = y_true, y_pred
    # Categorical encoded
    else:
        y_slice_true, y_slice_pred = y_true[..., class_ind], y_pred[..., class_ind]
    return y_slice_true, y_slice_pred
