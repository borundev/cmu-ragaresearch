import tensorflow as tf


class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """

    def __init__(self, num_classes, print_f1=True, print_precision=False, print_recall=False,
                 **kwargs):
        super(ConfusionMatrixMetric, self).__init__(name='confusion_matrix_metric',
                                                    **kwargs)  # handles base args (e.g., dtype)
        self.num_classes = num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes, num_classes),
                                        initializer="zeros")
        self.print_f1 = print_f1
        self.print_precision = print_precision
        self.print_recall = print_recall

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true, y_pred))
        return self.total_cm

    def result(self):
        return self.process_confusion_matrix()

    def confusion_matrix(self, y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_pred = tf.argmax(y_pred, 1)
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32,
                                      num_classes=self.num_classes)
        return cm

    def process_confusion_matrix(self):
        "returns precision, recall and f1 along with overall accuracy"
        cm = self.total_cm
        diag_part = tf.linalg.diag_part(cm)
        precision = diag_part / (tf.reduce_sum(cm, 0) + tf.constant(1e-15))
        recall = diag_part / (tf.reduce_sum(cm, 1) + tf.constant(1e-15))
        f1 = 2 * precision * recall / (precision + recall + tf.constant(1e-15))
        return precision, recall, f1

    def fill_output(self, output):
        results = self.result()
        for i in range(self.num_classes):
            if self.print_precision:
                output['precision_{}'.format(i)] = results[0][i]
            if self.print_recall:
                output['recall_{}'.format(i)] = results[1][i]
            if self.print_f1:
                output['F1_{}'.format(i)] = results[2][i]