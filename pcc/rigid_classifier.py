import numpy as np


class RigidClassifier:

    def __init__(self, result_label):
        self.result_label = result_label

    def predict(self, x_test):
        return [self.result_label] * len(x_test)

    def predict_proba(self, x_test):
        result = []
        for i in xrange(len(x_test)):
            if self.result_label == 0:
                result.append([1.0, 0.0])
            elif self.result_label == 1:
                result.append([0.0, 1.0])
        return np.asarray(result)



