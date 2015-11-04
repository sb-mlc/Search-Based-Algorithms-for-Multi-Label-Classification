import time
import sys
from scipy.sparse import vstack

from flipbit import FlipBit
from measures import calculate_loss
from mls.hc_utils import construct_sparse_attributes


class HCSearchRegression:

    def __init__(self, initial_h_regressor, initial_c_regressor, scoring_function, depth_of_search, number_of_labels):
        self.h_regressor = initial_h_regressor
        self.c_regressor = initial_c_regressor
        self.depth_of_search = depth_of_search
        self.number_of_labels = number_of_labels
        if scoring_function not in {'hamming', 'f_measure', 'accuracy'}:
            raise ValueError("Scoring function {0} is not supported.".format(scoring_function))
        self.scoring_function = scoring_function

    def fit(self, x_train, y_train):
        c_training_examples = []
        c_training_scores = []
        h_training_examples = []
        h_training_scores = []

        start_time = time.clock()
        for i in xrange(len(x_train)):
            flipbit = FlipBit(x_train[i], self.number_of_labels, self.scoring_function, true_output=y_train[i])
            flipbit.greedy_search(self.depth_of_search)  # Run greedy_search to construct H training examples
            h_training_examples.extend(flipbit.get_training_examples())
            h_training_scores.extend(flipbit.get_training_scores())

        h_construction_end_time = time.clock()
        print("H training examples construction time: {0:.4f}".format(h_construction_end_time-start_time))

        self.h_regressor.fit(vstack(h_training_examples, format='csr'), h_training_scores)
        h_fit_end_time = time.clock()
        print("H heuristic train time: {0:.4f}".format(h_fit_end_time-h_construction_end_time))

        for i in xrange(len(x_train)):
            flipbit = FlipBit(x_train[i], self.number_of_labels, self.scoring_function, fitted_regressor=self.h_regressor)
            outputs = flipbit.greedy_search(self.depth_of_search)  # Get outputs using fitted H heuristic

            for j in xrange(len(outputs)):
                example = construct_sparse_attributes(x_train[i], outputs[j])
                score = calculate_loss(self.scoring_function, outputs[j], y_train[i], self.number_of_labels)
                c_training_examples.append(example)
                c_training_scores.append(score)

        c_construction_end_time = time.clock()
        print("C training examples construction time: {0:.4f}".format(c_construction_end_time-h_fit_end_time))

        self.c_regressor.fit(vstack(c_training_examples, format='csr'), c_training_scores)
        c_fit_end_time = time.clock()
        print("C heuristic train time: {0:.4f}".format(c_fit_end_time-c_construction_end_time))

        print("Training examples - Total: {0}, H: {1}, C: {2}".format(len(x_train), len(h_training_examples),
                                                                      len(c_training_examples)))

    def fit_simplified(self, x_train, y_train):
        c_training_examples = []
        c_training_scores = []
        h_training_examples = []
        h_training_scores = []

        start_time = time.clock()
        print "Number of examples in training set: " + str(len(x_train))
        for i in xrange(len(x_train)):
            flipbit = FlipBit(x_train[i], self.number_of_labels, self.scoring_function, true_output=y_train[i])
            outputs = flipbit.greedy_search(self.depth_of_search)
            h_training_examples.extend(flipbit.get_training_examples())
            h_training_scores.extend(flipbit.get_training_scores())

            for j in xrange(len(outputs)):
                example = construct_sparse_attributes(x_train[i], outputs[j])
                score = calculate_loss(self.scoring_function, outputs[j], y_train[i], self.number_of_labels)
                c_training_examples.append(example)
                c_training_scores.append(score)

        generating_end_time = time.clock()

        self.h_regressor.fit(vstack(h_training_examples, format='csr'), h_training_scores)
        print "Number of H regression learning examples: " + str(len(h_training_examples))

        self.c_regressor.fit(vstack(c_training_examples, format='csr'), c_training_scores)
        print "Number of C regression learning examples: " + str(len(c_training_examples))

        fit_time = time.clock()

        construction_time = (generating_end_time - start_time)
        learning_time = (fit_time - generating_end_time)
        print("Construction time: {0:.4f}, Learning HC time: {1:.4f}".format(construction_time, learning_time))

    def predict(self, x_test):
        y_predicted = []
        for example in x_test:
            flipbit = FlipBit(example, self.number_of_labels, scoring_function=self.scoring_function,
                              fitted_regressor=self.h_regressor)
            outputs = flipbit.greedy_search(self.depth_of_search)
            best_output = self.predict_best_output(example, outputs)
            y_predicted.append(best_output)
        return y_predicted

    def predict_best_output(self, example, outputs):
        best_score = sys.maxint
        best_output = None
        for output in outputs:
            attributes = construct_sparse_attributes(example, output)
            score = self.c_regressor.predict(attributes)
            if score < best_score:
                best_score = score
                best_output = output
        return best_output
