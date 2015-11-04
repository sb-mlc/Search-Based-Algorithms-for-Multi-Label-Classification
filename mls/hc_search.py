import numpy as np
import sys
from sklearn import cross_validation
from mls.hc_prediction import predict_best_output
import time
from flipbit import FlipBit
from measures import calculate_loss
from scipy.sparse import vstack
from copy import deepcopy

from mls.hc_utils import construct_sparse_attributes


class HCSearchRanking:

    def __init__(self, initial_h_classifier, initial_c_classifier, scoring_function, depth_of_search,
                 number_of_labels, initial_br=None, h_reduction=1.0):
        self.h_classifier = initial_h_classifier
        self.c_classifier = initial_c_classifier
        self.depth_of_search = depth_of_search
        self.number_of_labels = number_of_labels
        if scoring_function not in {'hamming', 'f_measure', 'accuracy'}:
            raise ValueError("Scoring function {0} is not supported.".format(scoring_function))
        self.scoring_function = scoring_function
        self.initial_br = initial_br
        self.h_reduction = h_reduction

    def fit(self, x_train, y_train, cv=False):
        final_h_classifier = deepcopy(self.h_classifier)
        final_h_classifier = self.fit_heuristic_h(final_h_classifier, x_train, y_train, verbose=1)

        start_c_construction_time = time.clock()

        c_final_examples = []
        c_final_labels = []

        if cv:
            kf = cross_validation.KFold(len(x_train), n_folds=10, random_state=11)
            for train_index, test_index in kf:
                x_heuristic, x_cost = x_train[train_index], x_train[test_index]
                y_heuristic, y_cost = y_train[train_index], y_train[test_index]

                h_i_classifier = deepcopy(self.h_classifier)
                h_i_classifier = self.fit_heuristic_h(h_i_classifier, x_heuristic, y_heuristic)

                c_training_x, c_training_y = self.generate_examples_c(h_i_classifier, x_cost, y_cost, verbose=1)
                c_final_examples.extend(c_training_x)
                c_final_labels.extend(c_training_y)
        else:
            c_final_examples, c_final_labels = self.generate_examples_c(final_h_classifier, x_train, y_train, verbose=0)

        stop_c_construction_time = time.clock()
        print("C total construction time: {0:.4f}, Examples: {1}".format(stop_c_construction_time - start_c_construction_time,
                                                                         len(c_final_examples)))

        c_classifier = deepcopy(self.c_classifier)
        c_fit_start_time = time.clock()
        final_c_classifier = c_classifier.fit(vstack(c_final_examples, format='csr'), c_final_labels)
        c_fit_end_time = time.clock()
        print("C heuristic train time: {0:.4f}".format(c_fit_end_time - c_fit_start_time))

        self.h_classifier = final_h_classifier
        self.c_classifier = final_c_classifier

    def fit_heuristic_h(self, h_classifier, x_train, y_train, verbose=0):
        h_construction_start_time = time.clock()

        h_training_x = []
        h_training_y = []

        for i in xrange(len(x_train)):
            flipbit = FlipBit(x_train[i], self.number_of_labels, self.scoring_function, 'train',
                              initial_br=self.initial_br, true_output=y_train[i], reduction=self.h_reduction)
            flipbit.greedy_search(self.depth_of_search)  # Run greedy_search to construct H training examples

            h_training_x.extend(flipbit.get_training_examples())
            h_training_y.extend(flipbit.get_training_labels())

        h_construction_end_time = time.clock()
        if verbose > 0:
            print("H construction time: {0:.4f}, Examples: {1}".format(h_construction_end_time - h_construction_start_time,
                                                                       len(h_training_x)))
        h_classifier.fit(vstack(h_training_x, format='csr'), h_training_y)
        h_fit_end_time = time.clock()
        if verbose > 0:
            print("H heuristic train time: {0:.4f}".format(h_fit_end_time - h_construction_end_time))

        return h_classifier

    def generate_examples_c(self, fitted_h_classifier, x_train, y_train, verbose=0):
        c_start_time = time.clock()

        c_training_x = []
        c_training_y = []

        for i in xrange(len(x_train)):
            flipbit = FlipBit(x_train[i], self.number_of_labels, self.scoring_function, 'test',
                              initial_br=self.initial_br, fitted_classifier=fitted_h_classifier)
            outputs = flipbit.greedy_search(self.depth_of_search)  # Get outputs using fitted H heuristic

            best_loss = sys.maxint
            best_output = None
            for output in outputs:
                loss = calculate_loss(self.scoring_function, output, y_train[i], self.number_of_labels)
                if loss < best_loss:
                    best_loss = loss
                    best_output = output

            output_1_attributes = construct_sparse_attributes(x_train[i], best_output)
            for output in outputs:
                if best_output == output:
                    continue
                loss_2 = calculate_loss(self.scoring_function, output, y_train[i], self.number_of_labels)
                if best_loss == loss_2:
                    continue
                output_2_attributes = construct_sparse_attributes(x_train[i], output)

                c_training_x.append(output_1_attributes - output_2_attributes)
                c_training_y.append(np.sign(best_loss - loss_2))

                c_training_x.append(output_2_attributes - output_1_attributes)
                c_training_y.append(np.sign(loss_2 - best_loss))

        c_construction_end_time = time.clock()
        if verbose > 0:
            print("C construction time: {0:.4f}, Examples: {1}".format(c_construction_end_time - c_start_time,
                                                                       len(c_training_x)))
        return c_training_x, c_training_y

    def predict(self, x_test, y_test):
        y_predicted = []
        h_acc_sum = 0.0
        hc_loss_sum = 0.0
        for i in xrange(len(x_test)):

            flipbit = FlipBit(x_test[i], self.number_of_labels, self.scoring_function, 'test',
                              initial_br=self.initial_br, fitted_classifier=self.h_classifier)
            outputs = flipbit.greedy_search(self.depth_of_search)

            # Calculate prediction loss
            true_output = y_test[i]
            min_loss = 1.0
            for output in outputs:
                loss = calculate_loss(self.scoring_function, output, true_output, self.number_of_labels)
                if loss < min_loss:
                    min_loss = loss
                if (true_output == output).all():
                    h_acc_sum += 1.0
                    break

            hc_loss_sum += min_loss

            best_output = predict_best_output(x_test[i], outputs, self.c_classifier)
            y_predicted.append(best_output)
            # print y_test[i], outputs, best_output

        print('Loss 0/1 of H: {0}'.format(1.0 - (h_acc_sum/len(x_test))))
        print("Loss {0} of H: {1}".format(self.scoring_function, hc_loss_sum/len(x_test)))
        return y_predicted
