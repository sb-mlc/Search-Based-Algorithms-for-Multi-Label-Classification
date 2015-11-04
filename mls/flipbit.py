import numpy as np
import random
import sys

from measures import calculate_loss
from mls.hc_utils import construct_sparse_attributes
from mls.hc_prediction import predict_best_output


class FlipBit:

    def __init__(self, attr, number_of_labels, scoring_function, mode,
                 initial_br, true_output=None, fitted_classifier=None, reduction=1.0):
        self.attributes = attr
        self.number_of_labels = number_of_labels
        self.true_output = true_output
        self.classifier = fitted_classifier
        if initial_br is None:
            self.initial_state = [0] * number_of_labels
            self.flip_both_ways = False
        else:
            self.initial_state = initial_br.predict([attr])[0].tolist()
            self.flip_both_ways = True

        self.reduction = reduction
        self.h_training_examples = []
        self.h_training_labels = []
        self.mode = mode
        if scoring_function not in {'hamming', 'f_measure', 'accuracy'}:
            raise ValueError("Scoring function {0} is not supported.".format(scoring_function))
        self.scoring_function = scoring_function

    def successor_states(self, parent):
        children = []
        for i in xrange(len(parent)):
            child = parent[:]  # Copy labels of parent
            if self.flip_both_ways:
                child[i] = 1 - parent[i]  # Flip 0 to 1 or vice versa
            else:
                child[i] = 1  # Flip only to 1
            if child != parent:
                children.append(child)

        return children

    def greedy_search(self, depth_of_search):
        visited_nodes = self.visit_node_greedy(self.initial_state, [], depth_of_search)
        return visited_nodes

    def visit_node_greedy(self, node, already_visited, depth_of_search):
        # if node != self.initial_state:
        already_visited.append(node)
        depth_of_search -= 1

        if depth_of_search < 0:
            return already_visited
        else:
            children = self.successor_states(node)
            if self.mode == 'train':
                best_child = self.find_best_child_train(children)
            elif self.mode == 'test':
                best_child = predict_best_output(self.attributes, children, self.classifier)
            else:
                raise ValueError("Mode {0} must be 'train' or 'test'.".format(self.mode))
            return self.visit_node_greedy(best_child, already_visited, depth_of_search)

    def find_best_child_train(self, children):
        best_loss = sys.maxint
        best_child = None

        for child_1 in children:
            loss_1 = calculate_loss(self.scoring_function, child_1, self.true_output, self.number_of_labels)
            if loss_1 < best_loss:
                best_loss = loss_1
                best_child = child_1

        # Compare only with best
        best_attributes = construct_sparse_attributes(self.attributes, best_child)
        for child_1 in children:
            if child_1 == best_child:
                continue
            loss_1 = calculate_loss(self.scoring_function, child_1, self.true_output, self.number_of_labels)
            if loss_1 == best_loss:
                continue
            if random.uniform(0.0, 1.0) <= self.reduction:
                attributes_1 = construct_sparse_attributes(self.attributes, child_1)
                self.h_training_examples.append(attributes_1 - best_attributes)
                self.h_training_labels.append(np.sign(loss_1 - best_loss))

                self.h_training_examples.append(best_attributes - attributes_1)
                self.h_training_labels.append(np.sign(best_loss - loss_1))

        return best_child

    def get_training_examples(self):
        # assert self.h_training_examples, "Training examples shouldn't be empty. Run greedy search first"
        return self.h_training_examples

    def get_training_labels(self):
        # assert self.h_training_labels, "Training scores shouldn't be empty. Run greedy search first"
        return self.h_training_labels
