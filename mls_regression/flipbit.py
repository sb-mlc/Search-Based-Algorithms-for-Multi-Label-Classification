import sys

from measures import calculate_loss
from mls.hc_utils import construct_sparse_attributes


class FlipBit:

    def __init__(self, attr, number_of_labels, scoring_function, true_output=None, fitted_regressor=None, flip_both_ways=False):
        self.attributes = attr
        self.number_of_labels = number_of_labels
        self.true_output = true_output
        self.regressor = fitted_regressor
        self.initial_state = [0] * number_of_labels
        self.h_training_examples = []
        self.h_training_scores = []
        self.flip_both_ways = flip_both_ways
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
            children.append(child)

        return children

    def greedy_search(self, depth_of_search):
        visited_nodes = self.visit_node_greedy(self.initial_state, [], depth_of_search)
        return visited_nodes

    def visit_node_greedy(self, node, already_visited, depth_of_search):
        already_visited.append(node)
        depth_of_search -= 1

        if depth_of_search < 0:
            return already_visited
        else:
            children = self.successor_states(node)
            best_child = self.find_best_child(children)
            return self.visit_node_greedy(best_child, already_visited, depth_of_search)

    def find_best_child(self, children):
        best_score = sys.maxint
        best_child = None

        for child in children:
            example = construct_sparse_attributes(self.attributes, child)
            if (self.regressor is not None) and (self.true_output is None):
                score = self.regressor.predict(example)
            elif (self.true_output is not None) and (self.regressor is None):
                score = calculate_loss(self.scoring_function, child, self.true_output, self.number_of_labels)
                self.h_training_examples.append(example)
                self.h_training_scores.append(score)
            else:
                raise ValueError("Either regressor or true_output must not be None.")

            if score < best_score:
                best_score = score
                best_child = child

        return best_child

    def get_training_examples(self):
        assert self.h_training_examples, "Training examples shouldn't be empty. Run greedy search first"
        return self.h_training_examples

    def get_training_scores(self):
        assert self.h_training_scores, "Training scores shouldn't be empty. Run greedy search first"
        return self.h_training_scores
