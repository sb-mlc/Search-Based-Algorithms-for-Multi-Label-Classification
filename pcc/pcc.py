import heapq
import random
import numpy as np
from copy import deepcopy

from search_tree import SearchTree
from rigid_classifier import RigidClassifier


class ProbabilisticClassifierChain:

    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.classifier_dict = {}
        self.number_of_labels = None
        self.number_of_visited_classifiers = 0

    def fit(self, x_train, y_train, number_of_labels):
        self.number_of_labels = number_of_labels
        for classifier_nr in xrange(number_of_labels):
            y_train_single_label = y_train[:, classifier_nr]
            x = x_train if classifier_nr == 0 else np.hstack((x_train, y_train[:, 0:classifier_nr]))

            if 1 in y_train_single_label:
                node_classifier = deepcopy(self.base_classifier)
                node_classifier.fit(x, y_train_single_label)
            else:
                # Scikit-learn classifiers cannot be fitted when label is not present in training data,
                # thus we have to use dummy classifier with constant classification strategy.
                node_classifier = RigidClassifier(result_label=0)

            self.classifier_dict[classifier_nr] = node_classifier

        return self

    def predict(self, x_test, inference_type='ucs', epsilon=0.0, iterations=50):
        if inference_type == 'ucs':
            return self.predict_ucs(x_test, epsilon)
        elif inference_type == 'greedy':
            return self.predict_greedy(x_test)
        elif inference_type == 'full':
            return self.predict_full(x_test)
        elif inference_type == 'smart':
            return self.predict_smart(x_test)
        elif inference_type == 'monte-carlo-hamming':
            return self.predict_monte_carlo_hamming(x_test, k=iterations)
        elif inference_type == 'monte-carlo-f-measure':
            return self.predict_monte_carlo_f_measure(x_test, k=iterations)

    def predict_greedy(self, x_test):
        y = None
        for classifier_nr in xrange(self.number_of_labels):
            one_label_classifier = self.classifier_dict[classifier_nr]
            one_label_predictions = one_label_classifier.predict(x_test)
            y = one_label_predictions if classifier_nr == 0 else np.vstack((y, one_label_predictions))
            x_test = append_predictions_to_attributes(one_label_predictions, x_test)

        return np.transpose(y)

    def predict_with_weighted_probability(self, x_test):
        y = None
        for classifier_nr in xrange(self.number_of_labels):
            single_classifier = self.classifier_dict[classifier_nr]
            single_prob = single_classifier.predict_proba(x_test)
            single_predictions = []
            for example_nr in xrange(len(x_test)):
                prediction = 0.0 if random.uniform(0.0, 1.0) < single_prob[example_nr][0] else 1.0
                single_predictions.append(prediction)

            y = single_predictions if classifier_nr == 0 else np.vstack((y, single_predictions))
            x_test = append_predictions_to_attributes(single_classifier.predict(x_test), x_test)

        return np.transpose(y)

    def predict_monte_carlo_hamming(self, x_test, k):
        # Create multiple predictions
        predictions = np.asarray([self.predict_with_weighted_probability(x_test) for _ in xrange(k)])
        # Count how many times each label was predicted for given training example
        sums_of_columns = np.sum(predictions, axis=0)

        y = np.zeros((len(x_test), self.number_of_labels))
        for example_nr in xrange(len(x_test)):
            for label_nr in xrange(self.number_of_labels):
                # If label was positive for given example in more than
                # half predictions it should be considered as correct.
                if sums_of_columns[example_nr][label_nr] >= k/2:
                    y[example_nr][label_nr] = 1.0
        return y

    def predict_monte_carlo_f_measure(self, x_test, k, verbose=True):
        # Create multiple predictions
        predictions = np.asarray([self.predict_with_weighted_probability(x_test) for _ in xrange(k)])

        # Create matrix W
        matrix_w = np.zeros((self.number_of_labels, self.number_of_labels))
        for i in range(self.number_of_labels):
            for j in range(self.number_of_labels):
                matrix_w[i][j] = 1.0 / (i + j + 2.0)
        if verbose:
            print matrix_w

        y = []  # Initialize list of results

        # For each example create matrix P and calculate result
        for example_nr in range(len(x_test)):
            matrix_of_samples = predictions[:, example_nr, :]
            if verbose:
                print matrix_of_samples

            matrix_p = np.zeros((self.number_of_labels, self.number_of_labels))
            for i in range(self.number_of_labels):
                for j in range(self.number_of_labels):
                    ij_elem = 0.0
                    for row in matrix_of_samples:
                        if row[i] == 1.0 and np.sum(row) == j + 1:
                            ij_elem += 1.0
                    matrix_p[i][j] = ij_elem/k
            if verbose:
                print matrix_p

            # Calculate matrix F
            matrix_f = np.dot(matrix_p, matrix_w)
            if verbose:
                print matrix_f

            # Find result for given example
            best_candidate = None
            best_score = 0.0
            for label_nr in range(self.number_of_labels):
                row = matrix_f[:, label_nr]
                ind = np.argpartition(row, -(label_nr + 1))[-(label_nr + 1):]
                if verbose:
                    print row, ind, sum(row[ind])
                if sum(row[ind]) > best_score:
                    best_candidate = ind
                    best_score = sum(row[ind])

            if verbose:
                print best_candidate
            result = np.zeros(self.number_of_labels)
            result[best_candidate] = 1
            y.append(result)

        return y

    def look_ahead(self, search_tree, vertex, heap, current_best):
        current = vertex
        while len(current.get_labels()) != self.number_of_labels:
            children = self.generate_children(search_tree, current)
            if children[0].get_conditional_prob() >= children[1].get_conditional_prob():
                if children[0].get_conditional_prob() > current_best:
                    current = children[0]
                    heapq.heappush(heap, (children[1].get_inversed_cond_prob(), children[1]))
                else:
                    return None
            else:
                if children[1].get_conditional_prob() > current_best:
                    current = children[1]
                    heapq.heappush(heap, (children[0].get_inversed_cond_prob(), children[0]))
                else:
                    return None

        return current

    def predict_smart(self, x_test):
        y = []
        number_of_test_examples = len(x_test)
        for i in xrange(number_of_test_examples):
            search_tree = SearchTree(self.number_of_labels)
            root = search_tree.add_vertex("root_", 0, 1.0, 1.0, [], x_test[i])
            h = []
            heapq.heappush(h, (root.get_inversed_cond_prob(), root))
            best_leaf = None
            best_score = 0.0

            while len(h) != 0:
                vertex = heapq.heappop(h)[1]
                if vertex.get_conditional_prob() < best_score:
                    break
                else:
                    greedy_leaf = self.look_ahead(search_tree, vertex, h, best_score)
                    if greedy_leaf is None:
                        continue
                    else:
                        best_leaf = greedy_leaf
                        best_score = greedy_leaf.get_conditional_prob()

            y.append(best_leaf.get_labels())

        return np.array(y)

    def predict_ucs(self, x_test, epsilon):
        y = []
        number_of_test_examples = len(x_test)
        for i in xrange(number_of_test_examples):
            search_tree = SearchTree(self.number_of_labels)
            root = search_tree.add_vertex("root_", 0, 1.0, 1.0, [], x_test[i])
            q = []
            greedy_q = []
            heapq.heappush(q, (root.get_inversed_cond_prob(), root))
            best_leaf = None
            while len(q) != 0:
                vertex = heapq.heappop(q)[1]
                if len(vertex.get_labels()) == self.number_of_labels:  # If leaf
                    best_leaf = vertex
                    break
                else:
                    children = self.generate_children(search_tree, vertex)
                    no_children_inserted = True
                    for child in children:
                        # print child.get_inversed_cond_prob()
                        if child.get_probability() > epsilon:
                            heapq.heappush(q, (child.get_inversed_cond_prob(), child))
                            no_children_inserted = False
                    if no_children_inserted:
                        heapq.heappush(greedy_q, (vertex.get_inversed_cond_prob(), vertex))

            if best_leaf is None:
                best_score = 0.0

                while len(greedy_q) != 0:
                    vertex = heapq.heappop(greedy_q)[1]
                    greedy_leaf = self.look_ahead(search_tree, vertex, [], best_score)
                    if greedy_leaf is None:
                        continue
                    else:
                        best_leaf = greedy_leaf
                        best_score = greedy_leaf.get_conditional_prob()
                        # print best_leaf, best_score
                # print("------")
            y.append(best_leaf.get_labels())
            # print "Done: " + str(i) + "/" + str(number_of_training_examples)

        return np.array(y)

    def predict_full(self, x_test):
        y = []
        number_of_test_examples = len(x_test)
        for i in xrange(number_of_test_examples):
            search_tree = SearchTree(self.number_of_labels)
            root = search_tree.add_vertex("root_", 0, 1.0, 1.0, [], x_test[i])
            self.generate_search_tree(search_tree, root)
            best_leaf = search_tree.find_best_leaf()
            y.append(best_leaf.get_labels())
            # print "Done: " + str(i) + "/" + str(number_of_training_examples)

        return np.array(y)

    def generate_search_tree(self, search_tree, parent_vertex):
        children = self.generate_children(search_tree, parent_vertex)
        if children[0].get_classifier_nr() < self.number_of_labels:
            for child in children:
                self.generate_search_tree(search_tree, child)

    def generate_children(self, search_tree, parent_vertex):
        self.number_of_visited_classifiers += 1

        parent_attributes = parent_vertex.get_attributes()
        parent_name = parent_vertex.get_name()
        classifier_nr = parent_vertex.get_classifier_nr()
        class_values = [0, 1]
        result = []

        single_classifier = self.classifier_dict[classifier_nr]
        # print parent_attributes
        predicted_probability = single_classifier.predict_proba(parent_attributes)

        for class_value in class_values:
            vertex_name = parent_name + str(class_value)
            # print predicted_probability
            probability = predicted_probability[0, class_value]
            cond_probability = probability * parent_vertex.get_conditional_prob()
            new_labels = parent_vertex.get_labels() + [class_value]
            # print type(parent_attributes)
            # print parent_attributes.shape
            new_attributes = np.append(parent_attributes, class_value)
            vertex = search_tree.add_vertex(vertex_name, classifier_nr + 1, probability, cond_probability, new_labels, new_attributes)
            search_tree.add_edge(parent_name, vertex_name)
            result.append(vertex)

        return result

    def get_number_of_visited_classifiers(self):
        return self.number_of_visited_classifiers


def append_predictions_to_attributes(predictions, x_test):
    # To perform hstack on attributes and labels we need to treat predicted label for every example as list.
    each_prediction_as_list = map(lambda x: [x], predictions)
    x_test = np.hstack((x_test, each_prediction_as_list))
    return x_test