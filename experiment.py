import time
from copy import deepcopy
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from mls.hc_search import HCSearchRanking
from mls_regression.hc_search import HCSearchRegression
from measures import print_measures, calculate_average_loss
from read_data import read_arguments, read_dataset
from pcc.pcc import ProbabilisticClassifierChain


def prepare_experiment(dataset):

    return read_dataset(dataset['format'], dataset['name'], dataset['labels'])


def calculate_run_times(start_time, learning_end_time, prediction_end_time):
    learning_time = learning_end_time - start_time
    prediction_time = prediction_end_time - learning_end_time
    total_time = prediction_end_time - start_time
    return learning_time, prediction_time, total_time


def br_experiment(dataset, base_classifier):
    print("------- {0} -------".format(dataset['name']))
    print("BR experiment")
    x_train, y_train, x_test, y_test, number_of_labels = prepare_experiment(dataset)
    classifier = OneVsRestClassifier(base_classifier, n_jobs=-1)

    start_time = time.clock()
    classifier.fit(x_train, y_train)
    learning_end_time = time.clock()
    y_predicted = classifier.predict(x_test)
    prediction_end_time = time.clock()

    print_measures(y_predicted, y_test, number_of_labels)
    learning_time, prediction_time, total_time = calculate_run_times(start_time, learning_end_time, prediction_end_time)
    print("Learning time: {0:.4f}, Prediction time: {1:.4f}".format(learning_time, prediction_time))

    return classifier


def pcc_experiment(dataset, prediction_type, classifier, epsilon=0.0, mc_iterations=0):
    print("------- {0} -------".format(dataset['name']))
    print("PCC: {0}, epsilon: {1}, MC iterations: {2}".format(prediction_type, epsilon, mc_iterations))
    x_train, y_train, x_test, y_test, number_of_labels = prepare_experiment(dataset)

    pcc = ProbabilisticClassifierChain(classifier)

    start_time = time.clock()
    pcc.fit(x_train, y_train, number_of_labels)
    learning_end_time = time.clock()
    y_predicted = pcc.predict(x_test, prediction_type, epsilon, mc_iterations)
    prediction_end_time = time.clock()

    number_of_visited_classifiers = float(pcc.get_number_of_visited_classifiers()) / len(x_test)
    print("Number of used classifiers: {0}".format(number_of_visited_classifiers))
    print_measures(y_predicted, y_test, number_of_labels)

    learning_time, prediction_time, total_time = calculate_run_times(start_time, learning_end_time, prediction_end_time)
    print("Learning time: {0:.4f}, Prediction time: {1:.4f}".format(learning_time, prediction_time))


def hc_ranking_experiment(dataset, depth_of_search, loss_function,
                          classifier_h, classifier_c, parameter_grid=None,
                          br=None, reduction=1.0):
    print("------- {0} -------".format(dataset['name']))
    if reduction != 1.0:
        print("DATASET REDUCED TO: {0}%".format(reduction*100))
    print "HC: ranking, depth: {0}, Loss function: {1}".format(str(depth_of_search), loss_function)
    x_train, y_train, x_test, y_test, number_of_labels = prepare_experiment(dataset)

    if dataset['name'] in ['bibtex']:
        x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.90, random_state=42)

    h = deepcopy(classifier_h)
    c = deepcopy(classifier_c)

    if parameter_grid is not None:

        best_parameters_h = None
        best_parameters_c = None
        best_loss = 1.0
        x_train_train, x_train_valid, y_train_train, y_train_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

        for parameters_h in ParameterGrid(parameter_grid):
            for parameters_c in ParameterGrid(parameter_grid):
                print("------H params: {0}, C params: {1}------".format(parameters_h, parameters_c))
                h = deepcopy(classifier_h).set_params(**parameters_h)
                c = deepcopy(classifier_c).set_params(**parameters_c)

                y_predicted = hc_learn_and_predict(br, c, depth_of_search, h, loss_function, number_of_labels,
                                                   x_train_train, x_train_valid, y_train_train, y_train_valid,
                                                   reduction)

                calculated_loss = calculate_average_loss(loss_function, y_predicted, y_train_valid, number_of_labels)
                print("Calculated loss: {0}".format(calculated_loss))
                if calculated_loss < best_loss:
                    best_parameters_h = parameters_h
                    best_parameters_c = parameters_c
                    best_loss = calculated_loss

        print {"Final H params: {0}, Final C params: {1}".format(best_parameters_h, best_parameters_c)}
        h = h.set_params(**best_parameters_h)
        c = c.set_params(**best_parameters_c)

    y_predicted = hc_learn_and_predict(br, c, depth_of_search, h, loss_function, number_of_labels,
                                       x_train, x_test, y_train, y_test, reduction)

    print_measures(y_predicted, y_test, number_of_labels)


def hc_learn_and_predict(br, c, depth_of_search, h, loss_function, number_of_labels, x_train, x_test,
                         y_train, y_test, reduction):
    hc_search = HCSearchRanking(h, c, loss_function, depth_of_search, number_of_labels,
                                initial_br=br, h_reduction=reduction)
    start_time = time.clock()
    hc_search.fit(x_train, y_train)
    learning_end_time = time.clock()
    y_predicted = hc_search.predict(x_test, y_test)
    prediction_end_time = time.clock()
    learning_time, prediction_time, total_time = calculate_run_times(start_time, learning_end_time, prediction_end_time)
    print("Learning time: {0:.4f}, Prediction time: {1:.4f}".format(learning_time, prediction_time))
    return y_predicted


def hc_regression_experiment(dataset, depth_of_search, loss_function, regression_h, regression_c, br=None):
    print("------- {0} -------".format(dataset['name']))
    print "HC: regression, depth: {0}, Loss function: {1}".format(str(depth_of_search), loss_function)
    x_train, y_train, x_test, y_test, number_of_labels = prepare_experiment(dataset)

    hc_search = HCSearchRegression(regression_h, regression_c, loss_function, depth_of_search, number_of_labels)

    start_time = time.clock()
    hc_search.fit(x_train, y_train)
    learning_end_time = time.clock()
    y_predicted = hc_search.predict(x_test)
    prediction_end_time = time.clock()

    print_measures(y_predicted, y_test, number_of_labels)

    learning_time, prediction_time, total_time = calculate_run_times(start_time, learning_end_time, prediction_end_time)
    print("Learning time: {0:.4f}, Prediction time: {1:.4f}".format(learning_time, prediction_time))


def main():

    data = {'birds':    {'name': 'birds',       'format': 'dense',  'labels': 19},
            'emotions': {'name': 'emotions',    'format': 'dense',  'labels': 6},
            'scene':    {'name': 'scene',       'format': 'dense',  'labels': 6},
            'yeast':    {'name': 'yeast',       'format': 'dense',  'labels': 14},
            'bibtex':   {'name': 'bibtex',      'format': 'sparse', 'labels': 159},
            'enron':    {'name': 'enron',       'format': 'sparse', 'labels': 53},
            'medical':  {'name': 'medical',     'format': 'sparse', 'labels': 45},
            'tmc2007':  {'name': 'tmc2007',     'format': 'sparse', 'labels': 22}}

    br_cc_experiments(data)
    pcc_monte_carlo_experiments(data)
    pcc_ucs_epsilon_experiments(data)


def br_cc_experiments(data):
    print("BR AND CC")
    br_experiment(data['bibtex'], LogisticRegression())
    br_experiment(data['birds'], LogisticRegression())
    br_experiment(data['emotions'], LogisticRegression())
    br_experiment(data['enron'], LogisticRegression())
    br_experiment(data['medical'], LogisticRegression())
    br_experiment(data['scene'], LogisticRegression())
    br_experiment(data['yeast'], LogisticRegression())

    pcc_experiment(data['bibtex'], 'greedy', LogisticRegression())
    pcc_experiment(data['birds'], 'greedy', LogisticRegression())
    pcc_experiment(data['emotions'], 'greedy', LogisticRegression())
    pcc_experiment(data['enron'], 'greedy', LogisticRegression())
    pcc_experiment(data['medical'], 'greedy', LogisticRegression())
    pcc_experiment(data['scene'], 'greedy', LogisticRegression())
    pcc_experiment(data['yeast'], 'greedy', LogisticRegression())


def pcc_monte_carlo_experiments(data):
    print("PCC MONTE CARLO HAMMING AND GFM")
    pcc_experiment(data['emotions'], 'monte-carlo-hamming', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['scene'], 'monte-carlo-hamming', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['medical'], 'monte-carlo-hamming', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['birds'], 'monte-carlo-hamming', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['yeast'], 'monte-carlo-hamming', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['enron'], 'monte-carlo-hamming', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['bibtex'], 'monte-carlo-hamming', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['emotions'], 'monte-carlo-f-measure', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['scene'], 'monte-carlo-f-measure', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['medical'], 'monte-carlo-f-measure', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['birds'], 'monte-carlo-f-measure', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['yeast'], 'monte-carlo-f-measure', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['enron'], 'monte-carlo-f-measure', LogisticRegression(), mc_iterations=100)
    pcc_experiment(data['bibtex'], 'monte-carlo-f-measure', LogisticRegression(), mc_iterations=100)


def pcc_ucs_epsilon_experiments(data):
    print("PCC UCS WITH EPSILON EXTENSION")
    pcc_experiment(data['emotions'], 'ucs', LogisticRegression(), epsilon=0.00)
    pcc_experiment(data['emotions'], 'ucs', LogisticRegression(), epsilon=0.25)
    pcc_experiment(data['emotions'], 'ucs', LogisticRegression(), epsilon=0.50)
    pcc_experiment(data['scene'], 'ucs', LogisticRegression(), epsilon=0.00)
    pcc_experiment(data['scene'], 'ucs', LogisticRegression(), epsilon=0.25)
    pcc_experiment(data['scene'], 'ucs', LogisticRegression(), epsilon=0.50)
    pcc_experiment(data['birds'], 'ucs', LogisticRegression(), epsilon=0.00)
    pcc_experiment(data['birds'], 'ucs', LogisticRegression(), epsilon=0.25)
    pcc_experiment(data['birds'], 'ucs', LogisticRegression(), epsilon=0.50)
    pcc_experiment(data['medical'], 'ucs', LogisticRegression(), epsilon=0.00)
    pcc_experiment(data['medical'], 'ucs', LogisticRegression(), epsilon=0.25)
    pcc_experiment(data['medical'], 'ucs', LogisticRegression(), epsilon=0.50)
    pcc_experiment(data['enron'], 'ucs', LogisticRegression(), epsilon=0.00)
    pcc_experiment(data['enron'], 'ucs', LogisticRegression(), epsilon=0.25)
    pcc_experiment(data['enron'], 'ucs', LogisticRegression(), epsilon=0.50)
    pcc_experiment(data['yeast'], 'ucs', LogisticRegression(), epsilon=0.00)
    pcc_experiment(data['yeast'], 'ucs', LogisticRegression(), epsilon=0.25)
    pcc_experiment(data['yeast'], 'ucs', LogisticRegression(), epsilon=0.50)
    pcc_experiment(data['bibtex'], 'ucs', LogisticRegression(), epsilon=0.00)
    pcc_experiment(data['bibtex'], 'ucs', LogisticRegression(), epsilon=0.25)
    pcc_experiment(data['bibtex'], 'ucs', LogisticRegression(), epsilon=0.50)


if __name__ == "__main__":
    main()
