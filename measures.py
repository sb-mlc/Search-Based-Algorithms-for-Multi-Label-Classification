import numpy as np


def calculate_average_loss(loss_function, y_predicted, y_true, number_of_labels):
    sum_of_values = 0.0
    length = len(y_predicted)
    for i in range(length):
        sum_of_values += calculate_loss(loss_function, y_predicted[i], y_true[i], number_of_labels)
        # sum_of_values += loss(y_predicted[i], y_true[i], number_of_labels)
    return sum_of_values / length


def calculate_loss(loss_function, predicted_labels, true_labels, number_of_labels):
    if loss_function == 'hamming':
        return hamming_loss(predicted_labels, true_labels, number_of_labels)
    elif loss_function == 'exact_match':
        return exact_match_loss(predicted_labels, true_labels, number_of_labels)
    elif loss_function == 'f_measure':
        return f_measure_loss(predicted_labels, true_labels, number_of_labels)
    elif loss_function == 'accuracy':
        return accuracy_loss(predicted_labels, true_labels, number_of_labels)


def hamming_loss(predicted_labels, true_labels, number_of_labels):
    loss_sum = 0.0
    for i in range(number_of_labels):
        if predicted_labels[i] != true_labels[i]:
            loss_sum += 1.0
    return loss_sum/number_of_labels


def average_exact_match(y_predicted, y_true, number_of_labels):
    exact_match_sum = 0.0
    length = len(y_predicted)
    for i in range(length):
        exact_match_sum += exact_match_loss(y_predicted[i], y_true[i], number_of_labels)
    return exact_match_sum/length


def exact_match_loss(predicted_labels, true_labels, number_of_labels):
    for i in range(number_of_labels):
        if predicted_labels[i] != true_labels[i]:
            return 1.0
    return 0.0


def f_measure_loss(predicted_labels, true_labels, number_of_labels):
    sum_of_matches = sum(np.array(predicted_labels) * np.array(true_labels))
    f_measure_denominator = sum(predicted_labels) + sum(true_labels)
    if f_measure_denominator == 0.0:
        return 0.0
    return 1.0 - (2.0 * sum_of_matches / f_measure_denominator)


def accuracy_loss(predicted_labels, true_labels, number_of_labels):
    sum_of_matches = float(sum(np.array(predicted_labels) * np.array(true_labels)))
    accuracy_denominator = 0.0
    for i in range(number_of_labels):
        if (predicted_labels[i] == 1) or (true_labels[i] == 1):
            accuracy_denominator += 1.0
    if accuracy_denominator == 0.0:
        return 0.0
    return 1.0 - (sum_of_matches / accuracy_denominator)


def print_measures(y_predicted, y_test, number_of_labels):
    avg_hamming = calculate_average_loss('hamming', y_predicted, y_test, number_of_labels)
    avg_exact = calculate_average_loss('exact_match', y_predicted, y_test, number_of_labels)
    avg_f_measure = calculate_average_loss('f_measure', y_predicted, y_test, number_of_labels)
    avg_accuracy = calculate_average_loss('accuracy', y_predicted, y_test, number_of_labels)
    print("Labels nr: {0}, Examples in test set: {1}".format(number_of_labels, len(y_predicted)))
    print("Hamming loss: {0:.4f} Exact match loss: {1:.4f}".format(avg_hamming, avg_exact))
    print("F-measure loss: {0:.4f} Accuracy loss: {1:.4f}".format(avg_f_measure, avg_accuracy))