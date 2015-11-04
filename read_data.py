import sys
import scipy
import numpy as np

from scipy.io.arff import loadarff
from scipy.sparse import hstack


def read_arguments():
    if len(sys.argv) < 3:
        print "Usage: experiment.py <sparse/dense> <dataset_name> <number_of_labels>"
        sys.exit(0)

    sparse_or_dense = sys.argv[1]
    assert sparse_or_dense == "sparse" or sparse_or_dense == "dense"

    dataset_name = sys.argv[2]
    number_of_labels = int(sys.argv[3])
    return sparse_or_dense, dataset_name, number_of_labels


def read_dataset(sparse_or_dense, dataset_name, number_of_labels):

    path_train = 'data/' + sparse_or_dense + '/' + dataset_name + '/' + dataset_name + '-train.arff'
    path_test = 'data/' + sparse_or_dense + '/' + dataset_name + '/' + dataset_name + '-test.arff'

    if sparse_or_dense == 'sparse':
        x_train, y_train, x_test, y_test = read_sparse_arff_dataset(path_train, path_test, number_of_labels)
    elif sparse_or_dense == 'dense':
        x_train, y_train, x_test, y_test = read_dense_arff_dataset(path_train, path_test, number_of_labels)

    return x_train, y_train, x_test, y_test, number_of_labels


def read_dense_arff_dataset(train_path, test_path, number_of_labels):

    train_dataset, meta_train = loadarff(open(train_path, 'r'))
    test_dataset, meta_test = loadarff(open(test_path, 'r'))

    meta_names = meta_train.names()

    attributes = meta_names[0:-number_of_labels]
    classes = meta_names[-number_of_labels:len(meta_names)]

    x_train = np.asarray(train_dataset[:][attributes].tolist(), dtype=np.float32)
    y_train = np.asarray(train_dataset[:][classes].tolist(), dtype=np.float32)

    x_test = np.asarray(test_dataset[:][attributes].tolist(), dtype=np.float32)
    y_test = np.asarray(test_dataset[:][classes].tolist(), dtype=np.float32)

    return x_train, y_train, x_test, y_test


def read_sparse_arff_dataset(train_path, test_path, number_of_labels):

    x_row_indices, x_col_indices, y_row_indices, y_col_indices, labels, attributes, examples_nr = read_sparse_arff_file(train_path, number_of_labels)

    x_train = np.asarray(scipy.sparse.coo_matrix(([1]*len(x_row_indices), (x_row_indices, x_col_indices)),
                                                 shape=(examples_nr, len(attributes))).todense())
    y_train = np.asarray(scipy.sparse.coo_matrix(([1]*len(y_row_indices), (y_row_indices, y_col_indices)),
                                                 shape=(examples_nr, len(labels))).todense())

    x_row_indices_test, x_col_indices_test, y_row_indices_test, y_col_indices_test, labels_test, attributes_test, examples_nr_test = read_sparse_arff_file(test_path, number_of_labels)
    x_test = np.asarray(scipy.sparse.coo_matrix(([1]*len(x_row_indices_test), (x_row_indices_test, x_col_indices_test)),
                                                shape=(examples_nr_test, len(attributes))).todense())
    y_test = np.asarray(scipy.sparse.coo_matrix(([1]*len(y_row_indices_test), (y_row_indices_test, y_col_indices_test)),
                                                shape=(examples_nr_test, len(labels))).todense())

    assert set(attributes) == set(attributes_test)
    assert set(labels) == set(labels_test)

    return x_train, y_train, x_test, y_test


def read_sparse_arff_file(dataset_file_name, number_of_labels):
    x_row_indices = []
    x_col_indices = []
    y_row_indices = []
    y_col_indices = []
    attribute_marker = "@attribute"
    data_marker = "@data"
    attr_and_classes = []
    attributes = []
    labels = []
    is_data = False
    example_counter = 0

    with open(dataset_file_name) as f:
        for line in f.readlines():
            if not is_data:
                if line.startswith(attribute_marker):
                    line = line.replace(attribute_marker, "").strip().split()[0]
                    attr_and_classes.append(line)
                if line.startswith(data_marker):
                    is_data = True
                    attributes = attr_and_classes[0:-number_of_labels]
                    labels = attr_and_classes[-number_of_labels:len(attr_and_classes)]
            else:
                fields = line.strip().strip('{').strip('}').strip().split(',')
                for field in fields:
                    f_split = field.split()
                    if len(f_split) == 0:
                        continue
                    index = int(f_split[0])
                    if index < len(attributes):
                        x_row_indices.append(example_counter)
                        x_col_indices.append(index)
                    else:
                        y_row_indices.append(example_counter)
                        y_col_indices.append(index - len(attributes))

                example_counter += 1
    return x_row_indices, x_col_indices, y_row_indices, y_col_indices, labels, attributes, example_counter

