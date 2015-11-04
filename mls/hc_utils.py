import numpy as np
from scipy.sparse import csr_matrix, hstack


def construct_dense_attributes(original_attributes, labels, second_level_features=False):
    constructed_attributes = []

    for first_label in labels:
        constructed_attributes.extend([first_label * i for i in original_attributes])
        if second_level_features:
            for second_label in labels:
                constructed_attributes.extend([first_label * second_label * i for i in original_attributes])

    return constructed_attributes


def construct_sparse_attributes(original_attributes, labels, second_level_features=False):
    sparse_fragments = []
    number_of_attributes = len(original_attributes)
    empty_sparse_fragment = csr_matrix(([], ([], [])), shape=(1, number_of_attributes))

    row_indices = np.zeros(number_of_attributes)
    col_indices = np.array(range(number_of_attributes))
    full_sparse_fragment = csr_matrix((original_attributes, (row_indices, col_indices)),
                                      shape=(1, number_of_attributes))

    for label in labels:
        if label == 0:
            sparse_fragments.append(empty_sparse_fragment)
        elif label == 1:
            sparse_fragments.append(full_sparse_fragment)
        else:
            raise ValueError("Label should be 0 or 1 not: {0}".format(label))
        if second_level_features:
            for second_label in labels:
                if label == 1 and second_label == 1:
                    sparse_fragments.append(full_sparse_fragment)
                else:
                    sparse_fragments.append(empty_sparse_fragment)

    return hstack(sparse_fragments, format='csr')