import operator

from mls.hc_utils import construct_sparse_attributes


def predict_best_output(attributes, outputs, classifier):

    result_dict = {}
    for i in range(len(outputs)):
        pretendent_attributes = construct_sparse_attributes(attributes, outputs[i])
        result_dict[i] = classifier.decision_function(pretendent_attributes)[0]

    # print result_dict
    index_of_best = min(result_dict.iteritems(), key=operator.itemgetter(1))[0]
    return outputs[index_of_best]
