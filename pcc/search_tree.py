

class SearchTreeVertex:
    def __init__(self, vertex, classifier_nr, probability, conditional_prob, labels, attributes):
        self.name = vertex
        self.children = []
        self.probability = probability
        self.conditional_prob = conditional_prob
        self.labels = labels
        self.attributes = attributes
        self.classifier_nr = classifier_nr

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def get_name(self):
        return self.name

    def get_probability(self):
        return self.probability

    def get_conditional_prob(self):
        return self.conditional_prob

    def get_inversed_cond_prob(self):
        return 1.0 - self.conditional_prob

    def get_labels(self):
        return self.labels

    def get_attributes(self):
        return self.attributes

    def get_classifier_nr(self):
        return self.classifier_nr

    def __str__(self):
        return str(self.name) + ' ' + str(self.probability) + ' ' + str(self.labels) + ' ' + \
               str(self.conditional_prob) + ' children: ' + str([x.name for x in self.children])


class SearchTree:
    def __init__(self, number_of_labels):
        self.vertex_dict = {}
        self.vertex_count = 0
        self.number_of_labels = number_of_labels

    def __iter__(self):
        return iter(self.vertex_dict.values())

    def add_vertex(self, name, classifier_nr, probability, conditional_prob, labels, attributes):
        self.vertex_count += 1
        new_vertex = SearchTreeVertex(name, classifier_nr, probability, conditional_prob, labels, attributes)
        self.vertex_dict[name] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vertex_dict:
            return self.vertex_dict[n]
        else:
            return None

    def add_edge(self, frm, to):
        self.vertex_dict[frm].add_child(self.vertex_dict[to])

    def get_vertices(self):
        return self.vertex_dict.keys()

    def get_leaves(self):
        leaves = []
        for v in self.vertex_dict:
            if len(self.vertex_dict[v].get_labels()) == self.number_of_labels:
                leaves.append(v)
        return leaves

    def find_best_leaf(self):
        max_prob = 0.0
        min_leaf = None
        for v in self.vertex_dict:
            if len(self.vertex_dict[v].get_labels()) == self.number_of_labels:
                if self.vertex_dict[v].get_conditional_prob() > max_prob:
                    max_prob = self.vertex_dict[v].get_conditional_prob()
                    min_leaf = v

        return self.get_vertex(min_leaf)
