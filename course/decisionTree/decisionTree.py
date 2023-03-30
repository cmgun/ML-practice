from math import log
import operator

def cal_shannon_ent(data_set):
    """
    calculate shannon entropy
    :param data_set: the data set
    :return: shannon entropy
    """
    num_entries = len(data_set)
    label_counts = {}
    # calculate the different label number, last column is the label
    for featVec in data_set:
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    split the dataset, according to the target axis with target value.
    :param data_set: dataset
    :param axis: feature axis
    :param value: feature value
    :return: dataset without target axis and value
    """
    ret_data_set = []
    for featVec in data_set:
        # row data
        if featVec[axis] == value:
            # chop out axis used for splitting
            reduced_feat_vec = featVec[:axis]
            reduced_feat_vec.extend(featVec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def feature_selection(data_set):
    """
    feature selection based on maximize entropy
    :param data_set:
    :return: best feature axis
    """
    # the last column is used for the labels
    num_features = len(data_set[0]) - 1
    # base info
    base_entropy = cal_shannon_ent(data_set)
    best_info_gain = 0.0
    best_feature = -1
    # iterate over all the features
    for i in range(num_features):
        # create a list of all the examples of this feature
        feat_list = [example[i] for example in data_set]
        # feature unique value
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            # data set without feature i & value
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * cal_shannon_ent(sub_data_set)
        # current info gain
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_count(class_list):
    """
    count the class number, and sort by desc
    :param class_list: 
    :return: max count class name
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    create decision tree using ID3 algorithm.
    procedure:
    1. feature selection using maximize info gain, then create the tree node.
    2. crop out best feature and go to next level of the tree, repeat step 1.
    :param data_set:
    :param labels:
    :return:
    """
    # get y label
    class_list = [data[-1] for data in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        # stop splitting when all classes are equal
        return class_list[0]
    # stop splitting when there are no more features in dataSet
    if len(data_set[0]) == 1:
        return majority_count(class_list)
    best_feat = feature_selection(data_set)
    best_feat_label = labels[best_feat]
    tree = {best_feat_label: {}}
    del(labels[best_feat])
    feat_values = [data[best_feat] for data in data_set]
    unique_vals = set(feat_values)
    # print(uniqueVals)
    for value in unique_vals:
        # copy all labels, so trees don't mess up existing labels
        sub_labels = labels[:]
        sub_data_set = split_data_set(data_set, best_feat, value)
        tree[best_feat_label][value] = create_tree(sub_data_set, sub_labels)
    return tree


def classify(input_tree, feat_labels, test_vec):
    """
    using tree to classify
    :param input_tree: tree classifier
    :param feat_labels: feature labels
    :param test_vec: test data
    :return: predict labels
    """
    first_str = next(iter(input_tree))
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    key = test_vec[feat_index]
    value_of_feat = second_dict[key]
    if isinstance(value_of_feat, dict):
        class_label = classify(value_of_feat, feat_labels, test_vec)
    else:
        class_label = value_of_feat
    return class_label
