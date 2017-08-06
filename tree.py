# Python code to test the Decision trees ML algorithm
import numpy as np
import pandas as pd
import collections
import math


def shannon_entropy(dataset):
    output_list = list(dataset[:, -1])
    labels_dict = collections.Counter(output_list)
    entropy = 0.0
    total = float(sum(labels_dict.values()))
    for label, count in labels_dict.items():
        temp = float(int(count) / total)
        entropy -= temp * math.log(temp, 2)
    return entropy


def majority_count(class_list):
    class_count = collections.Counter(class_list)
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_count(class_list)
    best_feat = select_best_feature_split(dataset)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    unique_label = np.unique(dataset[:, best_feat])
    for value in unique_label:
        # temp = del (labels[best_feat])
        temp = [label for label in labels if label != best_feat_label]
        my_tree[best_feat_label][value] = create_tree(create_data_split(dataset, best_feat, value), temp)
    return my_tree


def create_data_split(dataset, axis, value):
    """
    Function used to create a split of data
    :param dataset: the input dataset as array
    :param axis: the feature axis to split upon
    :param value: value to be used for split
    :return: the array of split having the value
    """
    ret_split_val = []
    for val in dataset:
        if str(val[axis]) == str(value):
            temp = list(val)
            temp.remove(val[axis])
            ret_split_val.append(temp)
    return np.array(ret_split_val)


def select_best_feature_split(dataset):
    """
    Function to select the feature to be used for splitting
    :param dataset: the input dataset
    :return: List of the suitable features
    """
    labels_to_check = dataset.shape[1] - 1
    base_entropy = shannon_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(labels_to_check):
        unique_val = np.unique(dataset[:, i])
        new_entropy = 0.0
        for val in unique_val:
            sub_data_set = create_data_split(dataset, i, val)
            prob = len(sub_data_set) / float(len(dataset))
            new_entropy += prob * shannon_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def create_dataset(size):
    np.random.seed(1)
    dataset = np.random.randint(low=0, high=2, dtype=int, size=(size[0], size[1]))
    data_df = pd.DataFrame(dataset, columns=['no_surfacing', "flippers"])
    data_df["output"] = np.where((data_df.no_surfacing == 1) & (data_df.flippers == 1), "yes", "no")
    # line to add a random new label to increase the entropy
    # data_df.loc[:90, "output"] = ["maybe"] * len(data_df.loc[:90, "output"])
    return np.array(data_df), data_df.columns


def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if str(testVec[featIndex]) == str(key):
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def static_dataset():
    return np.array([[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']])


def store_tree(input_tree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


if __name__ == "__main__":
    lens_dat = pd.read_csv("datasets/lenses.dat", sep="\s+", header=None)
    lens_dat.columns = ["index", "patient_age", "specs_pres", "astigmatic", "tear_prod_rate", "classes"]
    lens_dat = lens_dat[["patient_age", "specs_pres", "astigmatic", "tear_prod_rate", "classes"]]
    data = np.array(lens_dat)
    labels = list(lens_dat.columns)
    print create_tree(data, labels[:-1])
    # store = False
    # data, label = create_dataset((100, 2))
    # data = static_dataset()
    # my_tree = create_tree(data, list(label))
    # if store:
    #     print "Storing the Classifier Model"
    #     store_tree(my_tree, "./trained_models/tree_classifier.txt")
    # print "Retrieving the Classifier Model..."
    # retrieved_tree = grab_tree("./trained_models/tree_classifier.txt")
    # print classify(retrieved_tree, list(label[:-1]), [0, 0])
