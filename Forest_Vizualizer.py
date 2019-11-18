import pandas as pd
import numpy as np
import collections

# for getting graphs
from sklearn.tree import export_graphviz
from subprocess import call

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import _tree


class ForestDescriber:

    def __init__(self, forest):
        self.forest = forest

    # helper function
    @staticmethod
    def __value2prob(value):
        return value / value.sum(axis=1).reshape(-1, 1)

    # found on stack overflow. Other than some cursory edits, treated like a black box
    def decision_path_of_data_point(self, data_index, model, xdata, feature_names, weight):

        node_indicator = model.decision_path(xdata)
        feature = model.tree_.feature
        threshold = model.tree_.threshold
        leave_id = model.apply(xdata)

        print("     Decision path", end=' ')
        print("where ", end=" ")
        print(weight, end="")
        print("% Of incorrect predictions for specified class followed it:")
        print("WHEN", end=' ')

        node_index = node_indicator.indices[node_indicator.indptr[data_index]:
                                            node_indicator.indptr[data_index + 1]]

        for n, node_id in enumerate(node_index):
            if leave_id[data_index] == node_id:
                values = model.tree_.value[node_id]
                probs = self.__value2prob(values)
                print('THEN Y={} (proportion={}) (values={})'.format(
                    probs.argmax(), probs.max(), values))
                continue
            if n > 0:
                print('&& ', end='')
            if xdata[data_index, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            if feature[node_id] != _tree.TREE_UNDEFINED:
                print(
                    "%s %s %s" % (
                        feature_names[feature[node_id]],
                        threshold_sign,
                        threshold[node_id]),
                    end=' ')
        print("")

    #   SUPERMETHOD FOR GETTING DESIRED DECISION PATHS
    # this method first predicts on input data. It then creates a subset of data where the prediction was inaccurate.
    # The method takes the desired class and outputs decision paths where this inaccurate data for that class goes for
    # each tree. It also outputs the % of the inaccurate data goes to that decision path
    def get_decision_paths(self, the_class, xdata, ydata, feature_names, threshold=0.1):

        x = 0  # ticker
        for model in self.forest.estimators_:

            # getting predictions
            xtemp = pd.DataFrame(data=xdata)
            xtemp["actual"] = ydata
            xtemp["prediction"] = model.predict(xdata)
            xtemp = xtemp[xtemp["actual"] != xtemp["prediction"]]

            # down selects to just elements of the desired test class
            xtemp = xtemp[xtemp["actual"] == the_class]

            # cleaning temp data for use
            x_miss = xtemp.drop(labels=["prediction", "actual"], axis=1).reset_index(drop=True)

            # getting the leaves we want the decision path from
            app = model.apply(x_miss)  # adds missing data to tree, where it sits in the leaves

            frequencies = collections.Counter(app)  # gets frequencies of each leaf for the data
            desiredleaves = []

            # prints which estimator the decision paths are part of
            print("_______________________________________________________________")
            print("                  ESTIMATOR #" + str(x + 1) + "                ")
            print("_______________________________________________________________")
            x += 1

            # this for loop checks the frequencies against total size
            for element in frequencies:
                counter = frequencies[element] / app.size
                if counter > threshold:
                    desiredleaves.append(element)

            # now iterating through the list of desired leaves
            for leaf in desiredleaves:
                index = xtemp.index[np.nonzero(app == leaf)[0][0]]
                # this pulls the first index of the missing data that hits that leaf

                self.decision_path_of_data_point(index, model, xdata, feature_names, frequencies[leaf] / app.size * 100)
                # runs func.

    # gets the images of the desired set of trees
    def forest_viz(self, feature_list, desired_trees=None):

        # prints all the trees
        if desired_trees is None:
            desired_trees = []
        if not desired_trees:
            i = 0
            for model in self.forest.estimators_:
                file_name_d = 'tree' + str(i) + '.dot'
                file_name_p = 'tree' + str(i) + '.png'
                export_graphviz(model, feature_names=feature_list,
                                # class_names = iris.target_names,
                                out_file=file_name_d,
                                rounded=True,
                                proportion=False, filled=True)

                # Here is where you would want to convert to images/graphs?
                call(['dot', '-Tpng', file_name_d, '-o', file_name_p, '-Gdpi=600'])

                i += 1

        # prints only the desired list of trees
        else:
            i = 0
            for element in desired_trees:
                file_name_d = 'tree' + str(element) + '.dot'
                file_name_p = 'tree' + str(element) + '.png'

                export_graphviz(self.forest.estimators_[element], feature_names=feature_list,
                                # class_names = iris.target_names,
                                out_file=file_name_d,
                                rounded=True,
                                proportion=False, filled=True)

                call(['dot', '-Tpng', file_name_d, '-o', file_name_p, '-Gdpi=600'])
                i += 1
