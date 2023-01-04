import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.stats import ttest_ind
from tqdm import tqdm
from tabulate import tabulate
from os import walk
direction = "C:/Users/PLUSR6000280/PycharmProjects/Magisterka/data/"
directionScores = "C:/Users/PLUSR6000280/PycharmProjects/Magisterka/scores/"

classifiers = [MLPClassifier(random_state=1, max_iter=300),
               KNeighborsClassifier(n_neighbors=3),
               DecisionTreeClassifier(),
               SVC()]

kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)

def classification():
    xList = []
    yList = []
    filenames = next(walk(direction), (None, None, []))[2]
    for file in filenames:
        if file.startswith("X"):
            xList.append(file)
        if file.startswith("y"):
            yList.append(file)

    xList.sort()
    yList.sort()

    if len(xList) != len(yList):
        print("ERROR! X LIST IS DIFFERENT THAN Y LIST")
        return

    print(xList)
    scores = np.zeros((len(xList), 4, 10))

    for i in tqdm(range(len(xList))):
        X = np.load(direction + xList[i], allow_pickle=True)
        y = np.load(direction + yList[i], allow_pickle=True)

        try:
            for fold_index, (train_index, test_index) in enumerate(tqdm(kf.split(X, y))):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                for cls_index, base_cls in enumerate(classifiers):
                    cls = clone(base_cls)
                    cls.fit(X_train, y_train)
                    y_pred = cls.predict(X_test)
                    score = balanced_accuracy_score(y_test, y_pred)
                    scores[i, cls_index, fold_index] = score
        except KeyError:
            print('Error! for data:' + str(xList[i]) + ' and ' + str(yList[i]))

    # xList[i].split('_', 1)[1]
    with open(directionScores + 'scores_' + str(len(xList)) + '_v3.npy', 'wb') as f:
        np.save(f, scores)

    scores_mean = np.mean(scores, axis=2)
    print(scores_mean)
    print(tabulate([scores_mean]))


def statistics(N):
    scoresDone = np.load(directionScores + 'scores_20_v3.npy', allow_pickle=True)
    # scoresDone = scoresDone.astype(np.float64)
    alfa = 0.05
    t_statistic = np.zeros((20, 4, 4))
    p_value = np.zeros((20, 4, 4))
    print(scoresDone[0, 0])

    for i in range(scoresDone.shape[0]):
        #DATASETS
        for j in range(scoresDone.shape[1]):
            #CLASSIFIERS
            for k in range(scoresDone.shape[1]):
                t_statistic[i, j, k], p_value[i, j, k] = ttest_ind(scoresDone[i, j], scoresDone[i, k])

        # print(t_statistic)
        print(p_value)

    # headers = ["MLPC", "KNN", "DTC", "SVC"]
    # names_column = np.array([["MLPC"], ["KNN"], ["DTC"], ["SVC"]])
    # t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    # t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    # p_value_table = np.concatenate((names_column, p_value), axis=1)
    # p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

if __name__ == '__main__':
    # scoresDone = np.load(directionScores + 'scores_20_v2', allow_pickle=True)
    # print(tabulate([scoresDone]))

    # classification()
    statistics(20)
