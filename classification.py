import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.stats import ttest_ind, rankdata
from tqdm import tqdm
from tabulate import tabulate
from os import walk
direction = "C:/Users/PLUSR6000280/PycharmProjects/Magisterka/data/"
directionScores = "C:/Users/PLUSR6000280/PycharmProjects/Magisterka/scores/"
directionSamples = "C:/Users/PLUSR6000280/PycharmProjects/Magisterka/samples/"

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

    # print(xList)
    scores = np.zeros((len(xList), 4, 10))

    for i in tqdm(range(len(xList))):
        print(xList[i])
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
    with open(directionScores + 'scores_' + str(len(xList)) + '_v1.npy', 'wb') as f:
        np.save(f, scores)

    scores_mean = np.mean(scores, axis=2)
    print(np.around(scores_mean, decimals=3))
    print(tabulate([scores_mean]))



def checkSamples():
    samplesList = []
    filenames = next(walk(directionSamples), (None, None, []))[2]
    for file in filenames:
        samplesList.append(file)

    samplesList.sort()
    uniqueGenres = np.load('C:/Users/PLUSR6000280/PycharmProjects/Magisterka/' + 'Y_UniqueGenres.npy', allow_pickle=True)
    # print(uniqueGenres)

    genresDifferenceArray = []
    for item in samplesList:
        itemSplitted = item.split('_')
        sample = np.load(directionSamples + item, allow_pickle=True)
        sampleConcat = sample[:, 1]

        if sampleConcat[0] > sampleConcat[1]:
            ir = sampleConcat[0] / sampleConcat[1]
        else:
            ir = sampleConcat[1] / sampleConcat[0]

        genresDifferenceArray.append((uniqueGenres[int(itemSplitted[1])],
                                      uniqueGenres[int(itemSplitted[2].split('.')[0])],
                                      sampleConcat[0],
                                      sampleConcat[1],
                                      ir))

    # Nm / (Nm + Nw)
    print(genresDifferenceArray)
    return genresDifferenceArray


def statistics(N):
    scoresDone = np.load(directionScores + 'scores_20_v1.npy', allow_pickle=True)

    # Ranks
    scores_mean = np.mean(scoresDone, axis=2)
    print(np.around(scores_mean, decimals=3))
    ranks = []
    for ms in scores_mean:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    mean_ranks = np.mean(ranks, axis=0)
    print("\nMean ranks:\n", mean_ranks)

    alfa = 0.05
    t_statistic = np.zeros((20, 4, 4))
    p_value = np.zeros((20, 4, 4))
    # print(scoresDone[0, 0])

    for i in range(scoresDone.shape[0]):
        #DATASETS
        for j in range(scoresDone.shape[1]):
            #CLASSIFIERS
            for k in range(scoresDone.shape[1]):
                t_statistic[i, j, k], p_value[i, j, k] = ttest_ind(scoresDone[i, j], scoresDone[i, k])

    significantlyBetterStatArray = np.logical_and(t_statistic > 0, p_value < alfa)
    print(significantlyBetterStatArray.astype(int))
    listOfTrue = np.argwhere(significantlyBetterStatArray)
    # print(listOfTrue)

    new_list, temp = [], listOfTrue[-1]
    for item in range(0, temp[0]+1):
        new_list.append([[x[1], x[2]] for x in listOfTrue if x[0] == item])

    mapping = {0: 'MLPC', 1: 'KNN', 2: 'DTC', 3: 'SVC'}

    indexed_matrix = [[(mapping[x[0]], mapping[x[1]]) for x in sublist] for sublist in new_list]

    for i in indexed_matrix:
        print(i)


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
    # checkSamples()
