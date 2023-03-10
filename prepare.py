#!/usr/bin/python
# import loader
import pandas as pd
from scipy.signal import medfilt2d
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from os import walk

# numpy.set_printoptions(threshold=sys.maxsize)
direction = "C:/Users/PLUSR6000280/PycharmProjects/Magisterka/"


def photos(execute, movies):
    if execute:
        allImagesArray = np.load(direction + "Photos.npy", allow_pickle=True)
        # for print purposes
        # arrayOfMovies = np.zeros((5900, 3))
        arrayOfMovies = np.zeros((len(movies), 3))

        imageArray = np.array(allImagesArray[movies][:, 1])

        # image
        for n, imag in enumerate(tqdm(imageArray)):
            img = np.mean(imag, axis=(0, 1))
            arrayOfMovies[n] = img
            # ekstrakcja danych do kolorow powyzsza linia ^

        # print(arrayOfMovies)

        # imagess = arrayOfMovies.reshape((59, 100, 3))
        # for i in range(3):
        #     imagess[:, :, i] = medfilt2d(imagess[:, :, i], kernel_size=5)
        # plt.imshow(imagess.astype(int))
        # plt.show()

        return arrayOfMovies

def text(execute, movies, maxFeatures):
    if execute:
        moviesPlots = []
        plot_array = np.load(direction + "Plots_WithoutArray.npy", allow_pickle=True)

        for movie in movies:
            moviesPlots.append(plot_array[movie])

        moviesPlots = np.array(moviesPlots, dtype=object)

        # to nie dziala - lemantyzacja
        # nlp = spacy.load('en_core_web_sm')
        # doc = nlp(plot_array)
        #
        # for i, sentence in enumerate(doc.sents):
        #     for token in sentence:
        #         print(i, token.lemma_, token.pos_, token.text)

        vectorizer = TfidfVectorizer(max_features=maxFeatures)
        response = vectorizer.fit_transform(moviesPlots[:, 1])
        # print(response)
        # df_tfidf_sklearn = pd.DataFrame(response.toarray(), columns=vectorizer.get_feature_names_out())
        # print(df_tfidf_sklearn)
        return response.toarray()


def getMoviesWithSpecificGenres(genresList):
    uniqueGenres = np.load(direction + 'Y_UniqueGenres.npy', allow_pickle=True)
    indexOfGenres = []
    labels = []
    movies = []

    for genre in genresList:
        uniqueGenresFound = np.where(uniqueGenres == genre)
        if uniqueGenresFound[0].shape[0] != 0:
            indexOfGenres.append(uniqueGenresFound[0][0])

    print(str(genresList) + ' -> ' + str(indexOfGenres))
    genresArray = np.load(direction + 'Y_GenresArray.npy', allow_pickle=True)

    for i, genre in enumerate(genresArray):
        for index in indexOfGenres:
            if genre[index] == 1:
                labels.append(index)
                movies.append(i)


    print('Movies found: ' + str(len(movies)))
    unique, counts = np.unique(labels, return_counts=True)
    samples = list(zip(unique, counts))
    return indexOfGenres, movies, labels, samples


def getMoviesWithRandomGenres(genresArray, genresList):
    labels = []
    movies = []

    for i, genre in enumerate(genresArray):
        for index in genresList:
            if genre[index] == 1:
                labels.append(index)
                movies.append(i)

    print('Movies found: ' + str(len(movies)))
    unique, counts = np.unique(labels, return_counts=True)
    samples = list(zip(unique, counts))
    return genresList, movies, labels, samples


def prepareNewRandomGenres(N):
    firstRandomGenres = random.sample(range(27), N)
    secondRandomGenres = random.sample(range(27), N)
    genresArray = np.load(direction + 'Y_GenresArray.npy', allow_pickle=True)

    for i in tqdm(range(N)):
        genresSeparated = [firstRandomGenres[i], secondRandomGenres[i]]
        genresIndexes, moviesGenresIndexesArray, y, samples = getMoviesWithRandomGenres(genresArray, genresSeparated)
        var1 = text(True, moviesGenresIndexesArray, 100)
        var2 = photos(True, moviesGenresIndexesArray)
        X = np.hstack([var1, var2])
        y = LabelEncoder().fit_transform(y)
        saveNpyFiles(X, y, samples, genresIndexes)


def saveNpyFiles(xValue, yValue, fileSamples, genresIndexList):
    filename = '_'.join(str(i) for i in genresIndexList)
    with open(direction + 'data/' + 'X_' + filename + '.npy', 'wb') as f:
        np.save(f, xValue)
    with open(direction + 'data/' + 'y_' + filename + '.npy', 'wb') as f:
        np.save(f, yValue)
    with open(direction + 'samples/' + 'sample_' + filename + '.npy', 'wb') as f:
        np.save(f, fileSamples)


def manualGenres():
    Genres = ['Sci-Fi', 'Crime']
    # PREPARE
    genresIndexes, moviesGenresIndexesArray, samples, y = getMoviesWithSpecificGenres(Genres)
    print(moviesGenresIndexesArray)
    var1 = text(True, moviesGenresIndexesArray, 100)
    var2 = photos(True, moviesGenresIndexesArray)
    # X = np.hstack([var1, var2])
    # y = LabelEncoder().fit_transform(y)
    # saveNpyFiles(X, y, samples, genresIndexes)


def checkSamples():
    samplesList = []
    samplesDirection = direction + '/samples/'
    filenames = next(walk(samplesDirection), (None, None, []))[2]
    for i, file in enumerate(filenames):
        samplesList.append((file, np.load(samplesDirection + file, allow_pickle=True)))
        print(samplesList[i])


if __name__ == '__main__':
    # uniqueGenres = np.load(direction + 'Y_UniqueGenres.npy', allow_pickle=True)
    # print(uniqueGenres)

    manualGenres()
    # prepareNewRandomGenres(1)
    # manualGenres()
    # checkSamples()