import pandas as pd
import random
import math
import nltk
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


	# function that makes the dataset have 2 classes (quality can be 'good'/1 or 'bad'/0)
def preprocess(wineDF):
    bins = (0, 5.5, 8)
    quality_categories = ['bad', 'good']
    wineDF['quality'] = pd.cut(wineDF['quality'], bins=bins, labels=quality_categories)

    labelencoder = LabelEncoder()
    wineDF['quality'] = labelencoder.fit_transform(wineDF['quality'])
    return wineDF

	#funcion that returns the SVM classifier with the best parameters
def best_svm_search(wine_data):

    # print(wine_data.head())
    # print(wine_data.info())
    # print(wine_data.isnull().sum())
    # print("\n\n")

    # preprocessing
    wine_data = preprocess(wine_data)

    # print("\n\n")
    # print(wine_data['quality'].value_counts())

    features = wine_data.drop('quality', axis = 1)
    labels = wine_data['quality']

    # splitting dataset into training and test sets (75% - 25%)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25,
                                                                                train_size=0.75, random_state=35)

    # normalization
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # exhaustive search to find the best parameters for the SVM classifier 
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
                  'kernel': ['rbf', 'sigmoid', 'linear']}
    # because the value 'poly' for the parameter 'kernel' takes too much time I do not take it into consideration
    # if you want to run the search for kernel = 'poly' comment lines 55-57 and uncomment lines 60-62
    # param_grid = {'C': [0.1, 1, 10, 100],
    #               'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
    #               'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, return_train_score=True)
    grid.fit(features_train, labels_train)

    print("best parameters for C gamma and kernel: ", grid.best_params_)
    print("best score of classifier with those parameters: ", grid.best_score_)
    print("best classifier: ", grid.best_estimator_)

    best_svm_clf = grid.best_estimator_
    best_svm_clf.fit(features_train, labels_train)
    best_svm_predictions = best_svm_clf.predict(features_test)
    print(classification_report(labels_test, best_svm_predictions))
    print(confusion_matrix(labels_test, best_svm_predictions))

    print()
    print()
    print("Default classifier results")
    default_svmclf = SVC()
    default_svmclf.fit(features_train, labels_train)
    default_svm_predictions = default_svmclf.predict(features_test)
    print(classification_report(labels_test, default_svm_predictions))
    print(confusion_matrix(labels_test, default_svm_predictions))
    return best_svm_clf

	# function that makes 1/3 oh the column 'pH' equal to 'NaN'
def one_third_ph_NaN(wine_data):
    # preprocessing
    wine_data = preprocess(wine_data)

    features = wine_data.drop('quality', axis=1)
    labels = wine_data['quality']

    # splitting dataset into training and test sets (75% - 25%)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25,
                                                                                train_size=0.75, random_state=35)
    features_train = features_train.copy()
    train_indices = features_train['pH'].index.values.tolist()
    random_indices = random.sample(train_indices, 396)
    for index in random_indices:
        features_train.loc[index, 'pH'] = float('NaN')

    return features_train, features_test, labels_train, labels_test

	# function that removes the column 'pH' from the dataset
def drop_col(features_train, features_test, labels_train, labels_test):
    features_train = features_train.drop('pH', axis=1)
    features_test = features_test.drop('pH', axis=1)
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)
    best_svm = SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                   decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
                   max_iter=-1, probability=False, random_state=None, shrinking=True,
                   tol=0.001, verbose=False)
    best_svm.fit(features_train, labels_train)
    svm_predictions = best_svm.predict(features_test)
    print("Best SVM classifier results: ")
    print(classification_report(labels_test, svm_predictions))
    print(confusion_matrix(labels_test, svm_predictions))

	# function that fills 'NaN' values with the average of the values of the column that are not 'NaN'
def fill_with_avg(features_train, features_test, labels_train, labels_test):
    avg = features_train['pH'].mean(skipna=True)
    features_train = features_train.fillna(avg)
    best_svm = SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                   decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
                   max_iter=-1, probability=False, random_state=None, shrinking=True,
                   tol=0.001, verbose=False)
    best_svm.fit(features_train, labels_train)
    svm_predictions = best_svm.predict(features_test)
    print("Best SVM classifier results: ")
    print(classification_report(labels_test, svm_predictions))
    print(confusion_matrix(labels_test, svm_predictions))

	#funcion that fills NaN values with the average of cluster (after clustering)
def fill_with_avg_cluster(wine_data):
    wine_data.index.name = 'id'
    # preprocessing
    preprocess(wine_data)

    # βγάλτε τα σχόλια από τις παρακάτω γραμμές εάν επιθυμείτε να δείτε το διάγραμμα για το elbow method
    # wcss = []
    # for i in range(1, 13):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #     kmeans.fit(wine_data)
    #     wcss.append(kmeans.inertia_)
    #
    # plt.plot(wcss)
    # plt.xlabel('k (#of clusters)')
    # plt.ylabel('Inertia')
    # plt.title('The Elbow Method using Inertia')
    # plt.show()

    labeler = KMeans(n_clusters=3, init='k-means++', max_iter=300, random_state=35)
    labeler.fit(wine_data)

    predicted = labeler.predict(wine_data)
    cluster_map = pd.DataFrame()
    wine_cluster = pd.DataFrame()
    cluster_map['id'] = wine_data.index.values
    cluster_map['cluster'] = predicted
    wine_cluster = pd.merge(wine_data, cluster_map, on='id')
    wine_cluster = wine_cluster.set_index('id')

    features = wine_cluster.drop('quality', axis=1)
    result = wine_cluster['quality']

    # splitting dataset into training and test sets (75% - 25%)
    features_train, features_test, result_train, result_test = train_test_split(features, result, test_size=0.25,
                                                                                train_size=0.75, random_state=35)
    # replace 33% of the column 'pH' with 'NaN'
    features_train = features_train.copy()
    train_indices = features_train['pH'].index.values.tolist()
    random_indices = random.sample(train_indices, 396)

    for i in random_indices:
        features_train.loc[i, 'pH'] = float('NaN')

    # calculate average value of pH for every cluster
    MeanValues = {}
    for i in range(3):
        number_of_wines = 1
        MeanValues[i] = 0
        for id in features_train.index:
            if features_train.loc[id, 'cluster'] == i:
                if not (math.isnan(features_train.loc[id, 'pH'])):
                    number_of_wines += 1
                    MeanValues[i] += features_train.loc[id, 'pH']
        MeanValues[i] = MeanValues[i] / number_of_wines

    # replace NaN values for each cluster with the average calculated above
    for id in features_train.index:
        if math.isnan(features_train.loc[id, 'pH']):
            cluster = features_train.loc[id, 'cluster']
            features_train.loc[id, 'pH'] = MeanValues[cluster]

    # normalization
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    best_svm_clf = SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                       decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
                       max_iter=-1, probability=False, random_state=None, shrinking=True,
                       tol=0.001, verbose=False)
    best_svm_clf.fit(features_train, result_train)
    best_svm_predictions = best_svm_clf.predict(features_test)
    print("Best SVM classifier results: ")
    print(classification_report(result_test, best_svm_predictions))
    print(confusion_matrix(result_test, best_svm_predictions))

	#function that fills NaN values using Logistic Regression
def fill_with_logreg( wine_data):
    wine_data = pd.read_csv('winequality-red.csv', sep=',')
    wine_data.index.name = 'id'
    # preprocessing
    bins = (0, 5.5, 8)
    quality_categories = ['bad', 'good']
    wine_data['quality'] = pd.cut(wine_data['quality'], bins=bins, labels=quality_categories)

    labelencoder = LabelEncoder()
    wine_data['quality'] = labelencoder.fit_transform(wine_data['quality'])

    features = wine_data.drop('quality', axis=1)
    result = wine_data['quality']

    # splitting dataset into training and test sets (75% - 25%)
    features_train, features_test, result_train, result_test = train_test_split(features, result, test_size=0.25,
                                                                                train_size=0.75, random_state=35)
    features_train = features_train.copy()

    # replacing pH values with 0 or 1 (so that I can use logistic regression)
    bins = (2.5, 3.375, 4.5)
    pH_categories = ['low', 'high']
    features_train['pH'] = pd.cut(features_train['pH'], bins=bins, labels=pH_categories)
    features_train['pH'] = labelencoder.fit_transform(features_train['pH'])

    # removing 33% of the column pH fromt the training dataset
    features_train = features_train.copy()
    train_indices = features_train['pH'].index.values.tolist()
    random_indices = random.sample(train_indices, 396)

    for i in random_indices:
        features_train.loc[i, 'pH'] = float('NaN')

    # Χ keeps every column except pF, y keeps only the column pH 
    X = features_train.drop('pH', axis=1)
    y = features_train['pH']

    # y_train keeps every pH value that is not NaN
    # X_test keeps the rows for which the pH value is not NaN
    # X_train keeps rows for which the pH value is NaN 
    y_train = y.dropna()
    X_temp = X
    for id in y_train.index:
        X_temp = X_temp.drop(id, axis=0)
    X_test = X_temp

    X_temp2 = X
    for id in X_temp.index:
        X_temp2 = X_temp2.drop(id, axis=0)
    X_train = X_temp2

    # training Logistic Regression Classifier to classify pH
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)

    # predicting pH for the rows that is has no value (NaN)
    ph_predictions = logreg.predict(X_test)
    X_test['pH'] = ph_predictions

    # replacing NaN values with the predicted values 
    for id in X_test.index:
        features_train.loc[id, 'pH'] = X_test.loc[id, 'pH']

    # use of the best classifier
    best_svm_clf = SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                       decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
                       max_iter=-1, probability=False, random_state=None, shrinking=True,
                       tol=0.001, verbose=False)
    best_svm_clf.fit(features_train, result_train)
    best_svm_predictions = best_svm_clf.predict(features_test)
    print("Best SVM classifier results: ")
    print(classification_report(result_test, best_svm_predictions))
    print(confusion_matrix(result_test, best_svm_predictions))


def newral_network(onion_data):
    # splitting titles into tokens (words) and storing them in a dicionary ,removing punctuation marks 
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = {}
    labels = []
    for i in onion_data.index:
        tokens[i] = tokenizer.tokenize(onion_data.loc[i, 'text'])
        labels.append(onion_data.loc[i, 'label'])

    # make all words be in lower-case letters 
    for title in tokens:
        tokens[title] = [word.lower() for word in tokens[title]]

    # stemmer algorithm (keep only the prefix for every word)
    stemmer = PorterStemmer()
    stemmed_titles = {}
    for title in tokens:
        stemmed_titles[title] = [stemmer.stem(word) for word in tokens[title]]

    # remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_titles = {}
    for title in stemmed_titles:
        filtered_titles[title] = [word for word in stemmed_titles[title] if not word in stop_words]

    # uniny the words so that we have titles (sentences) and not words 
    for title in filtered_titles:
        filtered_titles[title] = " ".join(filtered_titles[title])

    # find tf-tdf value for each word in the titles 
    corpus = []
    for title in filtered_titles:
        corpus.append(filtered_titles[title])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    # put tf-idf into a DataFrame and add the column 'label' to it 
    finalDF = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    finalDF['label'] = labels

    features = finalDF.drop('label', axis=1)
    result = finalDF['label']
    # splitting dataset into training and test sets (75% - 25%)
    features_train, features_test, result_train, result_test = train_test_split(features, result, test_size=0.25,
                                                                                train_size=0.75, random_state=35)

    # MLP classifier
    clf = MLPClassifier()
    clf.fit(features_train, result_train)
    predictions = clf.predict(features_test)
    print("Multi Layer Perceptron Results")
    print(classification_report(result_test, predictions))
    print(confusion_matrix(result_test, predictions))


# main
wine = pd.read_csv('winequality-red.csv', sep=',') #dataset

# find best parameters for svm classifier and print scores
#
# best_svm = best_svm_search(wine)


# after removing 33% of the column 'pH' and using the best svm classifier we have found

# remove completely the column pH and print scores of classifier 
#
# features_train, features_test, labels_train, labels_test = one_third_ph_NaN(wine) #33% of pH = NaN
# drop_col(features_train, features_test, labels_train, labels_test)             #remove column and classify


# fill NaN values with the average of the values of the column that are not 'NaN'
#
# features_train, features_test, labels_train, labels_test = one_third_ph_NaN(wine) #33% of pH = NaN
# fill_with_avg(features_train, features_test, labels_train, labels_test)       #fill with avg and classify


# fill NaN values using logistic Regression 
#
# fill_with_logreg(wine)    


# fill NaN values using KMeans (clustering)
#
# fill_with_avg_cluster(wine) 


onion = pd.read_csv('onion-or-not.csv') #dataset

# preprocess titles of movies and classify using neural networks (MLP classifier)
#
# newral_network(onion)


