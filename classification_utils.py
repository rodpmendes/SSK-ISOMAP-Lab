import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


#------------
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics


'''
 Performs clustering in data, returns the obtained labels and evaluates the clusters
'''
def Clustering(dados, target, DR_method, cluster):
    rand, ca, fm, v, dbs, ss = -1, -1, -1, -1, -1, -1
    labels = [-1,-2]
    try:
        print()
        print('Clustering results for %s features' %(DR_method))
        print()
        # Number of classes
        c = len(np.unique(target))
        # Clustering algorithm
        if cluster == 'kmeans':
            kmeans = KMeans(n_clusters=c, random_state=42).fit(dados.T)
            labels = kmeans.labels_
        elif cluster == 'gmm':
            labels = GaussianMixture(n_components=c, random_state=42).fit_predict(dados.T)
        else:
            ward = AgglomerativeClustering(n_clusters=c, linkage='ward').fit(dados.T)
            labels = ward.labels_
        # Computation of the cluster evaluation metrics    
        rand = rand_score(target, labels)    
        ca = calinski_harabasz_score(dados.T, labels)
        fm = fowlkes_mallows_score(target, labels)
        v = v_measure_score(target, labels)
        dbs = davies_bouldin_score(dados.T, labels)
        ss = silhouette_score(dados.T, labels)
        # Print evaluation metrics
        print('Rand index: ', rand)    
        print('Calinski Harabasz: ', ca)
        print('Fowlkes Mallows:', fm)
        print('V measure:', v)
        print('Davies Bouldin Score:', dbs)
        print('Silhouette Score:', ss)
        print()

    except Exception as e:
        print(DR_method + " -------- def Clustering error:", e)
    finally:
        return [np.float64(rand), np.float64(ca), np.float64(fm), np.float64(v), np.float64(dbs), np.float64(ss), labels.tolist()]



# Train and test eight different supervised classifiers
def Classification(dados, target, method):
    lista = []
    # 50% for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real, target, test_size=.5, random_state=42)
    # KNN
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train) 
    pred = neigh.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # SMV
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    pred = svm.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred = nb.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    pred = qda.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=1000)
    mpl.fit(X_train, y_train)
    pred = mpl.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    pred = gpc.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    pred = rfc.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados, target, metric='euclidean')
    # Computes the average accuracy
    average = sum(lista)/len(lista)
    maximo = max(lista)
    print()
    print('Maximum balanced accuracy for %s features: %f' %(method, maximo))
    print()
    return [sc, average, maximo]