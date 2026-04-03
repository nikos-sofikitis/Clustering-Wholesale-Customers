import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn import mixture
from scipy.cluster.hierarchy import dendrogram, linkage, complete
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text

data = pd.read_csv('Wholesale customers data.csv')
#dropping region and channel
data = data.iloc[:,2:]
print('columns:',data.columns)

print('Data Set size :\n',data.shape)
data.drop_duplicates(inplace=True)
print('\nDataset size w/o duplicates: \n',data.shape)


print('\nMissing values :\n',data.isnull().sum())

scaler = StandardScaler()
data = scaler.fit_transform(data)

from sklearn.decomposition import PCA


pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.title("Quick Overview of our Data(PCA)")
plt.show()

#Kmeans
print('\nClustering with Kmeans...\n')
def kmeansClustering(x):
    inertiasAll = []
    silhouettesAll = []
    clustersAll = []
    maxClusters = 12

    for n in range(2, maxClusters):
        # print 'Clustering for n=',n
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(data)
        y_kmeans = kmeans.predict(x)

        # get cluster centers
        kmeans.cluster_centers_

        # evaluate
        print(f"inertia for {n} clusters , had score of {round(kmeans.inertia_,2)}")

        silhouette_values = silhouette_samples(x, y_kmeans)
        print (f"silhouette for {n} clusters ,had a score of {round(np.mean(silhouette_values),2)}")

        inertiasAll.append(round(kmeans.inertia_,2))

        silhouettesAll.append(round(float(np.mean(silhouette_values)), 2))
        clustersAll.append(n)

#Kmeans Plotting
    plt.figure(2)
    plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, s=20, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title('Visualization k-means clsuter')

    plt.figure(3)
    plt.plot(range(2, 12), silhouettesAll, 'r*-')
    plt.ylabel('Silhouette score')
    plt.xlabel('Number of clusters')
    plt.title('Kmeans Silhouette')

    plt.figure(4)
    plt.plot(range(2, 12), inertiasAll, 'g*-')
    plt.ylabel('Inertia Score')
    plt.xlabel('Number of clusters')
    plt.title('Elbow Method(K-means)')
    plt.show()



    return clustersAll, silhouettesAll, inertiasAll

print(kmeansClustering(data))

print('\nClustering with Dbscan...\n')

#dbscan
#best parameters space
eps = [1,1.5,1.8,2]
min_samples = [2,3,4,5,6,7,8,9,10]

for e in eps:
    for m in min_samples:
        db = DBSCAN(eps=e, min_samples=m)
        db.fit(data)
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            score = silhouette_score(data, labels)
            print(f"For eps={e} and min_samples={m}, silhouette is {score:.2f} and they form {n_clusters} clusters")

        else:
            print(f"Not good Separation for {e} and {m}")

#Best DBSCAN
best_eps = 1.8
best_min_samples = 2

best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples).fit(data)
best_labels_db = best_dbscan.labels_

core_samples_mask = np.zeros_like(best_dbscan.labels_, dtype=bool)
core_samples_mask[best_dbscan.core_sample_indices_] = True

n_clusters1 = len(set(best_labels_db)) - (1 if -1 in best_labels_db else 0)
print(f"The best eps {best_eps} , min_samples {best_min_samples} forming {n_clusters1} clusters with silhouette score {round(silhouette_score(data, best_labels_db),2)}")


#TREE CHARACTERIZATION Dbscan
perc=0.2

X_train, X_test, Y_train, Y_test = train_test_split(data, best_dbscan.labels_, test_size=perc)

Dt_Dbscan =  tree.DecisionTreeClassifier()

Dt_Dbscan.fit(X_train, Y_train)


feature_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']

print('Tree rules=\n', tree.export_text(Dt_Dbscan, feature_names=feature_cols))

#  Confusion Matrix DBSCAN
Y_train_pred_DT_DB = Dt_Dbscan.predict(X_train)
Y_test_pred_DT_DB = Dt_Dbscan.predict(X_test)



print('\nConfusion Matrix Train:\n', confusion_matrix(Y_train, Y_train_pred_DT_DB))
print('\nConfusion Matrix Test:\n', confusion_matrix(Y_test, Y_test_pred_DT_DB))


#Plot for the best Dbscan Model
plt.figure(1)

unique_labels = set(best_labels_db)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (best_labels_db == k)

    # core nodes
    xy = data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10)


    # border nodes
    xy = data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=4)


plt.title('Estimated number of clusters: %d(Dbscan)' % n_clusters1)
plt.show()


#GMM
print('\nClustering with GMM...\n')
def gaussianMixture(x):
    bicAll = []

    logLikelihood = []
    clustersAll = []
    maxClusters = 15
    silhouettesAll = []

    for n in range(2, maxClusters):
        gmm = mixture.GaussianMixture(n_components=n, covariance_type='full').fit(x)
        logLikelihood.append(gmm.score(x))
        clustersAll.append(n)
        bicAll.append(gmm.bic(x))
        labels = gmm.predict(x)
        silhouetteScore = metrics.silhouette_score(x, labels)
        silhouettesAll.append(silhouetteScore)
        bic_5 = pd.Series(bicAll).sort_values(ascending=True)
        print ('mixtures, clusters', n, gmm.bic(x))
    bic_sort = pd.Series(bicAll,index=[clustersAll])
    bic_score = bic_sort.min()
    best_n = bic_sort.idxmin()

    print(f"\nBest Cluster Num with best Bic Score :  {int(best_n[0])} clusters and bic score: {round(bic_score,1)}")

    return bicAll, logLikelihood, clustersAll, silhouettesAll
print(gaussianMixture(data))

 #BEST GMM MODEL
Gmm1 = mixture.GaussianMixture(n_components=8, covariance_type='full').fit(data)
labels_best_Gmm = Gmm1.predict(data)

#plotting for GMM
plt.figure(2)
plt.scatter(data[:, 0], data[:, 1], c=labels_best_Gmm, s=50, cmap='viridis')
plt.title('GMM Clustering')
plt.show()

#TREE CHARACTERIZATION GMM
perc=0.2

X_train, X_test, Y_train, Y_test = train_test_split(data, labels_best_Gmm, test_size=perc)

Dt_gmm1 =  tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=8)

Dt_gmm1.fit(X_train,Y_train)


feature_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']

print('Tree rules=\n', tree.export_text(Dt_gmm1, feature_names=feature_cols))


Y_train_pred_DT_GMM = Dt_gmm1.predict(X_train)
Y_test_pred_DT_GMM = Dt_gmm1.predict(X_test)



print('\nConfusion Matrix Train:\n', confusion_matrix(Y_train, Y_train_pred_DT_GMM))
print('\nConfusion Matrix Test:\n', confusion_matrix(Y_test, Y_test_pred_DT_GMM))


print('Clustering with Agglomerative...')

linkeage_list = ['ward' , 'complete','average','single']
cluster_list = [2,3,4,5,6]
results_list = []

#Hyperparameter tuning
for k in cluster_list:
    for link in linkeage_list:
        hierarchical_cluster = AgglomerativeClustering(n_clusters=k, linkage=link).fit(data)
        labels = hierarchical_cluster.fit_predict(data)

        #score of Hierarchical
        score = silhouette_score(data, labels)


        results_list.append({
            'Clusters': k,
            'Linkage': link,
            'Silhouette': round(score, 3)
        })
results_df = pd.DataFrame(results_list)
print("\n--- Agglomerative Clustering Results ---")
print(results_df.sort_values(by='Silhouette', ascending=False).head(1))

#best Hierarchical(Agglomerative)
hierarchical_cluster1 = AgglomerativeClustering(n_clusters=2, linkage='complete').fit(data)
labels1 = hierarchical_cluster1.fit_predict(data)

        #silhouette score of Hierarchical
score = silhouette_score(data, labels1)

#TREE CHARACTERIZATION AGGLOMERATIVE
perc=0.2

X_train, X_test, Y_train, Y_test = train_test_split(data, hierarchical_cluster1.labels_, test_size=perc)

Dt_Agglomerative =  tree.DecisionTreeClassifier()

Dt_Agglomerative.fit(X_train, Y_train)


feature_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']

print('Tree rules=\n', tree.export_text(Dt_Agglomerative, feature_names=feature_cols))

# Confusion Matrices FOR AGGLOMERATIVE CLUSTERING
Y_train_pred_DT = Dt_Agglomerative.predict(X_train)
Y_test_pred_DT = Dt_Agglomerative.predict(X_test)



print('\nConfusion Matrix Train:\n', confusion_matrix(Y_train, Y_train_pred_DT))
print('\nConfusion Matrix Test:\n', confusion_matrix(Y_test, Y_test_pred_DT))



#plotting Agglomerative
plt.figure(2)
plt.scatter(data[:, 0], data[:, 1], c=hierarchical_cluster1.labels_, s=50, cmap='viridis')
plt.title('Agglomerative Clustering')
plt.show()

Z = linkage(data, method='complete')

plt.figure(figsize=(12, 7))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points (Cluster Size)')
plt.ylabel('Distance')
plt.legend('2 clusters Delicatessen > 10.90 ', loc='upper left')

#Plotting Dendrogram
#numbers without parenthesis is axis x is a specific client depicting him as an outlier.
#numbers with parenthesis represent the population of a cluster.
dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=0,
    leaf_font_size=12.,
    show_contracted=True
)

plt.show()