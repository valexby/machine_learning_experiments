#!/usr/bin/env python
import numpy as np
import scipy.cluster.vq as vq
import sys, pandas, json
import sklearn.cluster as skl
import matplotlib.pyplot as plt


def visualize(data, kmeans, iteration):
    data = data.copy()
    labels  = kmeans.labels_
    
    data.insert(loc=len(data.columns), column='Cluster', value=labels)

    colors = {0 : 'b', 1 : 'g', 2 : 'r', 3 : 'c', 4 : 'm', 5 : 'y', 6 : 'k'}
    f = open('tsne_representation', 'r')
    tsne_representation = np.array(json.load(f))
    f.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(tsne_representation[:, 0], tsne_representation[:, 1],
            c=data['Cluster'].map(colors));
    ax.set_title('Cluster')
    fig.savefig('representation{}.png'.format(iteration), fmt='png')


def main():
    
    #Таблица из примера с хабра
    data = pandas.read_csv('churn.csv')
    data['churn'] = data['churn'].astype('int64')
    d = {'no' : 0, 'yes' : 1}
    data['international plan'] = data['international plan'].map(d)
    data['voice mail plan'] = data['voice mail plan'].map(d)
    data = data.drop(['state', 'area code', 'phone number'], axis=1)#Выкинул все ненужное, все нужные признаки сделал числовыми
    data = data / ((data ** 2).sum() ** 0.5)#Нормирую признаки
    
    #Первая кластеризация со случайными центроидами. Одна итерация алгоритма k-means
    kmeans = skl.KMeans(n_clusters=3, max_iter=1, init='random').fit(data.values)
    visualize(data, kmeans, 0)
    print(data.info())

    for i in range(1, 10):
        #На каждой итерации производим k-means длинной в одну итерацию, и инициализируем центроиды результатами прошлой итерации
        init = kmeans.cluster_centers_
        print(init)
        kmeans = skl.KMeans(n_clusters=3, max_iter=1, init=init, n_init=1).fit(data.values)
        visualize(data, kmeans, i)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
