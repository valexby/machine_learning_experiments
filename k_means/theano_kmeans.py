#!/usr/bin/env python
import numpy as np
from scipy.spatial.distance import euclidean
import sys, pdb, seaborn, pandas, warnings, json 
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano import function, config, shared

#warnings.filterwarnings("ignore")#For presentation only!

def klp_kmeans(raw_data, cluster_num, alpha, epochs = -1, batch = 1, verbose = False, use_gpu=False):   
    '''
        Theano based implementation, likely to use GPU as well with required Theano
        configurations. Refer to http://deeplearning.net/software/theano/tutorial/using_gpu.html
        for GPU settings

        Inputs:
            data - [instances x variables] matrix of the data.
            cluster_num - number of requisite clusters 
            alpha - learning rate 
            epoch - how many epoch you want to go on clustering. If not given, it is set with
                Kohonen's suggestion 500 * #instances
            batch - batch size. Larger batch size is better for Theano and GPU utilization 
            verbose - True if you want to verbose the algorithm's iterations

        Output:
            W - final cluster centroids
    '''
    data = raw_data.as_matrix()

    #warnings.simplefilter("ignore", DeprecationWarning)
    #warnings.filterwarnings("ignore")

    rng = np.random
    # From Kohonen's paper
    if epochs == -1:
        print(data.shape[0])
        epochs = 500 * data.shape[0]

    # Symmbol variables
    X = T.dmatrix('X')
    WIN = T.dmatrix('WIN')

    # Init weights random
    W = np.ndarray((cluster_num, data.shape[1])) 
    for i in range(W.shape[0]):
        indx = int(rng.rand() * data.shape[0])
        W[i] = data[indx]
    W = theano.shared(W, name="W")
    W_old = W.get_value()

    # Find winner unit
    bmu = ((W**2).sum(axis=1, keepdims=True) + (X**2).sum(axis=1, keepdims=True).T - 2*T.dot(W, X.T)).argmin(axis=0)
    dist = T.dot(WIN.T, X) - WIN.sum(0)[:, None] * W
    err = abs(dist).sum()/X.shape[0]

    update = function([X,WIN],outputs=err,updates=[(W, W + alpha * dist)], allow_input_downcast=True)
    find_bmu = function([X], bmu, allow_input_downcast=True)

    if any([x.op.__class__.__name__ in ['Gemv', 'CGemv', 'Gemm', 'CGemm'] for x in
        update.maker.fgraph.toposort()]):
        print('Used the cpu')
    elif any([x.op.__class__.__name__ in ['GpuGemm', 'GpuGemv'] for x in
        update.maker.fgraph.toposort()]):
        print('Used the gpu')
    else:
        print('ERROR, not able to tell if theano used the cpu or the gpu')
        print(update.maker.fgraph.toposort())
    
    
    # Update
    labels = np.ndarray((epochs, data.shape[0]))
    for epoch in range(epochs):
        C = 0
        for i in range(0, data.shape[0], batch):
            batch_data = data[i:i+batch, :]
            D = find_bmu(batch_data)
            S = np.zeros([batch_data.shape[0],cluster_num])
            S[:,D] = 1
            for j in range(batch):
                labels[epoch][i + j] = D[j]
            cost = update(batch_data, S)
       
        if epoch%10 == 0 and verbose:
            print("Avg. centroid distance -- ", cost.sum(),"\t EPOCH : ", epoch)
    
    return labels, W.get_value()


def visualize(data, labels, iteration):
    data = data.copy()
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
    fig.savefig('representation{}.png'.format(chr(iteration + ord('a'))), fmt='png')
    plt.close()


def main():
    
    #Таблица из примера с хабра
    data = pandas.read_csv('churn.csv')
    data['churn'] = data['churn'].astype('int64')
    d = {'no' : 0, 'yes' : 1}
    data['international plan'] = data['international plan'].map(d)
    data['voice mail plan'] = data['voice mail plan'].map(d)
    data = data.drop(['state', 'area code', 'phone number'], axis=1)#Выкинул все ненужное, все нужные признаки сделал числовыми
    data = data / ((data ** 2).sum() ** 0.5)#Нормирую признаки
    
    labels, _ = klp_kmeans(data, 3, alpha=0.001, batch=1, verbose=True, epochs=30)
    for i in range(labels.shape[0]):
        visualize(data, labels[i], i)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
