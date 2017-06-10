#!/usr/bin/env python
import numpy as np
import sys, pdb, seaborn, warnings, json
import pyspark as spark
import matplotlib.pyplot as plt
from pyspark.ml.linalg import Vectors
from pyspark.mllib.clustering import KMeans, KMeansModel

#warnings.filterwarnings("ignore")#For presentation only!

def visualize(rdd, model, iteration):
    colors = {0 : 'b', 1 : 'g', 2 : 'r', 3 : 'c', 4 : 'm', 5 : 'y', 6 : 'k'}
    f = open('tsne_representation', 'r')
    tsne_representation = np.array(json.load(f))
    f.close()

    labels = model.predict(rdd).collect() #Определяем кластеры для центроид
    labels = [colors[x] for x in labels]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(tsne_representation[:, 0], tsne_representation[:, 1],
            c=labels);
    ax.set_title('Cluster')
    fig.savefig('representation{}.png'.format(iteration), fmt='png')

def read_csv(sc):
    rdd = sc.textFile('churn.csv')
    head = dict([(rdd.first().split(',')[x], x) for x in range(len(rdd.first().split(',')))]) #Получаем словарь, 
            #где название параметра из csv соответствует индексу параметра в векторах.
    rdd = rdd.zipWithIndex()


    def prepare_vect(x):#Функция, которая из троки csv таблицы делает вектор
        ls = x[0].split(',')
        ls[head['churn']] = 0 if ls[head['churn']] == 'False' else 1
        ls[head['international plan']] = 0 if ls[head['international plan']] == 'no' else 1
        ls[head['voice mail plan']] = 0 if ls[head['voice mail plan']] == 'no' else 1
        ls = ls[:head['phone number']] + ls[head['phone number'] + 1 :]
        ls = ls[:head['area code']] + ls[head['area code'] + 1 :]
        ls = ls[:head['state']] + ls[head['state'] + 1 :]
        ls = [float(x) for x in ls]
        return ls

    rdd = rdd.filter(lambda x: x[1] != 0) #Выбрасываем из RDD вектор с названиями параметров.
    rdd = rdd.map(prepare_vect)
    #Далее нормирование векторов
    def sum_vect(a, b):
        res = []
        for i in range(len(a)):
            res.append(a[i] + b[i])
        return res

    

    coef = rdd.map(lambda x: [i ** 2 for i in x]).reduce(sum_vect)
    coef = [x ** 0.5 for x in coef]

    def divide_vect(a):
        res = []
        for i in range(len(a)):
            res.append(a[i] / coef[i])
        return res

    rdd = rdd.map(divide_vect)

    return rdd
    

def main():

    sc = spark.SparkContext()#Инициализируем контекст, который налаживает соединение с кластером.
    rdd = read_csv(sc)
    clusters = KMeans.train(rdd, 3, maxIterations=1)#Одна итерация обучения выборкой

    visualize(rdd, clusters, 0)
    for i in range(1, 10):
        clusters = KMeans.train(rdd, 3, maxIterations=1, initialModel=clusters, epsilon=0.000001)
        visualize(rdd, clusters, i)

    return 0

if __name__ == '__main__':
    sys.exit(main())
