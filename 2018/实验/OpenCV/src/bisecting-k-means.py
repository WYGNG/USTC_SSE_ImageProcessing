# -*- coding:utf-8 -*-
# kmeans : Bisecting k-means cluster(二分K-means算法)

import numpy as np
import matplotlib.pyplot as plt


def readfile(filename):
    """
    读取数据集
    W：特征向量数组，只取前两个特征
    label：标签（类别）列表
    :param filename:
    :return:特征向量数组和标签集合列表
    """
    save_path = "G:\\"
    with open(save_path + filename, 'r') as f:
        length = len(f.readlines())
        print(filename, "length: %d" % length)
        W = np.zeros((length, 2))
        label = []
        i = 0

        f.seek(0, 0)
        for line in f.readlines():
            linestr = line.strip()
            linestrlist = line.split(',')
            print(linestrlist)
            # 鸢尾属植物数据集的特征共有四个，我们这里只取前两个特征作为特征向量，当然这样分类肯定是不准确的。
            number_data = [float(j) for j in linestrlist[0:2]]
            W[i, :] = np.array(number_data)
            label.append(linestrlist[4].strip('\n'))
            i += 1
    return W, label


def createDataset(filename):
    """
    创建待分类数据集
    """
    data_vector, label_str = readfile(filename)
    # print(data_vector,"\n",label)

    # 将原始数据集中非字符串标签改为用数字代表，用户后续画图
    label_num = []
    for i in label_str:
        if i == "Iris-setosa":
            label_num.append(0)
        elif i == "Iris-versicolor":
            label_num.append(1)
        else:
            label_num.append(2)
    return data_vector, label_num


# 计算欧式距离
def euclDistance(vector1, vector2):
    return np.sqrt(sum(pow(vector2 - vector1, 2)))  # pow()是自带函数


# 使用随机样例初始化质心
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    # numSample - 行，此处代表数据集数量  dim - 列，此处代表维度，例如只有xy轴的，dim=2
    centroids = np.zeros((k, dim))  # 产生k行，dim列零矩阵
    for i in range(k):
        index = int(np.random.uniform(0, numSamples))  # 给出一个服从均匀分布的在0~numSamples之间的整数
        # print("ok",index,numSamples)
        centroids[i, :] = dataSet[index, :]  # 第index行作为簇心
    # print(centroids)
    return centroids


# k均值聚类
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # print(numSamples)
    # frist column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid

    clusterAssment = np.zeros((numSamples, 2))
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in range(numSamples):
            minDist = 1000000.0  # 最小距离
            minIndex = 0  # 最小距离对应的点群
            ## for each centroid
            ## step2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])  # 计算每个数据到每个簇中心的欧式距离
                if distance < minDist:  # 如果距离小于当前最小距离
                    minDist = distance  # 则最小距离更新
                    minIndex = j  # 对应的点群也会更新

            ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:  # 如当前数据不属于该点群
                # 此处与书本上算法步骤稍微有点不同：当有一个数据的分类错误时就clusterChanged = True ，便会重新计算簇心。而书本上的终止条件是是新簇心等于上一次迭代后的簇心
                clusterChanged = True  # 聚类操作需要继续
                clusterAssment[i, :] = minIndex, minDist ** 2

                ## step 4: update centroids
        for j in range(k):
            # 提取同一类别的向量
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0] == j)[0]]
            # print("s",pointsInCluster.shape)
            # nonzeros返回的是矩阵中非零的元素的[行号]和[列号]
            # 将所有等于当前点群j的，赋给pointsInCluster，之后计算该点群新的中心
            if len(pointsInCluster) > 0:
                centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对每列求均值

    # print("center",centroids)
    return centroids, clusterAssment


# Bisecting k-means cluster(二分k-means算法)
def bisect_kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    clusterAssment = np.zeros((numSamples, 2))  # 保存数据点信息，所属类别和误差
    centrol_address = []  # 簇心坐标，注意是列表形式

    # step1 将所有点看做一个簇
    # print(np.mean(dataSet,axis=0).shape,np.mean(dataSet,axis=0))
    centrol_address.append(np.mean(dataSet, axis=0).tolist())  # numpy.ndarray.tolist()将数组变为列表形式

    for j in range(numSamples):
        clusterAssment[j, 1] = euclDistance(dataSet[j, :], centrol_address[0]) ** 2  # 计算SSE
    classes = len(centrol_address)

    while classes < k:
        lowestSSE = float("inf")  # python 使用float("inf")表示正无穷大，float("-inf")表示负无穷大

        for i in range(classes):
            # 获得属于某一类的数据
            second_kmeans_verctor = dataSet[np.nonzero(clusterAssment[:, 0] == i)[0], :]
            # 调用上一节写的k-means方法，此时k=2，二分法。输出簇心坐标和数据聚类信息
            second_centroids, second_clusterAssment = kmeans(second_kmeans_verctor, k=2)
            # 划分数据的SSE，加上未划分数据的SSE，得到总的SSE
            SSE = sum(second_clusterAssment[:, 1]) + sum(clusterAssment[np.nonzero(clusterAssment[:, 0] != i)[0], 1])
            # 找出最小的SSE以及相关数据信息
            if SSE < lowestSSE:
                lowestSSE = SSE
                bestCentToSplit = i  # 当前最适合继续二分法划分的类别
                bestNewCentrols = second_centroids  # 二分法划分后的两个簇心向量
                bestClustAss = second_clusterAssment.copy()  # 二分法划分后的数据信息

        # 由于是二分法划分，只有0，1两个类别，将属于1类别的所属信息转为下一个新的中心点的的类别号，即等于len(centrol_address)=i+1
        # 注意这里两次类别信息的替换顺序不能错位！！！！
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centrol_address)
        # 将属于0类别的类别信息替换为本次被划分的数据信息，bestCentToSplit=i
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit

        # 这里替换中心点信息
        centrol_address[bestCentToSplit] = bestNewCentrols[0, :].tolist()
        centrol_address.append(bestNewCentrols[1, :].tolist())
        # 替换本次被用来重新划分数据的所属类别和误差信息
        clusterAssment[np.nonzero(clusterAssment[:, 0] == bestCentToSplit)[0], :] = bestClustAss
        classes = len(centrol_address)
    # 返回中心点信息和数据类别和误差信息
    return np.array(centrol_address), clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment, old_label):
    numSamples, dim = dataSet.shape  # numSample - 样例数量  dim - 数据的维度
    if dim != 2:
        print(" not two-dimensional data")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("the k is too large! the max k is 10")
        return 1

    # draw all samples     对k-means聚类后的结果对数据进行绘图
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    plt.title(" The classification results of Bisecting k-means cluster")

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']

    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], ms=12.0)

    # 按照原始数据集自带的类别画图，用于与新分类后的数据进行对比
    plt.figure()  # 打开第二个窗口显示图片，而不是分屏显示
    for i in range(numSamples):
        markIndex = int(old_label[i])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    plt.title("Original classification result")
    plt.show()


'''
函数功能： 二分 k-means聚类  
采用UCI的数据集：鸢尾属植物数据库，Link：http://archive.ics.uci.edu/ml/machine-learning-databases/iris/
为了绘图方便，只取特征空间中的前两个维度，即萼片长度、萼片宽度两个特征，当然只采用这两个特征进行分类肯定是不准确滴！！
'''
if __name__ == "__main__":
    # 数据集名称
    filename = "iris.data"

    # data_vector, label分别为特征向量和原始标签
    data_vector, label = createDataset(filename)
    # initCentroids(data_vector,3)

    k = 3
    # centroids, clusterAssment=kmeans(data_vector,k)
    centroids, clusterAssment = bisect_kmeans(data_vector, k)

    # 按照原始标签和k-means聚类后的分类分别对数据进行绘图
    showCluster(data_vector, k, centroids, clusterAssment, label)