from __future__ import print_function
from numpy import *

# k-means 聚类算法
def kMeans(dataMat, k, distMeas = distEclud, createCent = randCent):
    m = shape(dataMat)[0]               # 行数
    clusterAssment = mat(zeros((m, 2))) # 创建一个与 dataMat 行数一样，但是有两列的矩阵，用来保存簇分配结果
    centroids = createCent(dataMat, k)  # 创建质心，随机k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):              # 循环每一个数据点并分配到最近的质心中去
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :],dataMat[i, :])  # 计算数据点到质心的距离
                if distJI < minDist:    # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 簇分配结果改变
                clusterChanged = True   # 簇改变
                clusterAssment[i, :] = minIndex, minDist**2  # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
        print(centroids)
        for cent in range(k):           # 更新质心
            ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 获取该簇中的所有点
            centroids[cent, :] = mean(ptsInClust, axis = 0)  # 将质心修改为簇中所有点的平均值，mean 就是求平均值的
    return centroids, clusterAssment