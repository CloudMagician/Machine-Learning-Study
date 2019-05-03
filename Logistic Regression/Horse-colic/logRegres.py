import numpy as np
import random

# sigmoid阶跃函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x)) 


# 随机梯度
def SGA(dataMatIn, classLabels, num = 100):
    # 矩阵
    m,n = np.shape(dataMatIn)
    trainWeights = np.ones(n)

    # 随机梯度
    for j in range(num):
        data = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01    # alpha会随着迭代不断减小
            index = int(random.uniform(0,len(data)))
            h = sigmoid(sum(dataMatIn[data[index]] * trainWeights))
            error = classLabels[data[index]] - h
            trainWeights = trainWeights + alpha * error * dataMatIn[data[index]]
            del(data[index])
    return trainWeights


# 分类函数
def classify(x, trainWeights):
    prob = sigmoid(sum(x * trainWeights))
    if prob > 0.5: 
        return 1.0
    else: 
        return 0.0


# 训练
def colicTrain():
    # 打开文件
    frTrain = open('horseColicTraining.txt')

    # 解析特征和标签
    trainingSet = []    # 特征
    trainingLabels = [] # 标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))

    # 随机梯度
    trainWeights = SGA(np.array(trainingSet), trainingLabels, 500)
    return trainWeights


# 测试
def colicTest():
    # 打开文件
    frTest = open('horseColicTest.txt')

    # 测试特征和标签
    numTestVec = 0.0
    testSet = []    # 特征
    testLabels = [] # 标签
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[21]))

    numTests = 10
    errorSum = 0.0
    for j in range(numTests):
        errorCount = 0
        # 测试
        for i in range(int(numTestVec)):
            if int(classify(np.array(testSet[i]), colicTrain())) != int(testLabels[i]):
                errorCount += 1
        errorRate = (float(errorCount) / numTestVec)
        print ("%d - errorRate: %f" % (j, errorRate))
        errorSum += colicTrain()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))) 
    
colicTest()