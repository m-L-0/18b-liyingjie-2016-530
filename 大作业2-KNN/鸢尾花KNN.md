# 用eager模式对鸢尾花数据集进行KNN分类 #

<br/>
## 开启eager模式 ##
    tf.enable_eager_execution()
##加载及查看数据##
    iris=load_iris()
    iris.data
    # 从使用train_test_split，利用随机种子random_state采样20%的数据作为测试集。
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=33)
##tensorflow求欧氏距离##
    def Eucdist(trainSet,testSet):
    tr=tf.constant(trainSet)
    te=tf.constant(testSet)
    
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(tr-te), 1)).numpy()
    return euclidean
##求k个邻近样本的索引##
    def Neighbors(trainSet,testSet,k):
    y=Eucdist(trainSet,testSet).argsort()
    neighbors = []
    for x in range(k):
    neighbors.append(y[x])
    return neighbors
argsort()函数：将数组升序排列并返回原数组索引
##计数##
    def getRes(neighbors):
    classVotes = {}
    classVotes[0]=0
    classVotes[1]=0
    classVotes[2]=0
    for n in neighbors:
    if y_train[n]==0:
    classVotes[0]+=1
    elif y_train[n]==1:
    classVotes[1]+=1
    elif y_train[n]==2:
    classVotes[2]+=1
    sortedVotes = sorted(classVotes.items(), key=itemgetter(1), reverse=True)
    return sortedVotes[0][0]
sorted()函数：根据字典的第二域降序排列并返回第一个值<br/>
classVotes：字典，把键类型设为int类型，方便后面“准确度”的比较
##准确度##
    def getAc(testy, predictions):
    correct = 0
    for x in range(len(testy)):
    if testy[x] == predictions[x]:
    correct += 1
    return (correct/float(len(testy))) * 100.0
##传入样本集进行分类##
    K=[i for i in range(1,21)]
    Ac=[]
    for k in K:
    predictions=[]
    for i in range(30):
    nei=Neighbors(X_train,X_test[i],k)
    res=getRes(nei)
    predictions.append(res)
    Ac.append(getAc(y_test, predictions))
##可视化选出k最佳值##
    import matplotlib.pyplot as plt
     
    x = K
    y = Ac
     
    plt.plot(x, y)
     
    plt.title('The accuracy about k')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
     
    plt.show()
![](https://github.com/m-L-0/18a-liyingjie-2016-530/blob/master/%E5%A4%A7%E4%BD%9C%E4%B8%9A2-KNN/%E5%87%86%E7%A1%AE%E5%BA%A6.png)