# SVM高光谱

加载.mat文件，print查看数据内容，

打开数据集部分['data2_train']

## 预处理

将类标签加入各类数据集

将所有类合并成一个sample

```python
sample=dt2
for i in [dt3,dt3,dt5,dt6,dt8,dt10,dt11,dt12,dt14]:
    sample=np.vstack((sample,i))
print(sample.shape)
```

将sample和test中data部分分别归一化（不要包括label列）

```python
data_D=sample[:,1:]
data_D=pd.DataFrame(data_D)
data_D=(data_D-min(data_D))/(max(data_D)-min(data_D))

test_data=pd.DataFrame(test_data)
test_data=(test_data-min(test_data))/(max(test_data)-min(test_data))
```



## 模型训练与拟合

因为是多分类，所以调用SVC

## 调参并得出准确度

经测验C=130时，准确度最高

准确度：0.9567137809187279

## 预测测试集并保存成csv文件

具体结果见“test_pred.csv”文件

## 模型不足

预处理没有去噪声

SVC模型较为复杂，内里详细原理没有搞明白，无法自己写实现代码，遂只能掉包

参数标准没有研究，遂选择数据纯靠直觉