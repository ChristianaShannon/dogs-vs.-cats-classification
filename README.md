# 猫狗大战项目的详细实现 #

#### 本项目参考 [猫狗大战](https://github.com/ypwhs/dogs_vs_cats)

## 0. 项目背景 ##
猫狗大战来源于Kaggle上的一个热门比赛，传送门：[Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)
该项目包括25000张图片的训练集和12500张图片的测试集，训练集中，猫狗图片各占一半；

## 1. 数据预处理 ##
- 由于本文使用Keras 的 ImageDataGenerator，因此在进行数据预处理时，会把训练集中的猫狗图片分开，并将测试数据集再额外放置于一个文件夹中（套一层‘外套’）

- 在开始训练之前，我先切分出了10%的图片为验证集，比起在Keras中使用valid_split, 这样做的好处是数据集结构比较清晰，且在开始时就打乱了数据集，不会出现在训练时忘了打乱验证集的情况：
 

> How is the validation split computed?

> If you set the validation_split argument in model.fit to e.g. 0.1, then the validation data used will be the last 10% of the data. If you set it to 0.25, it will be the last 25% of the data, etc. Note that the data isn't shuffled before extracting the validation split, so the validation is literally just the last x% of samples in the input you passed.

- 在进行数据集文件夹分类时，我采用了python中的 shutil.move 函数，可以自动地将不同图片文件分类放置，eg:

```
os.makedirs('test3'+'/test')
for file in os.listdir('test'):
    shutil.move('test/' + file,'test3/test/' + file)
```


最终获得数据集文件结构为：

```
├── train.zip
├── test.zip
├── test [0 images]
├── test3
│   └── test [12500 images]
├── train [0 images]
├── train3 [total 22500 images]
│    ├── cat 
│    └── dog 
└── valid3  [total 2500 images]
     ├── cat 
     └── dog 
```

## 2. Transfor Learning 模型 ##

在项目中，我先采用普通的Transfor Learning并进行fine tune，在这里，我使用Keras中的预训练模型：Resnet50，Inception V3，Xception，并使用预训练权重，最终取得的结果为：

| Model Name |  LogLoss score | 
| - | :-: | 
| Resnet50 |0.09039 | 
| Inception V3 | 0.08367 | 
| Xception | 0.08866 |
| Xception with data augmentation | 0.07847 |

具体可以参阅同名notebook（eg. Xception.ipynb）

## 3. Stacking 模型 ##
在完成以上工作后，我参照模型融合的思想，将Keras提供的预训练模型都使用了起来，具体包括5个模型：Resnet50，Inception V3，Xception，DenseNet201，InceptionResnetV2； 也就是说，使用这5个模型从原图片里提取猫狗的“特征”，再把这些特征作为输入，进行模型预测；

这样做的优势：
- logloss 显著降低，相当于将几个很强的模型“并联”，实现 stacking
- 预训练模型向量在训练完后可以保存，只需要再构建一个很小的网络，就可以使用这些向量权重，迅速进行预测

最终将预训练获得的特征保存为h5文件，每个h5文件5 文件包括五个 numpy 数组：

* train (22500, features)
* valid（2500，features）
* test (12500, features)
* train_label (22500,)
* valid_label (2500,)

其中features为该模型提取出来的特征；对于Resnet50，Inception V3，Xception，特征值为2048，对于DenseNet201，该值为1920，对于InceptionResnetV2，为1536；

## 4. Stacking 模型 预测调优 ##

####  模型预测： ####

将提取出的特征融合，变为新的模型输入：

```
X_train = []
X_valid = []
X_test = []

for filename in ["pre_ResNet50s.h5", "pre_Xceptions.h5", "pre_InceptionV3s.h5","pre_InceptionResNetV2s.h5","pre_DenseNet201s.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_valid.append(np.array(h['valid']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['train_label'])
        y_valid = np.array(h['valid_label'])

X_train = np.concatenate(X_train, axis=1)
X_valid = np.concatenate(X_valid, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train, y_train = shuffle(X_train, y_train)
```

然后构建一个两层的简单的预测模型：

```
input_tensor = Input(X_train.shape[1:])
x = input_tensor

x = BatchNormalization(axis=1)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)
```
在构建模型时，我使用了BN层，对前面各个模型提取出的features进行Normalization；

另外，在最终预测结果时，我参照原例，使用了clip方法，将预测输出限定在[0.005,0.995]内，事实证明，clip方法对降低Logloss确实有作用。

最终结果（2 epoch）：

| Model Name |  LogLoss score | 
| - | :-: | 
| Stacking model |0.03941 | 

####  预测结果加权： ####
在运行预测代码时，我发现模型在4epoch 就会产生过拟合，导致Logloss上升，而2 epoch 和3 epoch 获得的valid Loss 差不多，由此，我将2 epoch 和 3 epoch的结果进行加权：

```
df_a = pd.read_csv("epoch2.csv")
df_b = pd.read_csv("epoch3.csv")

pred_mix= np.zeros(12500)
for i,a,b in zip(df_a['id'],df_a['label'],df_b['label']):
    if a>=0.5 and b>=0.5:
        pred_mix[i-1]=np.max((a,b)) 
    elif a<0.5 and b<0.5:
        pred_mix[i-1]=np.min((a,b))
    else:
        pred_mix[i-1]=np.mean((a,b))
    df_a.set_value(i-1,'label',pred_mix[i-1])

df_a.to_csv('pred_mix.csv',index=None)
df_a.head(10)
```
结果有所提升，Logloss 有所下降：

| Model Name |  LogLoss score | 
| - | :-: | 
| Stacking model-average|0.03840 | 

该结果位于Leardboard的12名，进入了前1%；

如果要进一步提升，可以增加数据增强，或者在stacking模型的预训练模型中放开更多的层。

###  参考链接： ###

 1. [ResNet](https://arxiv.org/abs/1512.03385)
 2. [Inception v3](https://arxiv.org/abs/1512.00567) 
 3. [Xception](https://arxiv.org/abs/1610.02357) 
 4. [Inception-v4](https://arxiv.org/abs/1602.07261)
 5. [DenseNet](https://arxiv.org/abs/1608.06993)
 6. [Keras 文档](https://keras.io/)
