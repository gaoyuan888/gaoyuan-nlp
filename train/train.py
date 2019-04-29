# coding=gbk

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings
from keras.models import model_from_json
warnings.filterwarnings("ignore")

# 我们使用tensorflow的keras接口来建模
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# 一、词向量模型
cn_model = KeyedVectors.load_word2vec_format('../conf/sgns.zhihu.bigram', binary=False)

# 1.词向量维度，一个词向用300个维度向量表示，embedding_dim=300
embedding_dim= cn_model['山东大学'].shape[0]
print('词向量长度{}',format(embedding_dim))
array=cn_model['山东大学']
#打印词向量
print("array",array)

# 2.相似度，余弦相似度
similar=cn_model.similarity('感冒','橘子')
print("similar",similar)
# 余弦相似度计算方法    点积（矩阵/范数，矩阵/范数）  dot(["橘子"]/|["橘子"],["橘子"]/|["橘子"])
similar_=np.dot(cn_model["橘子"]/np.linalg.norm(cn_model["橘子"]),cn_model["橙子"]/np.linalg.norm(cn_model["橙子"]))
print("similar_",similar_)

# 3.寻找不是同一类的词语
test_words="老师 会计师 程序员 律师 医生 老人"
test_words_res=cn_model.doesnt_match(test_words.split())
print('在'+test_words+"中，不是同一类的词为：%s"%test_words_res)

# 4.找出相近的词语，余弦相似度
similar_1= cn_model.most_similar(positive=['大学'],topn=10)
print("similar_1",similar_1)

# 5.求一个词语在词向量中的索引
index1=cn_model.vocab["山东大学"].index
print("index1",index1);

# 6.根据索引求索引的词语  215 对应的此为老师
word=cn_model.index2word[215];
print("word",word);

# 二、样本训练
import os

path_prefix="data"
# 1.获取样本索引，样本存放在两个文件夹下面
# 每个文件是一个评价
pos_txts=os.listdir(path_prefix+"/pos")
neg_txts=os.listdir(path_prefix+"/neg")
print("样本总共"+str(len(pos_txts)+len(neg_txts))+"条");

# 2.现在我们将所有的评价内容放置到一个list里
# 存储所有评价，每例评价为一条string
train_texts_orig = []

# 添加完所有样本之后，train_texts_orig为一个含有4000条文本的list
# 其中前2000条文本为正面评价，后2000条为负面评价
for i in range(len(pos_txts)):
    with open(path_prefix+'/pos/'+pos_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()
for i in range(len(neg_txts)):
    with open(path_prefix+'/neg/'+neg_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()


# 3.进行分词和tokenize(索引化)
# train_tokens是一个长长的list，其中含有4000个小list，对应每一条评价
train_tokens = []
for text in train_texts_orig:
    # 去掉标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 结巴分词
    cut = jieba.cut(text)
    # 结巴分词的输出结果为一个生成器
    # 把生成器转换为list
    cut_list = [ i for i in cut ]
    for i, word in enumerate(cut_list):
        try:
            # 将词转换为词向量中的索引index
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            # 如果词不在字典中，则输出0
            cut_list[i] = 0
    train_tokens.append(cut_list)

# 4.获得所有tokens的长度
num_tokens = [len(tokens) for tokens in train_tokens]
num_tokens = np.array(num_tokens)

# 5.平均tokens的长度
avglen=np.mean(num_tokens)
print("所有词的平均长度",avglen)

# 6.最长的评价tokens的长度
maxlen=np.max(num_tokens)
print("最长的文本长度",maxlen)


# 7.观察一下样本分布情况
plt.hist(num_tokens,bins = 100)
plt.xlim((0,400))
plt.ylabel('number of tokens')
plt.xlabel('length of tokens')
plt.title('Distribution of tokens length')
plt.show()

# 8.样本正态分布
plt.hist(np.log(num_tokens), bins = 100)
plt.xlim((0,10))
plt.ylabel('number of tokens')
plt.xlabel('length of tokens')
plt.title('Distribution of tokens length')
plt.show()

# 9.取tokens平均值并加上两个tokens的标准差，
# 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
print("覆盖百分之九十五的文本长度",max_tokens)

# 10.取tokens的长度为236时，大约95%的样本被涵盖
# 我们对长度不足的进行padding，超长的进行修剪
pricent= np.sum( num_tokens < max_tokens ) / len(num_tokens)
print("取值文本长度覆盖样本长度百分比",pricent)

# 11.用来将tokens转换为文本
def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index2word[i]
        else:
            text = text + ' '
    return text

# 12.经过tokenize再恢复成文本
# 可见标点符号都没有了
reverse = reverse_tokens(train_tokens[0])
print("索引转化为句子后第一个句子：",reverse)
# 原始文本
print("原始文本：",train_texts_orig[0])


# 13.准备Embedding Matrix
# 现在我们来为模型准备embedding matrix（词向量矩阵），
# 根据keras的要求，我们需要准备一个维度为(numwords, embeddingdim)的矩阵，
# num words代表我们使用的词汇的数量，emdedding dimension在我们现在使用的预训练词向量模型中是300，
# 每一个词汇都用一个长度为300的向量表示。
# 注意我们只选择使用前50k个使用频率最高的词，
# 在这个预训练词向量模型中，一共有260万词汇量，如果全部使用在分类问题上会很浪费计算资源，
# 因为我们的训练样本很小，一共只有4k，如果我们有100k，200k甚至更多的训练样本时，在分类问题上可以考虑减少使用的词汇量。

print("embedding_dim:",embedding_dim)

# 14.初始化embedding_matrix
# 只使用前20000个词
num_words = 100000
# 初始化embedding_matrix，之后在keras上进行应用
embedding_matrix = np.zeros((num_words, embedding_dim))
# embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
# 维度为 50000 * 300
for i in range(num_words):
    # 从预加载的词向量模型中，选取前5w个词语（高频词），填充
    embedding_matrix[i,:] = cn_model[cn_model.index2word[i]]
embedding_matrix = embedding_matrix.astype('float32')


# 15.检查index是否对应，
# 输出300意义为长度为300的embedding向量一一对应
npsum = np.sum(cn_model[cn_model.index2word[333]] == embedding_matrix[333])
print("npsum",npsum)

# 16.确认一下词向量embedding_matrix的维度，
# 这个维度为keras的要求，后续会在模型中用到
emb_dim=embedding_matrix.shape
print("词向量的维度：",emb_dim)


# 17.进行padding和truncating， 输入的train_tokens是一个list
# 返回的train_pad是一个numpy array

# padding（填充）和truncating（修剪）
# 我们把文本转换为tokens（索引）之后，每一串索引的长度并不相等，
# 所以为了方便模型的训练我们需要把索引的长度标准化，
# 上面我们选择了236这个可以涵盖95%训练样本的长度，接下来我们进行padding和truncating，
# 我们一般采用'pre'的方法，这会在文本索引的前面填充0，因为根据一些研究资料中的实践，
# 如果在文本索引后面填充0的话，会对模型造成一些不良影响。
train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                            padding='pre', truncating='pre')


# 超出五万个词向量的词用0代替
train_pad[ train_pad>=num_words ] = 0


# 可见padding之后前面的tokens全变成0，文本在最后面
print("train_pad",train_pad[33])



# 准备target向量，前2000样本为1，后2000为0
train_target = np.concatenate((np.ones(len(pos_txts)),np.zeros(len(neg_txts))))

# 进行训练和测试样本的分割
from sklearn.model_selection import train_test_split

# 90%的样本用来训练，剩余10%用来测试，random_state可以打乱样本顺序，非常重要！
X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                    train_target,
                                                    test_size=0.1,
                                                    random_state=12)
# 查看训练样本，确认无误
print(reverse_tokens(X_train[35]))
print('class: ',y_train[35])


# 用LSTM对样本进行分类
# 现在我们用keras搭建LSTM模型，模型的第一层是Embedding层，
# 只有当我们把tokens索引转换为词向量矩阵之后，才可以用神经网络对文本进行处理。
#  keras提供了Embedding接口，避免了繁琐的稀疏矩阵操作。
# 在Embedding层我们输入的矩阵为：(batchsize, maxtokens)
#  输出矩阵为： $$(batchsize, maxtokens, embeddingdim)$$

model = Sequential()

# 模型第一层为embedding，trainable=false 不训练embedding层
model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_tokens,
                    trainable=False))
# 加lstm层，有32个训练单元
model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
model.add(LSTM(units=16, return_sequences=False))

# 注：构建模型
# 我在这个教程中尝试了几种神经网络结构，因为训练样本比较少，所以我们可以尽情尝试，训练过程等待时间并不长：
# （1）GRU：如果使用GRU的话，测试样本可以达到87%的准确率，但我测试自己的文本内容时发现，GRU最后一层激活函数的输出都在0.5左右，说明模型的判断不是很明确，信心比较低，而且经过测试发现模型对于否定句的判断有时会失误，我们期望对于负面样本输出接近0，正面样本接近1而不是都徘徊于0.5之间。
# （2）BiLSTM：测试了LSTM和BiLSTM，发现BiLSTM的表现最好，LSTM的表现略好于GRU，这可能是因为BiLSTM对于比较长的句子结构有更好的记忆，有兴趣的朋友可以深入研究一下。
# Embedding之后第，一层我们用BiLSTM返回sequences，然后第二层16个单元的LSTM不返回sequences，只返回最终结果，最后是一个全链接层，用sigmoid激活函数输出结果

# GRU的代码
# model.add(GRU(units=32, return_sequences=True))
# model.add(GRU(units=16, return_sequences=True))
# model.add(GRU(units=4, return_sequences=False))

model.add(Dense(1, activation='sigmoid'))
# 我们使用adam作为优化器 以0.001的learning rate进行优化
optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# 我们来看一下模型的结构，一共90k左右可训练的变量
model.summary()

path_checkpoint = 'output/sentiment_checkpoint.keras'
# 回调函数1 建立一个权重的存储点，save_weights_only标识只保存权重，save_best_only表示 当var_loss 有改善的时候才会储存权重，否则不保存
checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                      verbose=1, save_weights_only=True,
                                      save_best_only=True)


# 尝试加载已训练模型
try:
    model.load_weights(path_checkpoint)
except Exception as e:
    print(e)

# 回调函数2: 定义early stoping 回调函数，如果3个epoch内validation loss没有改善则停止训练
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# 回调函数3: 自动降低learning rate
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1, min_lr=1e-5, patience=0,
                                       verbose=1)

# 定义callback函数
callbacks = [
    earlystopping,
    checkpoint,
    lr_reduction
]

# 开始训练
model.fit(X_train, y_train,
           validation_split=0.1,
           epochs=20,
           batch_size=128,
           callbacks=callbacks)

# 查看训练结果，计算准确率
result = model.evaluate(X_test, y_test)
print('Accuracy:{0:.2%}'.format(result[1]))

model.save('output/nlp_model.h5')
json_string = model.to_json()
print("json_string",json_string)
# 定义一个函数，来预测我们自己的文本
def predict_sentiment(text):
    print(text)
    # 去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [ i for i in cut ]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            cut_list[i] = 0
    # padding
    tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
                           padding='pre', truncating='pre')
    # 预测
    # model2=model_from_json(json_string)
    # model2.load_weights(path_checkpoint)
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('是一例正面评价','output=%.2f'%coef)
    else:
        print('是一例负面评价','output=%.2f'%coef)

test_list = [
    '酒店设施不是新的，服务态度很不好',
    '纯粹一个大傻逼，在医院估计也是狗杂种问你病情你问我想要啥药我给你开狗娘省的要你妈',
    '酒店卫生条件非常不好',
    '耐心细心，医生回复及时',
    '很有水平，感谢医生，回复很快，',
    '谢谢，有问必答，一针见血，回答的都是需要的！'
]
# 预测结果
for text in test_list:
    predict_sentiment(text)


# 错误分类的文本 经过查看，发现错误分类的文本的含义大多比较含糊，
# 就算人类也不容易判断极性，如index为101的这个句子，好像没有一点满意的成分，
# 但这例子评价在训练样本中被标记成为了正面评价，
# 而我们的模型做出的负面评价的预测似乎是合理的。
y_pred = model.predict(X_test)
y_pred = y_pred.T[0]
y_pred = [1 if p>= 0.5 else 0 for p in y_pred]
y_pred = np.array(y_pred)

y_actual = np.array(y_test)

# 找出错误分类的索引
misclassified = np.where( y_pred != y_actual )[0]
print("错误分类的结果：",misclassified)


# 输出所有错误分类的索引
len(misclassified)
print(len(X_test))


# 我们来找出错误分类的样本看看
idx=101
print(reverse_tokens(X_test[idx]))
print('预测的分类', y_pred[idx])
print('实际的分类', y_actual[idx])



idx=1
print(reverse_tokens(X_test[idx]))
print('预测的分类', y_pred[idx])
print('实际的分类', y_actual[idx])



