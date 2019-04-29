# coding=gbk

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import re
import jieba # ��ͷִ�
# gensim��������Ԥѵ��word vector
from gensim.models import KeyedVectors
import warnings
from keras.models import model_from_json
warnings.filterwarnings("ignore")

# ����ʹ��tensorflow��keras�ӿ�����ģ
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# һ��������ģ��
cn_model = KeyedVectors.load_word2vec_format('../conf/sgns.zhihu.bigram', binary=False)

# 1.������ά�ȣ�һ��������300��ά��������ʾ��embedding_dim=300
embedding_dim= cn_model['ɽ����ѧ'].shape[0]
print('����������{}',format(embedding_dim))
array=cn_model['ɽ����ѧ']
#��ӡ������
print("array",array)

# 2.���ƶȣ��������ƶ�
similar=cn_model.similarity('��ð','����')
print("similar",similar)
# �������ƶȼ��㷽��    ���������/����������/������  dot(["����"]/|["����"],["����"]/|["����"])
similar_=np.dot(cn_model["����"]/np.linalg.norm(cn_model["����"]),cn_model["����"]/np.linalg.norm(cn_model["����"]))
print("similar_",similar_)

# 3.Ѱ�Ҳ���ͬһ��Ĵ���
test_words="��ʦ ���ʦ ����Ա ��ʦ ҽ�� ����"
test_words_res=cn_model.doesnt_match(test_words.split())
print('��'+test_words+"�У�����ͬһ��Ĵ�Ϊ��%s"%test_words_res)

# 4.�ҳ�����Ĵ���������ƶ�
similar_1= cn_model.most_similar(positive=['��ѧ'],topn=10)
print("similar_1",similar_1)

# 5.��һ�������ڴ������е�����
index1=cn_model.vocab["ɽ����ѧ"].index
print("index1",index1);

# 6.���������������Ĵ���  215 ��Ӧ�Ĵ�Ϊ��ʦ
word=cn_model.index2word[215];
print("word",word);

# ��������ѵ��
import os

path_prefix="data"
# 1.��ȡ������������������������ļ�������
# ÿ���ļ���һ������
pos_txts=os.listdir(path_prefix+"/pos")
neg_txts=os.listdir(path_prefix+"/neg")
print("�����ܹ�"+str(len(pos_txts)+len(neg_txts))+"��");

# 2.�������ǽ����е��������ݷ��õ�һ��list��
# �洢�������ۣ�ÿ������Ϊһ��string
train_texts_orig = []

# �������������֮��train_texts_origΪһ������4000���ı���list
# ����ǰ2000���ı�Ϊ�������ۣ���2000��Ϊ��������
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


# 3.���зִʺ�tokenize(������)
# train_tokens��һ��������list�����к���4000��Сlist����Ӧÿһ������
train_tokens = []
for text in train_texts_orig:
    # ȥ�����
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+��������������~@#��%����&*����]+", "",text)
    # ��ͷִ�
    cut = jieba.cut(text)
    # ��ͷִʵ�������Ϊһ��������
    # ��������ת��Ϊlist
    cut_list = [ i for i in cut ]
    for i, word in enumerate(cut_list):
        try:
            # ����ת��Ϊ�������е�����index
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            # ����ʲ����ֵ��У������0
            cut_list[i] = 0
    train_tokens.append(cut_list)

# 4.�������tokens�ĳ���
num_tokens = [len(tokens) for tokens in train_tokens]
num_tokens = np.array(num_tokens)

# 5.ƽ��tokens�ĳ���
avglen=np.mean(num_tokens)
print("���дʵ�ƽ������",avglen)

# 6.�������tokens�ĳ���
maxlen=np.max(num_tokens)
print("����ı�����",maxlen)


# 7.�۲�һ�������ֲ����
plt.hist(num_tokens,bins = 100)
plt.xlim((0,400))
plt.ylabel('number of tokens')
plt.xlabel('length of tokens')
plt.title('Distribution of tokens length')
plt.show()

# 8.������̬�ֲ�
plt.hist(np.log(num_tokens), bins = 100)
plt.xlim((0,10))
plt.ylabel('number of tokens')
plt.xlabel('length of tokens')
plt.title('Distribution of tokens length')
plt.show()

# 9.ȡtokensƽ��ֵ����������tokens�ı�׼�
# ����tokens���ȵķֲ�Ϊ��̬�ֲ�����max_tokens���ֵ���Ժ���95%���ҵ�����
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
print("���ǰٷ�֮��ʮ����ı�����",max_tokens)

# 10.ȡtokens�ĳ���Ϊ236ʱ����Լ95%������������
# ���ǶԳ��Ȳ���Ľ���padding�������Ľ����޼�
pricent= np.sum( num_tokens < max_tokens ) / len(num_tokens)
print("ȡֵ�ı����ȸ����������Ȱٷֱ�",pricent)

# 11.������tokensת��Ϊ�ı�
def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index2word[i]
        else:
            text = text + ' '
    return text

# 12.����tokenize�ٻָ����ı�
# �ɼ������Ŷ�û����
reverse = reverse_tokens(train_tokens[0])
print("����ת��Ϊ���Ӻ��һ�����ӣ�",reverse)
# ԭʼ�ı�
print("ԭʼ�ı���",train_texts_orig[0])


# 13.׼��Embedding Matrix
# ����������Ϊģ��׼��embedding matrix�����������󣩣�
# ����keras��Ҫ��������Ҫ׼��һ��ά��Ϊ(numwords, embeddingdim)�ľ���
# num words��������ʹ�õĴʻ��������emdedding dimension����������ʹ�õ�Ԥѵ��������ģ������300��
# ÿһ���ʻ㶼��һ������Ϊ300��������ʾ��
# ע������ֻѡ��ʹ��ǰ50k��ʹ��Ƶ����ߵĴʣ�
# �����Ԥѵ��������ģ���У�һ����260��ʻ��������ȫ��ʹ���ڷ��������ϻ���˷Ѽ�����Դ��
# ��Ϊ���ǵ�ѵ��������С��һ��ֻ��4k�����������100k��200k���������ѵ������ʱ���ڷ��������Ͽ��Կ��Ǽ���ʹ�õĴʻ�����

print("embedding_dim:",embedding_dim)

# 14.��ʼ��embedding_matrix
# ֻʹ��ǰ20000����
num_words = 100000
# ��ʼ��embedding_matrix��֮����keras�Ͻ���Ӧ��
embedding_matrix = np.zeros((num_words, embedding_dim))
# embedding_matrixΪһ�� [num_words��embedding_dim] �ľ���
# ά��Ϊ 50000 * 300
for i in range(num_words):
    # ��Ԥ���صĴ�����ģ���У�ѡȡǰ5w�������Ƶ�ʣ������
    embedding_matrix[i,:] = cn_model[cn_model.index2word[i]]
embedding_matrix = embedding_matrix.astype('float32')


# 15.���index�Ƿ��Ӧ��
# ���300����Ϊ����Ϊ300��embedding����һһ��Ӧ
npsum = np.sum(cn_model[cn_model.index2word[333]] == embedding_matrix[333])
print("npsum",npsum)

# 16.ȷ��һ�´�����embedding_matrix��ά�ȣ�
# ���ά��Ϊkeras��Ҫ�󣬺�������ģ�����õ�
emb_dim=embedding_matrix.shape
print("��������ά�ȣ�",emb_dim)


# 17.����padding��truncating�� �����train_tokens��һ��list
# ���ص�train_pad��һ��numpy array

# padding����䣩��truncating���޼���
# ���ǰ��ı�ת��Ϊtokens��������֮��ÿһ�������ĳ��Ȳ�����ȣ�
# ����Ϊ�˷���ģ�͵�ѵ��������Ҫ�������ĳ��ȱ�׼����
# ��������ѡ����236������Ժ���95%ѵ�������ĳ��ȣ����������ǽ���padding��truncating��
# ����һ�����'pre'�ķ�����������ı�������ǰ�����0����Ϊ����һЩ�о������е�ʵ����
# ������ı������������0�Ļ������ģ�����һЩ����Ӱ�졣
train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                            padding='pre', truncating='pre')


# ����������������Ĵ���0����
train_pad[ train_pad>=num_words ] = 0


# �ɼ�padding֮��ǰ���tokensȫ���0���ı��������
print("train_pad",train_pad[33])



# ׼��target������ǰ2000����Ϊ1����2000Ϊ0
train_target = np.concatenate((np.ones(len(pos_txts)),np.zeros(len(neg_txts))))

# ����ѵ���Ͳ��������ķָ�
from sklearn.model_selection import train_test_split

# 90%����������ѵ����ʣ��10%�������ԣ�random_state���Դ�������˳�򣬷ǳ���Ҫ��
X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                    train_target,
                                                    test_size=0.1,
                                                    random_state=12)
# �鿴ѵ��������ȷ������
print(reverse_tokens(X_train[35]))
print('class: ',y_train[35])


# ��LSTM���������з���
# ����������keras�LSTMģ�ͣ�ģ�͵ĵ�һ����Embedding�㣬
# ֻ�е����ǰ�tokens����ת��Ϊ����������֮�󣬲ſ�������������ı����д���
#  keras�ṩ��Embedding�ӿڣ������˷�����ϡ����������
# ��Embedding����������ľ���Ϊ��(batchsize, maxtokens)
#  �������Ϊ�� $$(batchsize, maxtokens, embeddingdim)$$

model = Sequential()

# ģ�͵�һ��Ϊembedding��trainable=false ��ѵ��embedding��
model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_tokens,
                    trainable=False))
# ��lstm�㣬��32��ѵ����Ԫ
model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
model.add(LSTM(units=16, return_sequences=False))

# ע������ģ��
# ��������̳��г����˼���������ṹ����Ϊѵ�������Ƚ��٣��������ǿ��Ծ��鳢�ԣ�ѵ�����̵ȴ�ʱ�䲢������
# ��1��GRU�����ʹ��GRU�Ļ��������������Դﵽ87%��׼ȷ�ʣ����Ҳ����Լ����ı�����ʱ���֣�GRU���һ�㼤������������0.5���ң�˵��ģ�͵��жϲ��Ǻ���ȷ�����ıȽϵͣ����Ҿ������Է���ģ�Ͷ��ڷ񶨾���ж���ʱ��ʧ�������������ڸ�����������ӽ�0�����������ӽ�1�����Ƕ��ǻ���0.5֮�䡣
# ��2��BiLSTM��������LSTM��BiLSTM������BiLSTM�ı�����ã�LSTM�ı����Ժ���GRU�����������ΪBiLSTM���ڱȽϳ��ľ��ӽṹ�и��õļ��䣬����Ȥ�����ѿ��������о�һ�¡�
# Embedding֮��ڣ�һ��������BiLSTM����sequences��Ȼ��ڶ���16����Ԫ��LSTM������sequences��ֻ�������ս���������һ��ȫ���Ӳ㣬��sigmoid�����������

# GRU�Ĵ���
# model.add(GRU(units=32, return_sequences=True))
# model.add(GRU(units=16, return_sequences=True))
# model.add(GRU(units=4, return_sequences=False))

model.add(Dense(1, activation='sigmoid'))
# ����ʹ��adam��Ϊ�Ż��� ��0.001��learning rate�����Ż�
optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# ��������һ��ģ�͵Ľṹ��һ��90k���ҿ�ѵ���ı���
model.summary()

path_checkpoint = 'output/sentiment_checkpoint.keras'
# �ص�����1 ����һ��Ȩ�صĴ洢�㣬save_weights_only��ʶֻ����Ȩ�أ�save_best_only��ʾ ��var_loss �и��Ƶ�ʱ��Żᴢ��Ȩ�أ����򲻱���
checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                      verbose=1, save_weights_only=True,
                                      save_best_only=True)


# ���Լ�����ѵ��ģ��
try:
    model.load_weights(path_checkpoint)
except Exception as e:
    print(e)

# �ص�����2: ����early stoping �ص����������3��epoch��validation lossû�и�����ֹͣѵ��
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# �ص�����3: �Զ�����learning rate
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1, min_lr=1e-5, patience=0,
                                       verbose=1)

# ����callback����
callbacks = [
    earlystopping,
    checkpoint,
    lr_reduction
]

# ��ʼѵ��
model.fit(X_train, y_train,
           validation_split=0.1,
           epochs=20,
           batch_size=128,
           callbacks=callbacks)

# �鿴ѵ�����������׼ȷ��
result = model.evaluate(X_test, y_test)
print('Accuracy:{0:.2%}'.format(result[1]))

model.save('output/nlp_model.h5')
json_string = model.to_json()
print("json_string",json_string)
# ����һ����������Ԥ�������Լ����ı�
def predict_sentiment(text):
    print(text)
    # ȥ���
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+��������������~@#��%����&*����]+", "",text)
    # �ִ�
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
    # Ԥ��
    # model2=model_from_json(json_string)
    # model2.load_weights(path_checkpoint)
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('��һ����������','output=%.2f'%coef)
    else:
        print('��һ����������','output=%.2f'%coef)

test_list = [
    '�Ƶ���ʩ�����µģ�����̬�Ⱥܲ���',
    '����һ����ɵ�ƣ���ҽԺ����Ҳ�ǹ��������㲡����������Ҫɶҩ�Ҹ��㿪����ʡ��Ҫ����',
    '�Ƶ����������ǳ�����',
    '����ϸ�ģ�ҽ���ظ���ʱ',
    '����ˮƽ����лҽ�����ظ��ܿ죬',
    'лл�����ʱش�һ���Ѫ���ش�Ķ�����Ҫ�ģ�'
]
# Ԥ����
for text in test_list:
    predict_sentiment(text)


# ���������ı� �����鿴�����ִ��������ı��ĺ�����ȽϺ�����
# ��������Ҳ�������жϼ��ԣ���indexΪ101��������ӣ�����û��һ������ĳɷ֣�
# ��������������ѵ�������б���ǳ�Ϊ���������ۣ�
# �����ǵ�ģ�������ĸ������۵�Ԥ���ƺ��Ǻ���ġ�
y_pred = model.predict(X_test)
y_pred = y_pred.T[0]
y_pred = [1 if p>= 0.5 else 0 for p in y_pred]
y_pred = np.array(y_pred)

y_actual = np.array(y_test)

# �ҳ�������������
misclassified = np.where( y_pred != y_actual )[0]
print("�������Ľ����",misclassified)


# ������д�����������
len(misclassified)
print(len(X_test))


# �������ҳ�����������������
idx=101
print(reverse_tokens(X_test[idx]))
print('Ԥ��ķ���', y_pred[idx])
print('ʵ�ʵķ���', y_actual[idx])



idx=1
print(reverse_tokens(X_test[idx]))
print('Ԥ��ķ���', y_pred[idx])
print('ʵ�ʵķ���', y_actual[idx])



