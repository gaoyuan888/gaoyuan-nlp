# coding=gbk
# %matplotlib inline
import re
import jieba # ��ͷִ�
# gensim��������Ԥѵ��word vector
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")
# flask ��gevent��ͬ�ṩ����
from gevent.pywsgi import WSGIServer
from flask import Flask
app = Flask(__name__)
# ����ʹ��tensorflow��keras�ӿ�����ģ
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
cn_model = KeyedVectors.load_word2vec_format('conf/sgns.zhihu.bigram', binary=False)

# 4.�ҳ�����Ĵ���������ƶ�
similar_1= cn_model.most_similar(positive=['��ѧ'],topn=10)
print("similar_1",similar_1)

# 3.Ѱ�Ҳ���ͬһ��Ĵ���
test_words="��ʦ ���ʦ ����Ա ��ʦ ҽ�� ����"
test_words_res=cn_model.doesnt_match(test_words.split())
print('��'+test_words+"�У�����ͬһ��Ĵ�Ϊ��%s"%test_words_res)

#����ģ��

# model2 = keras.models.load_model('my_model.h5')
model2 = load_model('conf/nlp_model.h5')
model2.summary()
def predict_sentiment(text):
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
    tokens_pad = pad_sequences([cut_list], maxlen=236,
                               padding='pre', truncating='pre')
    # Ԥ��
    result = model2.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('��һ����������','output=%.2f'%coef)
    else:
        print('��һ����������','output=%.2f'%coef)
    return coef

@app.route("/emotion/<s>")
def urlParams(s):
    res= predict_sentiment(s);
    return  str(res)

http_server = WSGIServer(('',5000),app)
http_server.serve_forever()

