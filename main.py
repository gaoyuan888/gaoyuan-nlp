# coding=gbk
# %matplotlib inline
import re
import jieba # 结巴分词
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")
# flask 和gevent共同提供服务
from gevent.pywsgi import WSGIServer
from flask import Flask
app = Flask(__name__)
# 我们使用tensorflow的keras接口来建模
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
cn_model = KeyedVectors.load_word2vec_format('conf/sgns.zhihu.bigram', binary=False)

# 4.找出相近的词语，余弦相似度
similar_1= cn_model.most_similar(positive=['大学'],topn=10)
print("similar_1",similar_1)

# 3.寻找不是同一类的词语
test_words="老师 会计师 程序员 律师 医生 老人"
test_words_res=cn_model.doesnt_match(test_words.split())
print('在'+test_words+"中，不是同一类的词为：%s"%test_words_res)

#加载模型

# model2 = keras.models.load_model('my_model.h5')
model2 = load_model('conf/nlp_model.h5')
model2.summary()
def predict_sentiment(text):
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
    tokens_pad = pad_sequences([cut_list], maxlen=236,
                               padding='pre', truncating='pre')
    # 预测
    result = model2.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('是一例正面评价','output=%.2f'%coef)
    else:
        print('是一例负面评价','output=%.2f'%coef)
    return coef

@app.route("/emotion/<s>")
def urlParams(s):
    res= predict_sentiment(s);
    return  str(res)

http_server = WSGIServer(('',5000),app)
http_server.serve_forever()

