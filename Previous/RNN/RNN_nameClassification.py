'''
在处理自然语言的时候，通常会把字/词用one-hot向量表示，但由于使用one-hot表示的向量，有维度高。稀疏的缺点
使用一个嵌入层 Embed，目的是降维，将向量变成一个低维、稠密的向量
在处理序列时，序列长度不一
'''
#一般存在序列问题，可以考虑RNN,如语言模型，故诗词预测
# 图片存在二维序列， 一般考虑CNN

import math
from datetime import time

'''
模型
第一层：嵌入层
第二层：GRU Layer GRU（门控循环单元）是LSTM一种变体，比LSTM结构简单，且效果也很好
第三层：Linear Layer 将GRU的最终输出结果hidden_N经过线性层变换用于分类
'''


#set集合去除重复元素，sorted按首字母ASCII排序

def time_since(since):
    s = time.time()-since
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds'% (m,s)