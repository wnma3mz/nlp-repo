# from textblob import TextBlob
from snownlp import SnowNLP 
import matplotlib.pyplot as plt
import numpy as np

def draw_pic(data_list, date_list, title=u"微博情感分析图"):
    plt.figure(figsize=(50, 6), dpi=80)
    x = range(len(date_list))
    plt.plot(data_list, 'ro-', color='b')
    plt.xticks(x, date_list, rotation=45)
    plt.title(title)
    plt.savefig('%s.png' % title)
    plt.show()

def rank_data(data_list, content_list, date_list, num=10):
    
    import heapq

    # 查询数值的索引
    index_func = lambda num, x_list: x_list.index(num)

    # 找出最大/最小的num个数据
    max_list = heapq.nlargest(num, data_list)
    min_list = heapq.nsmallest(num, data_list)

    # 索引数值的列表
    max_index_list = [index_func(num, data_list) for num in max_list]
    min_index_list = [index_func(num, data_list) for num in min_list]

    # 输出时间 + 内容
    content_max_list = [print(date_list[index], ' ', content_list[index], '\n') for index in max_index_list]
    content_min_list = [print(date_list[index], ' ', content_list[index], '\n') for index in min_index_list]  

if __name__ == '__main__':
    
    data_list = []
    # 读取数据数组
    data_arr = np.load("data_arr.npy")

    # 获取时间和内容列表
    date_list = data_arr[0].tolist()[1:-1][::-1]
    content_list = data_arr[1].tolist()[1:-1][::-1]

    # 进行分析，黑箱操作
    for content in content_list:
        s = SnowNLP(content)
        data_list.append(s.sentiments)

    # 画图
    # draw_pic(data_list, date_list)
    
    # 排行
    rank_data(data_list, content_list, date_list)


