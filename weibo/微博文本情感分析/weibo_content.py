# coding: utf-8

import requests
import re
import io
import sys
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')
import numpy as np
"""
获取微博博主所有的微博文字内容
"""

url = 'https://m.weibo.cn/api/container/getIndex'

data_arr = np.array([['date'], ['text']])

for num in range(1, 21):
    data = {
    #     'uid': '3741092507',
    #     'luicode': '10000011',
    #     'lfid': '100103type=3&q=森林木西',
    #     'featurecode': '20000320',
        'type': 'uid',
        'value': '3741092507',
        'containerid': '1076033741092507',
        'page': num,
    }

    s = requests.get(url, params=data)
    userinfo_json = s.json()


    weibo_json = userinfo_json['cards']
    # 筛选出 博主原创内容 + ' '+ 转发微博的内容
    text_list = [item['mblog']['text'] + ' ' + (item['mblog']['retweeted_status']['text'] if 'retweeted_status' in item['mblog'] else ' ') for item in weibo_json]

    # 微博创建时间
    date_list = [item['mblog']['created_at'] for item in weibo_json] 

    # 利用sub函数去除不必要的字符
    # 去除CSS样式和其他人的艾特文字
    text_list = [re.sub(r'//.+', '', re.sub(r'<(.+?)>', '', text)) for text in text_list]
    # 利用正则去除一些影响内容    
    text_list = [re.sub(r'网页链接? ', '', re.sub(r'@.+? ', '', text)) for text in text_list]
    text_list = [re.sub(r'#.+?#', '', re.sub(r'抱歉.+?。', '', text)) for text in text_list]
    text_list = [re.sub(r'转发微博? ', '', re.sub(r'转发? ', '', text)) for text in text_list]
    text_list = [re.sub(r' 查看帮助：http：', '', re.sub(r'查看图片? ', '', text)) for text in text_list]
    text_list = [re.sub(r' 查看帮助：', '', text) for text in text_list] 

    # 保存微博时间和微博内容为numpy数组
    new_arr = np.array([date_list, text_list])
    # 进行合并
    data_arr = np.hstack((data_arr, new_arr))

# 获取文本内容的列表，并删除空白微博的列
arr_list = data_arr[1].tolist()
data_arr = np.delete(data_arr, [index for index, item in enumerate(arr_list) if '' == item], axis=1)

print(data_arr)
