# README.md

## 说明

爬取微博用户的微博内容，通过情感分析工具检测用户的情感变化。通过画折线图的方式展现出来，并且筛选出情感表现最好和最糟糕的排行内容


## API部分

```python
url = 'https://m.weibo.cn/api/container/getIndex'

# 请求信息，前四项暂不需要
data = {
        # 'uid': '3741092507',
        # 'luicode': '10000011',
        # 'lfid': '100103type=3&q=森林木西',
        # 'featurecode': '20000320',
        'type': 'uid',
        'value': '3741092507',
        'containerid': '1076033741092507',
        'page': num,    
}
```

#### 参数说明

 - type: 查询方法
 - value: 查询的值
 - containerid: 查询的内容
    - 100505 + uid: 信息
    - 107603 + uid: 微博
 - page: 页数

```python
# 这里用weibo_json表示获取到的json，item表示其中的每一项

item['mblog']                             有关微博内容的信息
item['mblog']['created_at']               微博创建时间
item['mblog']['attitudes_count']          微博点赞数
item['mblog']['comments_count']           微博评论数
item['mblog']['reposts_count']            微博转发数
item['mblog']['text']                     微博文本内容（含CSS样式）


# 转发微博信息，需要通过key值的存在来判断==>if 'retweeted_status' in item['mblog']

item['mblog']['retweeted_status']['text']                   转发微博文本内容
item['mblog']['retweeted_status']['created_at']             转发微博创建时间
item['mblog']['retweeted_status']['attitudes_count']        转发微博点赞数     
item['mblog']['retweeted_status']['comments_count']         转发微博评论数     
item['mblog']['retweeted_status']['reposts_count']          转发微博转发数      
```

## 第三方模块

 - requests     # 爬虫
 - numpy        # 处理数据
 - matplotlib   # 画图
 - snownlp      # 情感分析

## 数据保存

```python
# 这里采取直接保存numpy数据的方式

# 个人喜好使用方法一，此处方法三不能进行保存应该是编码问题暂未解决
# 方法二在读取数据时，需要指定dtype
1. np.save("data_arr.npy", data_arr) 
2. np.tofile("data_arr.bin", data_arr)
3. np.savetxt("data_arr.txt", data_arr, delimiter=",", fmt="%d")

# 读取数据方式
1. np.load("data_arr.npy")
2. np.fromfile("data_arr.bin", dtype = "U162")
3. np.loadtxt("data_arr.txt", delimiter = ",", fmt = "%d")

```

## 关于snownlp的函数（选读）

```python

text = u"会哭的QAQ"
# text = 'happy, sad'
s = SnowNLP(text)

s.words             # 分词
s.tags              # 暂没理解
s.sentiments        # 情感趋向
s.pinyin            # 转为拼音
s.keywords(2)       # 提取两个关键字
s.summary(3)        # 提取摘要
s.tf                # 计算词频
s.idf               # 计算逆文档频率，不懂QAQ
s.sim(u'生病')      # 计算文本相似度

```