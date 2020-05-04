import pandas as pd
import numpy as np
# 处理Consumer & Visitor Insights For Neighborhoods数据集
io = "D:/dataMiner/cbg_patterns.csv"
data = pd.read_csv(io)
# 由于popularity_by_hour与popularity_by_day这两列是一个字符串，分别是包含24个数值数据与7个数值数据的数组。
# 因此对这两列进行处理，分别分裂成24列与7列。
data['popularity_by_hour'].replace('[]','[,,,,,,,,,,,,,,,,,,,,,,,]',inplace = True)
data['popularity_by_hour'] = data['popularity_by_hour'].map(lambda x:x[1:len(x) - 1])
for i in range(1,25):
    data['popularity_by_hour' + str(i) ] = data['popularity_by_hour'].map(lambda x:x.split(',')[i - 1])
    data['popularity_by_hour' + str(i) ].replace('',np.NaN,inplace = True)
    data['popularity_by_hour' + str(i) ] = data['popularity_by_hour' + str(i) ].astype('float')
data['popularity_by_day'].replace('{}','{"Monday":,"Tuesday":,"Wednesday":,"Thursday":,"Friday":,"Saturday":,"Sunday":}',inplace = True)
data['popularity_by_day'] = data['popularity_by_day'].map(lambda x:x[1:len(x) - 1])
for i in range(1,8):
    data['popularity_by_day' + str(i) ] = data['popularity_by_day'].map(lambda x:x.split(',')[i - 1])
    data['popularity_by_day' + str(i) ] = data['popularity_by_day' +  str(i)].map(lambda x:x.split(':')[1])
    data['popularity_by_day' + str(i) ].replace('',np.NaN,inplace = True)
    data['popularity_by_day' + str(i) ] = data['popularity_by_day' + str(i) ].astype('float')

# 标称数据列名。
category_col = ['census_block_group']
# 数值数据列名。
number_col = ['raw_visit_count','raw_visitor_count','distance_from_home']
for i in range(1,25):
    number_col.append('popularity_by_hour' + str(i))
for i in range(1,8):
    number_col.append('popularity_by_day' + str(i))
# 统计标称数据的频数。
for item in category_col:
    print (item, '频数：\n', pd.value_counts(data[item].values), '\n')

# 最大值
data_show = pd.DataFrame(data = data[number_col].max(), columns = ['max'])
# 最小值
data_show['min'] = data[number_col].min()
# 均值
data_show['mean'] = data[number_col].mean()
# 中位数
data_show['median'] = data[number_col].median()
# 四分位数
data_show['quartile'] = data[number_col].describe().loc['25%']
# 缺失值个数
data_show['missing'] = data[number_col].isnull().sum()
print(data_show)

import matplotlib.pyplot as plt
# 对于数值属性画盒图
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col:
    ax = fig.add_subplot(12, 3, i)
    data[item].plot(kind = 'box')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 对于数值属性画直方图
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col:
    ax = fig.add_subplot(12, 3, i)
    data[item].plot(kind = 'hist', title = item, ax = ax,bins=100)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.8)

# 方法1:删除表格中的空值
data_filtrated = data.dropna()

# 对于方法一处理之后的数值属性画盒图
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col:
    ax = fig.add_subplot(12, 3, i)
    data_filtrated[item].plot(kind = 'box')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 对于方法一处理之后的数值属性画直方图
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col:
    ax = fig.add_subplot(12, 3, i)
    data_filtrated[item].plot(kind = 'hist', title = item, ax = ax,bins=100)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.8)

# 建立原始数据的拷贝，利用方法2：高频值填充
data_filtrated2 = data.copy()
# 对每一列数据，分别进行处理
for item in number_col:
    # 计算最高频率的值
    most_frequent_value = data_filtrated2[item].value_counts().idxmax()
    # 替换缺失值
    data_filtrated2[item].fillna(value = most_frequent_value, inplace = True)
for item in category_col:
    # 计算最高频率的值
    most_frequent_value = data_filtrated2[item].value_counts().idxmax()
    # 替换缺失值
    data_filtrated2[item].fillna(value = most_frequent_value, inplace = True)

# 对于方法二处理之后的数值属性画盒图
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col:
    ax = fig.add_subplot(12, 3, i)
    data_filtrated2[item].plot(kind = 'box')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 对于方法二处理之后的数值属性画直方图
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col:
    ax = fig.add_subplot(12, 3, i)
    data_filtrated2[item].plot(kind = 'hist', title = item, ax = ax,bins=100)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 1)

# 利用knn方法预测缺失值，利用属性之间的相关性，方法3
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
data_filtrated3 = data.copy()
for item in number_col:
    # 假设column1为需要填充的列
    df_train= data_filtrated2[data[item].notnull()][number_col].values
    df_test= data_filtrated2[data[item].isnull()][number_col].values
    # y为目标值
    y = data_filtrated2[data[item].notnull()][item].values
    # X为特征属性值
    X = df_train
    #rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    #rfr.fit(X, y)
    knn=neighbors.KNeighborsClassifier()
    knn.fit(X,y)
    result= knn.predict(df_test)
    # 用得到的预测结果填补原缺失数据
    data_filtrated3.loc[ (data_filtrated3[item].isnull()), item ] = result

# 对于方法三处理之后的数值属性画盒图
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col:
    ax = fig.add_subplot(12, 3, i)
    data_filtrated3[item].plot(kind = 'box')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 对于方法三处理之后的数值属性画直方图
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col:
    ax = fig.add_subplot(12, 3, i)
    data_filtrated3[item].plot(kind = 'hist', title = item, ax = ax,bins=100)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 1)

# 用方法4处理表格，对于有缺失值的一行，计算这一行与其他行的欧式距离，最后取距离最近的行填充缺失值
# 由于计算欧式距离过程中，会存在其他属性缺失，因此拷贝一份表格副本，用高频数填补缺失值，再计算欧式距离
# 由于电脑性能有限，进行大量的循环操作十分耗时，因此这里仅贴出代码，不做结果运算
data_filtrated4 = data.copy()
for item in  number_col:
    mList = data[data[item].isnull()].index.tolist()
    for j in mList:
        mMin = 2147483647
        index = 0
        for i in range(data.shape[0]):
            if i != j and i % 100000 == 1:
                temp = np.linalg.norm(data_filtrated2[number_col][i:i+1].values - data_filtrated2[number_col][j:j+1].values)
                if temp < mMin:
                    mMin = temp
                    index = i
        data_filtrated4.loc[ j:j+1, item ] = data_filtrated2[item][i:i+1].values

# 处理Wine Reviews数据集，将两个表格连接在一起
io2 = "D:/dataMiner/winemag-data_first150k.csv"
data2 = pd.read_csv(io2)
io2 = "D:/dataMiner/winemag-data-130k-v2.csv"
data2_1 = pd.read_csv(io2)
data2 = data2.append(data2_1)

# 处理dataFrame，使得序号连续
data2 = data2.reset_index(drop = True)

# 数值属性
number_col_2= ['price','points']
# 标称属性
category_col_2 = ['country','province','region_1','region_2','taster_name','taster_twitter_handle','variety','winery']
# 统计标称数据的频数。
for item in category_col_2:
    print (item, '频数：\n', pd.value_counts(data2[item].values), '\n')

# 最大值
data_show_2 = pd.DataFrame(data = data2[number_col_2].max(), columns = ['max'])
# 最小值
data_show_2['min'] = data2[number_col_2].min()
# 均值
data_show_2['mean'] = data2[number_col_2].mean()
# 中位数
data_show_2['median'] = data2[number_col_2].median()
# 四分位数
data_show_2['quartile'] = data2[number_col_2].describe().loc['25%']
# 缺失值个数
#data_show['missing'] = data[number_col].describe().loc['count'].apply(lambda x : 368-x)
data_show_2['missing'] = data2[number_col_2].isnull().sum()
print(data_show_2)

# 对于数值属性画盒图
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col_2:
    ax = fig.add_subplot(3, 2, i)
    data2[item].plot(kind = 'box')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 对于数值属性画直方图
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col_2:
    ax = fig.add_subplot(3, 2, i)
    data2[item].plot(kind = 'hist', title = item, ax = ax,bins=100)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.8)

# 删除缺失值，方法一
data2_filtrated = data2.dropna()

# 对于方法一处理之后的数值属性画盒图，左边是原始数据，右边为处理后数据
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col_2:
    ax = fig.add_subplot(3, 2, i)
    data2[item].plot(kind = 'box')
    i += 1
    ax = fig.add_subplot(3, 2, i)
    data2_filtrated[item].plot(kind = 'box')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 对于方法一处理之后的数值属性画直方图，左边是原始数据，右边为处理后数据
fig = plt.figure(figsize = (20,20)
i = 1
for item in number_col_2:
    ax = fig.add_subplot(3, 2, i)
    data2[item].plot(kind = 'hist', title = item, ax = ax,bins=100)
    i += 1
    ax = fig.add_subplot(3, 2, i)
    data2_filtrated[item].plot(kind = 'hist', title = item, ax = ax,bins=100)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.8)

# 建立原始数据的拷贝，用高频填补缺失值，方法二
data2_filtrated2 = data2.copy()
# 对每一列数据，分别进行处理
for item in number_col_2:
    # 计算最高频率的值
    most_frequent_value = data2_filtrated2[item].value_counts().idxmax()
    # 替换缺失值
    data2_filtrated2[item].fillna(value = most_frequent_value, inplace = True)
for item in category_col_2:
    # 计算最高频率的值
    most_frequent_value = data2_filtrated2[item].value_counts().idxmax()
    # 替换缺失值
    data2_filtrated2[item].fillna(value = most_frequent_value, inplace = True)

# 对于方法二处理之后的数值属性画盒图，左边是原始数据，右边为处理后数据
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col_2:
    ax = fig.add_subplot(3, 2, i)
    data2[item].plot(kind = 'box')
    i += 1
    ax = fig.add_subplot(3, 2, i)
    data2_filtrated2[item].plot(kind = 'box')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 对于方法二处理之后的数值属性画直方图，左边是原始数据，右边为处理后数据
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col_2:
    ax = fig.add_subplot(3, 2, i)
    data2[item].plot(kind = 'hist', title = item, ax = ax,bins=100)
    i += 1
    ax = fig.add_subplot(3, 2, i)
    data2_filtrated2[item].plot(kind = 'hist', title = item, ax = ax,bins=100)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.8)

# 用knn填充缺失值，方法三
data2_filtrated3 = data2.copy()
number_col_3 = ['price']
for item in number_col_3:
    # 假设column1为需要填充的列
    df_train= data2_filtrated2[data2[item].notnull()][number_col_3].values
    df_test= data2_filtrated2[data2[item].isnull()][number_col_3].values
    # y为目标值
    y = data2_filtrated2[data2[item].notnull()][item].values
    # X为特征属性值
    X = df_train
    #rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    #rfr.fit(X, y)
    knn=neighbors.KNeighborsClassifier()
    knn.fit(X,y)
    result= knn.predict(df_test)
    # 用得到的预测结果填补原缺失数据
    data2_filtrated3.loc[ (data2_filtrated3[item].isnull()), item ] = result

# 对于方法三处理之后的数值属性画盒图，左边是原始数据，右边为处理后数据
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col_2:
    ax = fig.add_subplot(3, 2, i)
    data2[item].plot(kind = 'box')
    i += 1
    ax = fig.add_subplot(3, 2, i)
    data2_filtrated3[item].plot(kind = 'box')
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

# 对于方法三处理之后的数值属性画直方图，左边是原始数据，右边为处理后数据
fig = plt.figure(figsize = (20,20))
i = 1
for item in number_col_2:
    ax = fig.add_subplot(3, 2, i)
    data2[item].plot(kind = 'hist', title = item, ax = ax,bins=100)
    i += 1
    ax = fig.add_subplot(3, 2, i)
    data2_filtrated3[item].plot(kind = 'hist', title = item, ax = ax,bins=100)
    i += 1
plt.subplots_adjust(wspace = 0.3, hspace = 0.8)


# 用方法4处理表格，对于有缺失值的一行，计算这一行与其他行的欧式距离，最后取距离最近的行填充缺失值
# 由于计算欧式距离过程中，会存在其他属性缺失，因此拷贝一份表格副本，用高频数填补缺失值，再计算欧式距离
# 由于电脑性能有限，进行大量的循环操作十分耗时，因此这里仅贴出代码，不做结果运算
data2_filtrated4 = data2.copy()
for item in  number_col_2:
    mList = data2[data2[item].isnull()].index.tolist()
    for j in mList:
        mMin = 2147483647
        index = 0
        for i in range(data2.shape[0]):
            if i != j and i % 100000 == 1:
                temp = np.linalg.norm(data2_filtrated2[number_col_2][i:i+1].values - data2_filtrated2[number_col_2][j:j+1].values)
                if temp < mMin:
                    mMin = temp
                    index = i
        data2_filtrated4.loc[ j:j+1, item ] = data2_filtrated2[item][i:i+1].values