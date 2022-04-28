import json

import pandas as pd

print(pd.__version__)

# pandas series
# 类似表格中的一列，即一维数组，可以保存任何数据类型
a = [1,2,3]  # 使用数组创建
var = pd.Series(a)
print(var)  # 输出索引、数据、数据类型

var = pd.Series(a, index=["x","y","z"])  # 指定数据和索引
print(var)
print(var["x"])

# 使用字典创建
a = {1:"google", 2:"facebook", 3:"amazon"}
var = pd.Series(a)
print(var)


print("----------------------------------------")
# pandas dataframe
# DataFrame 是一个表格型的数据结构
# 它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。
# DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）。
# pandas.DataFrame( data, index, columns, dtype, copy)
# data为数据，index为索引值（行标签），columns为列标签，dtype为数据类型，copy为拷贝数据，默认False
# 使用数组创建
a = [["google", 10], ["facebook", 11], ["amazon", 12]]
var = pd.DataFrame(a, columns=["name", "num"], dtype=float)
print(var)

# 使用字典创建
a = {'name':['google', 'facebook', 'amazon'], 'num':[10,11,12]}
var = pd.DataFrame(a)
print(var)

print(var.loc[0])  # 返回指定行的数据，返回结果其实就是一个 Pandas Series 数据
#name    google
#num         10
#Name: 0, dtype: object

print(var.loc[[0,1]])  # 返回两行数据



print('-----------------------------------------------')
# pandas CSV
# 读csv文件
df = pd.read_csv('data/2.2.csv')  # df就是一个DataFrame
print(df)
# 保存为csv
df.to_csv("data/2.2.csv")
# 打印前几行（默认5）
print(df.head())
# 打印后几行（默认5）
print(df.tail())
# 打印表格基本信息
print(df.info())





print('--------------------------------------------------')
# pandas JSON
df = pd.read_json("data/2.2.json")
print(df)
# 使用DataFrame构造函数也可以读取JSON字符串转化为DataFrame
URL = 'https://static.runoob.com/download/sites.json'  # 从网路读取JSON
df = pd.read_json(URL)
print(df)

# 嵌套JSON有时候不能直接解析
df = pd.read_json("data/2.2-1.json")
print(df)
# 需要解析
with open("data/2.2-1.json", "r") as f:
    data = json.loads(f.read())
df = pd.json_normalize(data, record_path="students", meta = ['school_name', 'class'])
print(df)











