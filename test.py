import numpy as np
import json
import csv

# print("hello")

def load_data():
    # load dataset
    datafile = './dataset/housing.csv'
    with open(datafile,encoding = 'utf-8') as f:
        data = np.loadtxt(f,delimiter = ",", skiprows = 1)
        # print(data[:2])

    # reshape data
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                    'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)
    data=data.reshape([data.shape[0],feature_num])
    # print("datalen:",len(data))
    # x = data[0]
    # print(x.shape)
    # print(x)


    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    # print("training_data.shape:",training_data.shape)
    (maximums, minimums, avgs) = (training_data.max(axis=0), 
                                training_data.min(axis=0), 
                                training_data.sum(axis=0) / training_data.shape[0])
    # 对数据进行归一化处理
    for i in range(feature_num):
        # print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])   # 使得所有值在0~1之间

    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data





if __name__ == "__main__":
    training_data, test_data=load_data()
    print("training_data.shape, test_data.shape",training_data.shape, test_data.shape)

    