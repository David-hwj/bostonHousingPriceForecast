import numpy as np
import json
import csv
import matplotlib.pyplot as plt

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

class Network(object):
    def __init__(self, num_of_weights):
        # self.w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    # 前向计算
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    # 损失函数:均方差
    def loss(self, z, y):
        error = z - y
        # print("error:" ,error)
        cost = error * error
        # print("cost:" ,cost)
        cost = np.mean(cost)
        return cost

    # 计算梯度
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        return gradient_w, gradient_b

    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
        
    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i+1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses

    def train_SGD(self, training_data, num_epoches, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epoches):
            # 在每轮迭代开始之前，将训练数据的顺序随机的打乱，
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                #print(self.w.shape)
                #print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                                 format(epoch_id, iter_id, loss))
        
        return losses

if __name__ == "__main__":
    # 获取数据
    train_data, test_data = load_data()
    x = train_data[:, :-1]
    y = train_data[:, -1:]
    # 创建网络
    net = Network(13)
    num_iterations=10000
    # 启动训练
    losses = net.train(x,y, iterations=num_iterations, eta=0.01)
    # losses = net.train_SGD(train_data, num_epoches=50, batch_size=100, eta=0.1)

    # 画出损失函数的变化趋势
    plot_x = np.arange(num_iterations)
    # plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()