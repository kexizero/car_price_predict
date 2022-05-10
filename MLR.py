import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
import pandas as pd

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) /std
    return X

Train_data = pd.read_csv('./data/used_car_train_20200313.csv', sep=' ')
TestB_data = pd.read_csv('./data/used_car_testB_20200421.csv', sep=' ')

# print('Train data shape:', Train_data.shape)
# print('TestA data shape:', TestB_data.shape)

# Train_data.info()

numerical_cols = Train_data.select_dtypes(exclude='object').columns
categorical_cols = Train_data.select_dtypes(include='object').columns
# print(categorical_cols)
# print(numerical_cols)

feature_cols = [col for col in numerical_cols if col not in ['SaleID', 'name', 'regDate', 'creatDate', 'price', 'model', 'brand', 'regionCode', 'seller']]
feature_cols = [col for col in feature_cols if 'Type' not in col]
# print(feature_cols)

X_data = Train_data[feature_cols]
X_data = np.array(X_data)

x_data = normalize(X_data)
Y_data = Train_data['price']
Y_data = np.array(Y_data)


X_test = TestB_data[feature_cols]

# print('X train shape:', X_data.shape)
# print('X test shape:', X_test.shape)

m = 150000
n = 18
X = tf.placeholder(tf.float32, name='X', shape=[m, n])
Y = tf.placeholder(tf.float32, name='Y')

bias = tf.Variable(0.0)
w = tf.Variable(tf.random_normal([n, 1]))
Y_hat = tf.matmul(X, w) + bias

loss = tf.reduce_mean(Y - Y_hat, name='loss') + 0.6 * tf.nn.l2_loss(w)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

init = tf.global_variables_initializer()
total = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        l = sess.run([optimizer, loss], feed_dict={X:X_data, Y:Y_data})
        total.append(l)
        print('Epoch{0}:Loss{1}'.format(i, l))

        b_value, w_value = sess.run(bias, w)
plt.plot(total)
plt.show()

