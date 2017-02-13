import os
import errno
import tensorflow as tf
import numpy as np
import pickle
import signal
import matplotlib.pyplot as plt

class ANN_5FCL:
    def initialize(self, INPUT_DIMS, NR_CLASSES, learningRate = 0.0003, LEVELS = [200,100,60,30]):
        self.INPUT_DIMS = INPUT_DIMS
        self.NR_CLASSES = NR_CLASSES
        self.X = tf.placeholder(tf.float32, [None, INPUT_DIMS])  # INPUT
        self.Y_ = tf.placeholder(tf.float32, [None, NR_CLASSES])  # EXPECTED PROB: one-hot

        # five layers and their number of neurons (tha last layer has NR_CLASSES softmax neurons)
        L = LEVELS[0]
        M = LEVELS[1]
        N = LEVELS[2]
        O = LEVELS[3]

        self.W1 = tf.Variable(tf.truncated_normal([INPUT_DIMS, L], stddev=0.1))
        B1 = tf.Variable(tf.zeros([L]))
        W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
        B2 = tf.Variable(tf.zeros([M]))
        W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
        B3 = tf.Variable(tf.zeros([N]))
        W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
        B4 = tf.Variable(tf.zeros([O]))
        W5 = tf.Variable(tf.truncated_normal([O, NR_CLASSES], stddev=0.1))
        B5 = tf.Variable(tf.zeros([NR_CLASSES]))

        Y1 = tf.nn.sigmoid(tf.matmul(self.X, self.W1) + B1)
        Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
        Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
        Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
        Ylogits = tf.matmul(Y4, W5) + B5  # we save this for the X-entropy

        self.Y = tf.nn.softmax(Ylogits)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=self.Y_)
        self.cross_entropy = tf.reduce_mean(self.cross_entropy) * NR_CLASSES

        self.is_correct = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))
        # training step, learning rate = 0.003
        self.train_step = tf.train.AdamOptimizer(learningRate).minimize(self.cross_entropy)
        self.all_saver = tf.train.Saver()
        self.sess = tf.Session()
        self.stats_ac = []
        self.stats_ce = []

    def loadCheckpoint(self, checkpoint_path):
        print("Loading checkpoint file from " + checkpoint_path + "\n")
        self.all_saver.restore(self.sess, checkpoint_path)

    def saveCheckpoint(self, checkpoint_path):
        print("Writing checkpoint file to " + checkpoint_path + "\n")
        self.all_saver.save(self.sess, checkpoint_path)

    def train(self, training_data, test_data, batch_size = 1, epochs = 1, test_every = 1):
        currentCount = 0
        currentEpoch = 0
        trainingDataLength = training_data["data"].shape[0]
        print(trainingDataLength )
        self.sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            while currentCount < trainingDataLength:
                currentBatchSize = min(trainingDataLength - currentCount, batch_size)
                train_dict = {
                    self.X: training_data["data"][currentCount: currentCount + currentBatchSize],
                    self.Y_: training_data["labels"][currentCount: currentCount + currentBatchSize]
                }
                self.sess.run(self.train_step, feed_dict=train_dict)
                currentCount += currentBatchSize

                if (currentCount % test_every) == 0:
                    test_dict = {
                        self.X: test_data["data"],
                        self.Y_: test_data["labels"]
                    }
                    a, c = self.sess.run([self.accuracy, self.cross_entropy], feed_dict=test_dict)
                    print("Epoch: {} Accuracy: {} \t cross_entropy: {}".format(currentEpoch, a, c))
                    self.stats_ac.append(a)
                    self.stats_ce.append(c)

            currentCount = 0
            currentEpoch += 1

    def predict(self, data_to_predict):
        predict_dict = {
            self.X: data_to_predict
        }
        classification = self.sess.run(self.Y, predict_dict)
        return classification


def test():
    mdl = ANN_5FCL()
    mdl.initialize(2,3, learningRate= 0.0003, LEVELS = [50,32,16,8])

    # taken more or less from
    # http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html#sphx-glr-auto-examples-svm-plot-oneclass-py


    # Generate train data
    X = 0.3 * np.random.randn(10000, 2)
    X_train = np.r_[X + 2, X - 2, X + 5]
    X_train_labels = np.zeros((len(X_train), 3))
    X_train_labels[0:len(X),0] = 1
    X_train_labels[len(X):2*len(X),1] = 1
    X_train_labels[2*len(X):,2] = 1

    training_data = {
        "labels": X_train_labels,
        "data": X_train
    }

    # Generate some regular novel observations
    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2, X - 2, X + 5]
    X_test_labels = np.zeros((len(X_test), 3))
    X_test_labels[0:len(X),0] = 1
    X_test_labels[len(X):2*len(X),1] = 1
    X_test_labels[2*len(X):,2] = 1

    test_data = {
        "labels": X_test_labels,
        "data": X_test
    }
    #print(X_test_labels)

    #print(np.hstack((test_data["labels"], test_data["data"])))
    mdl.train(training_data, test_data, 100, 200, 1000)

    # test one prediction
    print(mdl.predict(X_test[0:1,:]))

    # test a lot of predictions

    xx, yy = np.meshgrid(np.linspace(-5, 8, 50), np.linspace(-5, 8, 50))
    grid_coordinates = np.swapaxes(np.vstack((np.reshape(xx, 2500), np.reshape(yy, 2500))), 0, 1)
    prediction = mdl.predict(grid_coordinates)

    b1 = plt.scatter(grid_coordinates[:,0],grid_coordinates[:,1], facecolors=prediction)
    plt.axis('tight')
    plt.xlim((-5, 8))
    plt.ylim((-5, 8))
    plt.show()


if __name__ == '__main__':
    test()
