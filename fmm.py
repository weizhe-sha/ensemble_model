import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import defaultdict


class FuzzyMinMaxNN:

    def __init__(self, sensitivity = 1, theta=0.4, weighting_factor = 0.6):
        self.gamma = sensitivity
        self.hyperboxes = {}
        self.classes = None
        self.V = []
        self.W = []
        self.U = []
        self.hyperbox_class = []
        self.theta = theta
        self.C_j = defaultdict(int)
        self.max_C_j = defaultdict(int)
        self.P_j = defaultdict(int)
        self.max_P_j = defaultdict(int)
        self.CF_j = {}
        self.CF_gamma = weighting_factor
        self.trust = 0
        self.alpha = 0.05
        self.beta = 0.1

    def fuzzy_membership(self, x, v, w, gamma=1):
        """
            returns the fuzzy menbership function :
            b_i(xh,v,w) = 1/2n*(------)
        """
        return ((sum([max(0, 1 - max(0, gamma * min(1, x[i] - w[i]))) for i in range(len(x))]) + sum(
            [max(0, 1 - max(0, gamma * min(1, v[i] - x[i]))) for i in range(len(x))])) / (2 * len(x))
                )

    def get_hyperbox(self, x, d):

        tmp = [0] * self.classes
        tmp[d[0]] = 1

        """
            If no hyperbox present initially so create new
        """
        if len(self.V) == 0 and len(self.W) == 0:
            self.V.append(x)
            self.W.append(x)
            self.hyperbox_class.append(d)
            self.U.append(tmp)
            expand = False
            return len(self.V) - 1, expand

        """
            returns the most sutaible hyperbox for input pattern x
            otherwise None
        """
        mylist = []
        for i in range(len(self.V)):

            if self.hyperbox_class[i] == d:
                mylist.append((self.fuzzy_membership(x, self.V[i], self.W[i])))
            else:
                mylist.append(-1)

        if len(mylist) > 0:
            for box in sorted(mylist)[::-1]:
                i = mylist.index(box)

                n_theta = sum([(max(self.W[i][j], x[j]) - min(self.V[i][j], x[j])) for j in range(len(x))])

                if len(x) * self.theta >= n_theta:
                    expand = True
                    return i, expand
            '''
                No hyperbox follow expansion criteria so create new
            '''
            self.V.append(x)
            self.W.append(x)
            self.hyperbox_class.append(d)
            self.U.append(tmp)
            expand = False
            return len(self.V) - 1, expand

        else:
            """
                If no hyperbox present for pattern x of class d so create new 
            """
            self.V.append(x)
            self.W.append(x)
            self.hyperbox_class.append(d)
            self.U.append(tmp)
            expand = False
            return len(self.V) - 1, expand

    def expand(self, x, key):
        self.V[key] = [min(self.V[key][i], x[i]) for i in range(len(x))]
        self.W[key] = [max(self.W[key][i], x[i]) for i in range(len(x))]

    def overlap_Test(self):
        del_old = 1
        del_new = 1
        box_1, box_2, delta = -1, -1, -1
        for j in range(len(self.V)):
            for k in range(j + 1, len(self.V)):

                for i in range(len(self.V[j])):

                    """
                        Test Four cases given by Patrick Simpson
                    """

                    if (self.V[j][i] < self.V[k][i] < self.W[j][i] < self.W[k][i]):
                        del_new = min(del_old, self.V[j][i] - self.V[k][i])
                    elif (self.V[k][i] < self.V[j][i] < self.W[k][i] < self.W[j][i]):
                        del_new = min(del_old, self.W[k][i] - self.V[j][i])
                    elif (self.V[j][i] < self.V[k][i] < self.W[k][i] < self.W[j][i]):
                        del_new = min(del_old, min(self.W[k][i] - self.V[j][i], self.W[j][i] - self.V[k][i]))
                    elif (self.V[k][i] < self.V[j][i] < self.W[j][i] < self.W[k][i]):
                        del_new = min(del_old, min(self.W[j][i] - self.V[k][i], self.W[k][i] - self.V[j][i]))

                    """
                        Check dimension for which overlap is minimum
                    """
                    # print(del_old , del_new , del_old-del_new , i)
                    if del_old - del_new > 0.0:
                        delta = i
                        box_1, box_2 = j, k
                        del_old = del_new
                        del_new = 1
                    else:
                        pass

        return delta, box_1, box_2

    def contraction(self, delta, box_1, box_2):
        if (self.V[box_1][delta] < self.V[box_2][delta] < self.W[box_1][delta] < self.W[box_2][delta]):
            self.W[box_1][delta] = (self.W[box_1][delta] + self.V[box_2][delta]) / 2
            self.V[box_2][delta] = (self.W[box_1][delta] + self.V[box_2][delta]) / 2

        elif (self.V[box_2][delta] < self.V[box_1][delta] < self.W[box_2][delta] < self.W[box_1][delta]):
            self.W[box_2][delta] = (self.W[box_2][delta] + self.V[box_1][delta]) / 2
            self.V[box_1][delta] = (self.W[box_2][delta] + self.V[box_1][delta]) / 2

        elif (self.V[box_1][delta] < self.V[box_2][delta] < self.W[box_2][delta] < self.W[box_1][delta]):
            if (self.W[box_2][delta] - self.V[box_1][delta]) < (self.W[box_1][delta] - self.V[box_2][delta]):
                self.V[box_1][delta] = self.W[box_2][delta]
            else:
                self.W[box_1][delta] = self.V[box_2][delta]

        elif (self.V[box_2][delta] < self.V[box_1][delta] < self.W[box_1][delta] < self.W[box_2][delta]):
            if (self.W[box_2][delta] - self.V[box_1][delta]) < (self.W[box_1][delta] - self.V[box_2][delta]):
                self.W[box_2][delta] = self.V[box_1][delta]
            else:
                self.V[box_2][delta] = self.W[box_1][delta]

    def test(self, x):
        mylist = []
        for i in range(len(self.V)):
            mylist.append([self.fuzzy_membership(x, self.V[i], self.W[i])])

        result = np.multiply(mylist, self.U)
        # for i in range(self.classes):
        #     print('pattern {} belongs to class {} with fuzzy membership value : {}'.format(x, i + 1, max(result[:, i])))
        # return prediction with trust value and CF of the activated node
        result_reshape = result.transpose()
        memb_values = [(np.argmax(lst), max(lst)) for lst in result_reshape]
        predict = None
        memb1, memb2 = [x for x in sorted(memb_values, key=lambda tup: tup[1], reverse=True)][:2]
        max_node = memb1[0]
        if abs(self.CF_j[memb1[0]] - self.CF_j[memb2[0]]) >= self.alpha and self.CF_j[memb1[0]] >= self.beta:
            for i, (node, _) in enumerate(memb_values):
                if node == max_node:
                    predict = i
        return (predict, self.CF_j[max_node])

    def predict(self, X, d_):
        # print("Start calculating confidence factor")
        total_sample = 0
        rejection = 0
        recognition = 0
        trust_records = []
        correct = 0
        for x, d in zip(X, d_):
            total_sample += 1
            mylist = []
            for i in range(len(self.V)):
                mylist.append([self.fuzzy_membership(x, self.V[i], self.W[i])])
            result = np.multiply(mylist, self.U)
            result_reshape = result.transpose()

            memb_values = [(np.argmax(lst),max(lst)) for lst in result_reshape]
            max_memb = 0
            max_node = None
            predict = None
            for i, (node, memb) in enumerate(memb_values):
                if max_memb < memb:
                    max_memb = memb
                    max_node = node
                    predict = i
            if max_memb and max_node:
                self.C_j[max_node] += 1
                if predict == d[0]:
                    self.P_j[max_node] += 1
                    trust_records.append([1] + memb_values)
                    correct += 1
                else:
                    trust_records.append([0] + memb_values)
            else:
                trust_records.append([0] + memb_values)
        for i, hc in enumerate(self.hyperbox_class):
            self.max_P_j[hc[0]] = max(self.max_P_j[hc[0]], self.P_j[i])
            self.max_C_j[hc[0]] = max(self.max_C_j[hc[0]], self.C_j[i])

        for i, hc in enumerate(self.hyperbox_class):
            self.CF_j[i] = (1 - self.CF_gamma) * self.C_j[i] / self.max_C_j[hc[0]] \
                        + self.CF_gamma * self.P_j[i] / self.max_P_j[hc[0]]

        # print("Confidence factor calculation completed")
        # print("Confidence Factor: " + str(self.CF_j))

        # print("Initial trust (reputation) calculation starts")

        for record in trust_records:
            node1, node2 = [x[0] for x in sorted(record[1:], key=lambda tup: tup[1], reverse=True)][:2]
            if abs(self.CF_j[node1] - self.CF_j[node2]) < self.alpha:
                rejection += 1
                continue
            if self.CF_j[node1] < self.beta:
                rejection += 1
                continue
            if record[0] == 1:
                recognition += 1
        rec_rate = recognition/total_sample
        rej_rate = rejection/total_sample
        self.trust = rec_rate/(1-rej_rate)

        # print(recognition, rejection)
        # print("Initial trust (reputation) calculation completed")
        # print("Initial trust (reputation): " + str(self.trust))

    # def calc_reputation(self, X, d_, alpha, beta):
    #     print("Start calculating initial trust value (reputation)")
    #     rejection = 0
    #     total_sample = 0
    #     for x, d in zip(X, d_):
    #         total_sample += 1
    #         mylist = []
    #         for i in range(len(self.V)):
    #             mylist.append([self.fuzzy_membership(x, self.V[i], self.W[i])])
    #         result = np.multiply(mylist, self.U)
    #         result_reshape = result.transpose()
    #
    #         memb_values = [(np.argmax(lst), max(lst)) for lst in result_reshape]
    #         max_memb = 0
    #         max_node = None
    #         predict = None
    #         node1, node2 = sorted(memb_values, key = lambda tup: tup[1], reverse=True)[1, :][:2]
    #         if abs(self.CF_j[node1] - self.CF_j[node2]) < alpha:
    #             rejection += 1
    #             continue
    #         for i, (node, memb) in enumerate(memb_values):
    #             if max_memb < memb:
    #                 max_memb = memb
    #                 max_node = node
    #                 predict = i
    #         if max_memb and max_node and predict == d[0]:
    #             self.P_j[max_node] += 1



    def train(self, X, d_):
        # print("Training starts")
        self.classes = len(np.unique(np.array(d_)))
        # for _ in range(epochs):
        #     print('epoch : {}'.format(_ + 1))
        #     print('=' * 50)

        for x, d in zip(X, d_):
            '''Get most sutaible hyperbox!!'''
            i, expand = self.get_hyperbox(x, d)
            # print('input pattern : ', x, d)
            # print('Hyperbox : {} , {} '.format(self.V[i], self.W[i]))

            if expand:
                self.expand(x, i)
                # print("Expanded Hyperbox : ", self.V[i], self.W[i])
                delta, j, k = self.overlap_Test()
                if delta != -1:
                    self.contraction(delta, j, k)
                    # print("Contracted  Hyperbox 1 : ", self.V[j], self.W[j])
                    # print("Contracted Hyperbox 2 : ", self.V[k], self.W[k])

                # print('=' * 50)

        # print("Training completed")
        # print('final hyperbox : ')
        # print('V : ', self.V)
        # print('W : ', self.W)


    def draw_box(self, ax, a, b, color):
        width = abs(a[0] - b[0])
        height = abs(a[1] - b[1])
        ax.add_patch(patches.Rectangle(a, width, height, fill=False, edgecolor=color))

    def show_hyperbox(self):
        """
            plot dataset
        """
        fig1 = plt.figure()
        ax = fig1.add_subplot(111, aspect='equal', alpha=0.7)

        """
            plot Hyperboxes
        """
        for i in range(len(self.V)):
            if self.hyperbox_class[i] == [1]:
                self.draw_box(ax, fuzzy.V[i], fuzzy.W[i], color='g')
            else:
                self.draw_box(ax, fuzzy.V[i], fuzzy.W[i], color='r')

        for i in range(len(X)):
            if self.hyperbox_class[i] == [1]:
                ax.scatter(X[i][0], X[i][1], marker='o', c='g')
            else:
                ax.scatter(X[i][0], X[i][1], marker='*', c='r')

        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Hyperboxes created during training')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.legend(('class 1','class 2'))
        plt.show()

# fuzzy = FuzzyMinMaxNN(1,theta=.275, weighting_factor=.6)
#
# bc = datasets.load_breast_cancer()
# X, y = bc.data, bc.target
# y = y.reshape(-1,1)
#
# n_samples, n_features = X.shape
#
# # split test and other datasets
# X_sample, X_test, y_sample, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)
#
# # split training and prediction datasets
# X_train, X_pred, y_train, y_pred = train_test_split(X_sample, y_sample, test_size = 0.2, shuffle=True)
#
# fuzzy.train(X_train, y_train)
#
# fuzzy.predict(X_pred, y_pred)

# fuzzy.calc_reputation(X_pred, y_pred)

# fuzzy.show_hyperbox()

# for x in X_test:
#     fuzzy.predict(x)
#     print('='*80)
