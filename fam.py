import numpy as np
from collections import defaultdict
from sklearn import datasets
from sklearn.model_selection import train_test_split

class ART():
    Y = []
    Js = []
    _W = []

    def __init__(self, I, alpha=0.001, rho=0.7, beta=1):
        self._alpha = alpha
        self._rho = rho
        self._beta = beta
        self.setI(I)
        self.setW(np.ones((1, len(self.I[0]))))

    @staticmethod
    def layerF0(I, valueMax=0):
        IC = np.asarray(I)
        IC = ART.normalize(IC, valueMax)
        IC = ART.complement(IC)
        return IC

    @staticmethod
    def normalize(arr, valueMax=0):
        if valueMax == 0:
            valueMax = arr.max()

        if valueMax == 0:
            valueMax = 1

        I = np.divide(arr, valueMax)
        return I

    @staticmethod
    def complement(I):
        I = np.concatenate((I, (1 - I)), axis=1)
        return I

    def AND(self, arr1, arr2):
        try:
            return np.minimum(arr1, arr2)
        except Exception as e:
            print("AND", arr1, arr2)
            quit()

    @property
    def I(self):
        return self._I

    def setI(self, I):
        if isinstance(I, np.ndarray):
            self._I = I
        else:
            self._I = np.array(I)

    @property
    def W(self):
        return self._W

    def setW(self, W):
        if isinstance(W, np.ndarray):
            self._W = W
        else:
            self._W = np.array(W)

    def hadRessonance(self, IC, W, rho):
        try:
            IC = np.asarray(IC)
            x = np.asarray(self.AND(IC, W))
            y = x.sum(axis=0) / IC.sum(axis=0)
            return (y >= rho)
        except Exception as e:
            print(e)
            print("HAD", IC, W)
            quit()

    def vigilanceValue(self, IC, W):
        x = self.AND(IC, W)
        return (sum(x) / sum(IC))

    def categories(self, I, W):
        x = np.minimum(I, W)
        categories = x.sum(axis=1) / (self._alpha + W.sum(axis=1))
        return categories


class ARTFUZZY(ART):
    championIndex = 0
    championValue = 0
    categoriesArray = []

    def __init__(self, I, alpha=0.001, rho=0.7, beta=1):
        super().__init__(I, alpha, rho, beta)

    def learn(self, IC, W):
        temp1 = self._beta * self.AND(IC, W)
        temp2 = (1 - self._beta) * IC
        return temp1 + temp2

    def activate(self, i):
        temp = np.zeros(len(self.I))
        temp[i] = 1
        self.Y.append(list(temp))

    def hadRessonance(self, IC, W):
        x = self.AND(IC, W)
        return ((sum(x) / sum(IC)) >= self._rho)

    def train(self):
        for i in self.I:
            self.match(i)

    def match(self, inputValue):
        categories = self.categories(inputValue, self.W)
        champion = categories.max()
        championIndex = categories.argmax()

        while champion != 0:
            if self.hadRessonance(inputValue, self.W[championIndex]):
                self.W[championIndex] = self.learn(inputValue, self.W[championIndex])
                self.activate(championIndex)

                self.championIndex = championIndex
                self.championValue = champion

                break
            else:
                categories[championIndex] = 0
                champion = categories.max()
                championIndex = categories.argmax()
        else:
            self.setW(
                np.insert(self.W, len(self.W), inputValue, 0)
            )
            championIndex += 1
            self.activate(championIndex)
            self.championIndex = championIndex
            self.championValue = champion

        return self.championIndex

class ARTMAPFUZZY(ART):
    rho = 0
    WAB = []
    championsA = []

    def __init__(self, INPUT, OUTPUT, rhoARTa=0.7, rhoARTb=0.7, alphaARTa=0.001, betaARTa=1, alphaARTb=0.001,
                 betaARTb=1, maxValueArta=1, maxValueArtb=1, epsilon=0.001, weighting_factor = 0.6):
        self.ArtA = ARTFUZZY(self.layerF0(INPUT, maxValueArta), rho=rhoARTa, alpha=alphaARTa, beta=betaARTa)
        self.ArtB = ARTFUZZY(self.layerF0(OUTPUT, maxValueArtb), rho=rhoARTb, alpha=alphaARTb, beta=betaARTb)
        self.epsilon = epsilon

        self.rho = 1
        self.WAB = np.ones([1, OUTPUT.shape[0]])

        self.C_j = defaultdict(int)
        self.max_C_j = defaultdict(int)
        self.P_j = defaultdict(int)
        self.max_P_j = defaultdict(int)
        self.CF_j = {}
        self.CF_gamma = weighting_factor
        self.trust = 0
        self.alpha = 0.05
        self.beta = 0.1

    def train(self):
        interator = 0
        for inputB in self.ArtB.I:
            championIndexB = self.ArtB.match(inputB)

            rhoArtABase = self.ArtA._rho
            for inputA in self.ArtA.I[interator:]:
                categories = self.ArtA.categories(inputA, self.ArtA.W)
                championA = categories.max()
                championIndexA = categories.argmax()

                while championA != 0:
                    if self.ArtA.hadRessonance(inputA, self.ArtA.W[championIndexA]):

                        if self.hadRessonance(self.ArtB.Y[championIndexB], self.WAB[championIndexA], self.rho):
                            self.ArtA.W[championIndexA] = self.ArtA.learn(inputA, self.ArtA.W[championIndexA])
                            self.WAB[championIndexA] = self.activate(self.WAB[championIndexA], championIndexB)
                            break

                        else:
                            x = self.AND(inputA, self.ArtA.W[championIndexA])
                            newRho = (sum(x) / sum(inputA))

                            self.ArtA._rho = newRho + self.epsilon
                    else:
                        categories[championIndexA] = 0
                        championA = categories.max()
                        championIndexA = categories.argmax()
                else:
                    self.ArtA.setW(
                        np.insert(self.ArtA.W, len(self.ArtA.W), inputA, 0)
                    )
                    self.ArtA.activate(championIndexA + 1)
                    self.WAB = np.insert(self.WAB, len(self.WAB),
                                         self.activate(self.WAB[championIndexA], championIndexB), 0)

                self.ArtA._rho = rhoArtABase
                interator += 1
                break

    def predict(self, X, d_):
        total_sample = 0
        rejection = 0
        recognition = 0
        trust_records = []
        correct = 0
        for x, d in zip(X, d_):
            total_sample += 1
            x = np.divide(x, 1)
            x = np.concatenate((x, (1 - x)), axis=0)
            categories = self.ArtA.categories(x, self.ArtA.W)
            ind = np.argsort(categories, axis = 0)
            championIndexA = ind[-1]
            runnerupIndexA = ind[-2]
            t = list(self.WAB[championIndexA])
            predict = t.index(1)
            i = -2
            while list(self.WAB[runnerupIndexA]).index(1) == t:
                i -= 1
                runnerupIndexA = ind[i]
            self.C_j[championIndexA] += 1
            if predict == d[0]:
                self.P_j[championIndexA] += 1
                trust_records.append([1] + [championIndexA, runnerupIndexA])
                correct += 1
            else:
                trust_records.append([0] + [championIndexA, runnerupIndexA])
        for i, lst in enumerate(self.WAB):
            self.max_P_j[list(lst).index(1)] = max(self.max_P_j[list(lst).index(1)], self.P_j[i])
            self.max_C_j[list(lst).index(1)] = max(self.max_C_j[list(lst).index(1)], self.C_j[i])

        for i, lst in enumerate(self.WAB):
            if self.max_P_j[list(lst).index(1)] == 0:
                self.CF_j[i] = (1 - self.CF_gamma) * self.C_j[i] / self.max_C_j[list(lst).index(1)]
            elif self.max_C_j[list(lst).index(1)] == 0:
                self.CF_j[i] = self.CF_gamma * self.P_j[i] / self.max_P_j[list(lst).index(1)]
            else:
                self.CF_j[i] = (1 - self.CF_gamma) * self.C_j[i] / self.max_C_j[list(lst).index(1)] + \
                               self.CF_gamma * self.P_j[i] / self.max_P_j[list(lst).index(1)]

        # print("Confidence factor calculation completed")
        # print("Confidence Factor: " + str(self.CF_j))

        # print("Initial trust (reputation) calculation starts")

        for record in trust_records:
            if abs(self.CF_j[record[1]] - self.CF_j[record[2]]) < self.alpha:
                rejection += 1
                continue
            if self.CF_j[record[1]] < self.beta:
                rejection += 1
                continue
            if record[0] == 1:
                recognition += 1
        rec_rate = recognition / total_sample
        rej_rate = rejection / total_sample
        print(recognition, rejection, total_sample)
        self.trust = rec_rate / (1 - rej_rate) if rej_rate != 1 else 0

    def activate(self, W, i):
        temp = np.zeros(len(W))
        temp[i] = 1
        return list(temp)

    def test(self, INPUT, maxInputValue=1):
        INPUT = np.divide(INPUT, maxInputValue)
        INPUT = np.concatenate((INPUT, (1 - INPUT)), axis=0)
        categories = self.ArtA.categories(INPUT, self.ArtA.W)
        championA = categories.max()
        championIndexA = categories.argmax()
        t = list(self.WAB[championIndexA])
        return (t.index(1), self.CF_j[championIndexA])
        # rhoTest = self.ArtA._rho - (self.ArtA._rho * 0.1)
        # # rhoTest = self.ArtA._rho
        # while rhoTest > 0.00001:
        #     while championA != 0:
        #         if self.hadRessonance(INPUT, self.ArtA.I[championIndexA], rhoTest):
        #             t = list(self.WAB[championIndexA])
        #             print(t)
        #             artB = list(self.ArtB.W[t.index(1)])
        #             s = [str(i) for i in artB]
        #             return {
        #                 "index": t.index(1),
        #                 "ArtB": artB,
        #                 "id": "".join(s).replace(".", "")
        #             }
        #         else:
        #             categories[championIndexA] = 0
        #             championA = categories.max()
        #             championIndexA = categories.argmax()
        #     categories = self.ArtA.categories(INPUT, self.ArtA.W)
        #     championA = categories.max()
        #     championIndexA = categories.argmax()
        #     rhoTest = rhoTest - (rhoTest * 0.25)

        # return -1
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
# fam = ARTMAPFUZZY(X_train, y_train, rhoARTa=0.7, rhoARTb=0.7, alphaARTa=0.001, betaARTa=0.8, alphaARTb=0.001,
#                  betaARTb=0.8, maxValueArta=1, maxValueArtb=1, epsilon=0.001, weighting_factor = 0.6)
#
# fam.train()
# fam.predict(X_pred, y_pred)
#
# for x,y in zip(X_test,y_test):
#     print(x)
#     print(fam.test(x)==y[0])

# input  = np.array([
#         [0, 0],
#         [0, 1],
#         [1, 0],
#         [1, 1]
# ])
#
# output  = np.array([
#         [0],
#         [0],
#         [0],
#         [1],
# ])
#
# ArtMap = ARTMAPFUZZY(input, output, rhoARTa=0.6, rhoARTb=0.9)
# ArtMap.train()
#
# ArtMap.test([0, 0]) #{'index': 0, 'ArtB': [0.0, 1.0], 'id': '0010'}
# ArtMap.test([0, 1]) #{'index': 0, 'ArtB': [0.0, 1.0], 'id': '0010'}
# ArtMap.test([1, 0]) #{'index': 0, 'ArtB': [0.0, 1.0], 'id': '0010'}
# ArtMap.test([1, 1]) #{'index': 1, 'ArtB': [1.0, 0.0], 'id': '1000'}