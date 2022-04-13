from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from auction_env import Manager_Selection
from manager import Manager
from agent import Agent
from fmm import FuzzyMinMaxNN
from fam import ARTMAPFUZZY
import threading

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
y = y.reshape(-1,1)

n_samples, n_features = X.shape

# split test and other datasets
X_sample, X_test, y_sample, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)

# split training and prediction datasets
X_train, X_pred, y_train, y_pred = train_test_split(X_sample, y_sample, test_size = 0.2, shuffle=True)


manager_fmm = Manager("FMM", 100)
manager_fam = Manager("FAM", 100)
fmm_agents = []
fam_agents = []


for i in range(3):
    fmm = FuzzyMinMaxNN(sensitivity=1, theta=.275, weighting_factor=.6)
    agent = Agent(i,fmm, "FMM")
    X_train, y_train = shuffle(X_train, y_train)
    X_pred, y_pred = shuffle(X_pred, y_pred)
    agent.train(X_train, y_train, X_pred, y_pred)
    fmm_agents.append(agent)

for i in range(3):
    X_train, y_train = shuffle(X_train, y_train)
    X_pred, y_pred = shuffle(X_pred, y_pred)
    fam = ARTMAPFUZZY(X_train, y_train, rhoARTa=0.7, rhoARTb=0.7, alphaARTa=0.001, betaARTa=0.8, alphaARTb=0.001,
                     betaARTb=0.8, maxValueArta=1, maxValueArtb=1, epsilon=0.001, weighting_factor = 0.6)
    agent = Agent(i,fam, "FAM")
    agent.train(X_train, y_train, X_pred, y_pred)
    fam_agents.append(agent)

test_samples = len(X_test)
correct = 0
non_predict = 0
for i, (x, y) in enumerate(zip(X_test, y_test)):
    decision1, value1 = Manager_Selection(i, x, y, manager_fmm, fmm_agents, test_samples).execute_auction()
    decision2, value2 = Manager_Selection(i, x, y, manager_fam, fam_agents, test_samples).execute_auction()
    if value1 > value2:
        final = decision1
    else:
        final = decision2
    if final:
        if final == y[0]:
            correct += 1
    else:
        non_predict += 1

print("Total number of test samples: ", test_samples)
print("Number of correctly classified test samples: ", correct)
print("Number of non-predicted samples: ", non_predict)
print("Test accuracy: " + str(correct/ (test_samples - non_predict) * 100.0) + "%")
# for i, (x, y) in enumerate(zip(X_test, y_test)):
#     auction = Manager_Selection(x,y, i, manager_fmm, fmm_agents)
#     auction1.execute_auction()
