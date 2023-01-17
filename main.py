import pandas as pd
import statistics as sc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTree

banknote_datadset = pd.read_csv('BankNote_Authentication.csv')

accuracies = []
treeSizes = []

for i in range(5):
    banknote_datadset = banknote_datadset.sample(frac=1)
    dataset_features = banknote_datadset.iloc[:, :-1].values
    dataset_labels = banknote_datadset.iloc[:, -1].values.reshape(-1, 1)
    train_features, test_features, train_labels, test_labels = train_test_split(dataset_features, dataset_labels,
                                                                                test_size=0.25)
    classifier = DecisionTree(min_samples_split=3, max_depth=3)
    classifier.fit(train_features, train_labels)
    size = classifier.print_size()
    treeSizes.append(size)
    print("=============================================================================================")
    Y_pred = classifier.predict(test_features)
    acc = accuracy_score(test_labels, Y_pred)
    print("accuracy: ", acc)
    accuracies.append(float(acc * 100))
print(accuracies)
print(treeSizes)
d = pd.DataFrame()
d['accuracies'] = accuracies
d['treeSizes'] = treeSizes
d.to_csv('first 5 Runs results.csv',index=False)
print("-----------------------------------------------------------")
runs = [0.7, 0.6, 0.5, 0.4, 0.3]
df = pd.DataFrame()
accuracies = []
treeSizes = []
maximum_acc = []
minimum_acc = []
average_acc = []
max_size = []
min_size = []
avg_size = []
for i in runs:
    for j in range(5):
        banknote_datadset = banknote_datadset.sample(frac=1)
        dataset_features = banknote_datadset.iloc[:, :-1].values
        dataset_labels = banknote_datadset.iloc[:, -1].values.reshape(-1, 1)
        train_features, test_features, train_labels, test_labels = train_test_split(dataset_features, dataset_labels,
                                                                                    test_size=i)
        classifier = DecisionTree(min_samples_split=3, max_depth=3)
        classifier.fit(train_features, train_labels)
        size = classifier.print_size()
        treeSizes.append(size)
        Y_pred = classifier.predict(test_features)
        acc = accuracy_score(test_labels, Y_pred)
        accuracies.append(float(acc * 100))

    print("Training set Size = ", 1 - i)
    max_acc = max(accuracies)
    print("Maximum accuracy is : ", max_acc)
    maximum_acc.append(max_acc)
    min_acc = min(accuracies)
    print("minimum accuracy is : ", min_acc)
    minimum_acc.append(min_acc)
    avgAcc = sc.mean(accuracies)
    print("mean accuracy is : ", avgAcc)
    average_acc.append(avgAcc)
    mxTS = max(treeSizes)
    print("Maximum treeSizes is : ", mxTS)
    max_size.append(mxTS)
    mnTS = min(treeSizes)
    print("minimum treeSizes is : ", mnTS)
    min_size.append(mnTS)
    avTS = sc.mean(treeSizes)
    print("mean treeSizes is : ", avTS)
    avg_size.append(avTS)
    print("------------------------------------------------------------")

run = [0.3, 0.4, 0.5, 0.6, 0.7]
df["Traning set Size"] = run
df["Maximum accuracy"] = maximum_acc
df["Manimum accuracy"] = minimum_acc
df["mean accuracy"] = average_acc
df["maximum Tree Size"] = max_size
df["minimum Tree Size"] = min_size
df["mean Tree Size"] = avg_size
df.to_csv('runReport.csv', index=False)
plt.plot(average_acc, run)
plt.show()
