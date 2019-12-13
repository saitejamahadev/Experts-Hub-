import numpy as np
from sklearn import datasets
iris=datasets.load_iris()
iris_data=iris.data
iris_labels=iris.target

indices = np.random.permutation(len(iris_data))
n_training_samples = 12
learnset_data = iris_data[indices[:-n_training_samples]]
learnset_labels = iris_labels[indices[:-n_training_samples]]
testset_data = iris_data[indices[-n_training_samples:]]
testset_labels = iris_labels[indices[-n_training_samples:]]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
colours = ("r", "b")
X = []
for iclass in range(3):
    X.append([[], [], []])
    for i in range(len(learnset_data)):
        if learnset_labels[i] == iclass:
            X[iclass][0].append(learnset_data[i][0])
            X[iclass][1].append(learnset_data[i][1])
            X[iclass][2].append(sum(learnset_data[i][2:]))
colours = ("r", "g", "y")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for iclass in range(3):
       ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
plt.show()


def distance(instance1,instance2):
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1-instance2)

def get_neighbors(training_set,labels,test_instance,k):
    distances=[]
    for index in range(len(training_set)):
        dist = distance(test_instance,training_set[index])
        distances.append((training_set[index],dist,labels[index]))
    distances.sort(key=lambda x:x[1])
    neighbors = distances[:k]
    return neighbors

def get_class(neighbors):
    class0=0
    class1=0
    class2=0
    for i in neighbors:
        if i[2]==0:
            class0 +=1
        elif i[2]==1:
            class1 +=1
        elif i[2]==2:
            class2+=1
    classes={'class0':class0,'class1':class1,'class2':class2}
    winner= max(classes,key=classes.get)
    print('point belongs to +',winner)
    
    
n=get_neighbors(learnset_data,learnset_labels,testset_data[0],3)
        

get_class(n)





from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(learnset_data, learnset_labels) 
print("Predictions form the classifier:")
print(knn.predict([testset_data[0]]))
print("Target values:")
print(testset_labels[0])
