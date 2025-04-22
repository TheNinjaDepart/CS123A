import pandas as pd
import numpy as num 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Reading in data
data = pd.read_csv("C:\python\LSDS-54_microscopy_ConfocalMicroscopy_627_TRANSFORMED.csv")
metadata = pd.read_csv("C:\python\s_OSD-628.txt", sep='\t')
subset_meta = metadata[["Factor Value[Growth Environment]"]]

# Setting the attributes and target variables
X = data.values[:,1:4]
Y = subset_meta

# Training the model
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3, random_state = 69)
clf = DecisionTreeClassifier(criterion = "gini", random_state = 69, max_depth = 5, min_samples_leaf = 10)
clf.fit(X_train, Y_train)

# Running a prediction test
Y_pred = clf.predict(X_test)

# Evaluating the model
confuse_matrix = confusion_matrix(Y_test, Y_pred)
test_acc = accuracy_score(Y_test, Y_pred)
print(confuse_matrix)
print(test_acc)

# Visualizing the tree
plt.figure(figsize=(20, 15))
plot_tree(clf, filled = True, feature_names = ['biofilm_mass', 'mean_thickness', 'surface_coverage'], class_names = ['Carbon Fiber', 'SS316', 'Silicone', 'MIT Grass', 'Titanium', 'AL6061'], rounded = True)
plt.show()