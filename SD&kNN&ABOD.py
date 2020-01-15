import numpy as np
import pandas as pd
from scipy import stats
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager
import statistics
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.utils.data import generate_data, get_outliers_inliers

# Read the data file
data = pd.read_csv("file name")

# Read in specific column data
x1 = np.array(data[["x variable"]])
y1 = np.array(data[["y variable "]])
occupation = np.array(data[["data point label"]])

# Combine data columns
coordinates = np.column_stack((x1, y1))

##############################################################################
#statistics to determine outliers

# Generate random data
X_train, Y_train = generate_data(n_train=58, train_only=True, n_features=2)

# Assign random data points to our data
X_train = coordinates

# Initialize variables
dist = []
msd = 0
md = 0

# Find the median of the data points in each column
x_median = statistics.median(x1)
y_median = statistics.median(y1)


# Append all distances from data points to median to distance array
for i in range(0, 58):
    dist.append(math.sqrt((x1[i] - x_median) ** 2 + (y1[i] - y_median) ** 2))

# Adding all distances
for i in range(0, 58):
    md += dist[i]

# Finding the average distance
md /= 58

# Adding the distances of all data points from the average distance
for i in range(0, 58):
    msd += (dist[i] - md) ** 2

# Finding the average deviation from the average distance
msd /= 58

# Taking the square root to find the mean standard deviation
msd_new = math.sqrt(msd)

# Deciding whether each data point is an inlier or outlier
for i in range(0, 58):
    if dist[i] > md + 1.5 * msd_new or dist[i] < md - 1.5 * msd_new:
        Y_train[i] = 1
    else:
        Y_train[i] = 0

# Print the outliers with their corresponding occupation
for i in range(0, 58):
    if (Y_train[i] == 1):
        print(occupation[i], ",", coordinates[i])

#########################################################################################################
#graphing of points onto a meshgrid

# Decides how specific the graph is to decide outliers
# Range is (0, 0.5]
# Lower number -> Less outliers
# Higher number -> More outliers
outlier_fraction = 0.5

# Stores inliers and outliers in different arrays
x_outliers, x_inliers = get_outliers_inliers(X_train, Y_train)

# Finds the length of each array
n_inliers = len(x_inliers)
n_outliers = len(x_outliers)

# Separates the two features and use it to plot the data
F1 = X_train[:, [0]].reshape(-1, 1)
F2 = X_train[:, [1]].reshape(-1, 1)

# Creates a meshgrid
xx, yy = np.meshgrid(np.linspace(-100, 100, 200), np.linspace(-100, 100, 200))

# Graphs the scatter plot
# Labels inliers with blue
# Labels outliers with red and gender/occupation
for i in range(0, 58):
    if Y_train[i] == 0:
        plt.scatter(F1[i], F2[i], color="blue")
    else:
        plt.scatter(F1[i], F2[i], color="red")
        if 1 <= i <= 30:
            plt.annotate(occupation[i % 29] + " (Male)", (F1[i], F2[i]), textcoords='offset points', ha='left',
                         va='bottom')
        else:
            plt.annotate(occupation[i % 29] + " (Female)", (F1[i], F2[i]), textcoords='offset points', ha='left',
                         va='bottom')

plt.axhline(0, color='black')
plt.axvline(0, color='black')

################################################################################################
#K Nearest Neighbors using PyOD and Matplotlib

classifiers = {
    'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outlier_fraction),
    'K Nearest Neighbors (KNN)': KNN(contamination=outlier_fraction)
}

# Sets the figure size
plt.figure(figsize=(100, 100))

for i, (clf_name, clf) in enumerate(classifiers.items()):
    # Fits the dataset to the model
    clf.fit(X_train)

    # Predicts raw anomaly score
    scores_pred = clf.decision_function(X_train) * -1

    # Is the prediction of an inlier or outlier
    y_pred = clf.predict(X_train)

    # Finds the number of errors in prediction
    n_errors = (y_pred != Y_train).sum()
    print('No of Errors : ', clf_name, n_errors)

    # Code below creates visualization

    # Decides the threshold value to consider an inlier and outlier
    threshold = stats.scoreatpercentile(scores_pred, 100 * outlier_fraction)

    # Decision function that calculates the raw anomaly score for every point
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)

    subplot = plt.subplot(1, 2, i + 1)

    # Fills blue colormap from minimum anomaly score to threshold value
    subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 10), cmap=plt.cm.Blues_r)

    # Draws red contour line where anomaly score is equal to threshold
    a = subplot.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')

    # Fills orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')

    # Makes scatter plot of inliers with white dots
    b = subplot.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c='white', s=20, edgecolor='k')

    # Makes scatter plot of outliers with black dots
    c = subplot.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c='black', s=20, edgecolor='k')


    for x in range(0, 58):
        if (Y_train[x] == 0):
            temp_b = subplot.scatter(X_train[x, 0], X_train[x, 1], c='white', s=20, edgecolor='k')
        else:
            temp_c = subplot.scatter(X_train[x, 0], X_train[x, 1], c='black', s=20, edgecolor='k')

    # Builds axis
    subplot.axis('tight')

    # Labels graph with legend
    subplot.legend(
        [a.collections[0], b, c],
        ['learned decision function', 'true inliers', 'true outliers'],
        prop=matplotlib.font_manager.FontProperties(size=10),
        loc='lower right')

    # Titles graph
    subplot.set_title(clf_name)

    # Sets axis length
    subplot.set_xlim((-100, 100))
    subplot.set_ylim((-100, 100))
    subplot.axhline(0, color='black')
    subplot.axvline(0, color='black')

# Displays graph
plt.show()