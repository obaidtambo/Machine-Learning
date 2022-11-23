import os

# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, metrics
from sklearn.cluster import (DBSCAN, AffinityPropagation,
                             AgglomerativeClustering, Birch, KMeans, MeanShift,
                             SpectralClustering)
from tqdm import tqdm_notebook

 

#  well intersect two classifiers to get the best output
def plotter_inter(df,labels, labels1):

    x1=[]
    y1=[]
    x0=[]
    y0=[]
    new_lab=[]
    i=0
    for l in labels:
        x=df['X'][i]
        y=df['Y'][i]
        l1=labels1[i]
        if(l!=1 and l!=0):
            # print("l")
            l=l1
            new_lab.append(l)
        else:
            new_lab.append(l)


        if (l==1):
    #        cluster1.append([x,y])
            x1.append(x)
            y1.append(y)
        else:
            x0.append(x)
            y0.append(y)
        i=i+1  
    plt.scatter(x1, y1, c ="blue")
    plt.scatter(x0, y0, c ="red")
    plt.grid()
    # changing the size of figure to 2X2
    # plt.figure(figsize=(15, 15))
 # To show the plot
    plt.show()
    return new_lab


# cluster1=[]
def plotter(df,labels):

    x1=[]
    y1=[]
    x0=[]
    y0=[]
    i=0
    for l in labels:
        x=df['X'][i]
        y=df['Y'][i]
        if (l==1):
    #        cluster1.append([x,y])
            x1.append(x)
            y1.append(y)
        else:
            x0.append(x)
            y0.append(y)
        i=i+1  
    plt.scatter(x1, y1, c ="blue")
    plt.scatter(x0, y0, c ="red")
    plt.grid()
    # changing the size of figure to 2X2
    # plt.figure(figsize=(15, 15))
 # To show the plot
    plt.show()


def comp(labels1):
    label_comp=[]
    for l in labels1:
        if l==1:
            label_comp.append(0)
        # print("change")
        else:
            label_comp.append(1)  
    return label_comp  
              


df = pd.read_csv("C:/Users/obaid/Downloads/data.csv",names=['X','Y'])
plt.scatter(df['X'], df['Y'], c ="blue")
 # To show the plot
plt.show()

print("Instructions:")
print(" You may to change the link for test labels")
print(" to change the classifier...follow the instructions ")

print(" 1 for Birch")
print(" 2 for AgglomerativeClustering")
print(" 3 for SpectralClustering")
print(" 4 for DBscan")
print(" press Any other: adefault is DBscan and spectral intersection")

labels1=[]
labels=[]
a=input("Enter Your Choice:  ")
if(a.isnumeric()):
    if(int(a)==1):
        # Fitting Birch to data
        print("# Fitting Birch to data")
        bicrh_clust_model = Birch(n_clusters=2)
        bicrh_clust_model.fit(df)
        labels1 = bicrh_clust_model.labels_
        plotter(df,labels1)

    elif(int(a)==2):
        print("# Fitting Algomerative to data")
        algo_clust_model=AgglomerativeClustering(n_clusters=2 , linkage='average')
        algo_clust_model.fit(df)  
        labels1 = algo_clust_model.labels_
        plotter(df,labels1)

    elif(int(a)==3):
        print("# Fitting spectral to data")
        spec_clust_model=SpectralClustering(n_clusters=2, random_state=1,affinity='nearest_neighbors')
        spec_clust_model.fit(df)
        labels1 = spec_clust_model.labels_
        plotter(df,labels1)

    elif(int(a)==4):
        print("# Fitting DBscan to data")
        dbscan = DBSCAN(eps = 0.082, min_samples = 10).fit(df) # fitting the model
        labels1 = dbscan.labels_ 
        plotter(df,labels1)

      
    else:
        print("# Fitting default to data")
        dbscan = DBSCAN(eps = 0.082, min_samples = 10).fit(df) # fitting the model
        labels = dbscan.labels_ 

        spec_clust_model=SpectralClustering(n_clusters=2, random_state=1,affinity='nearest_neighbors')
        spec_clust_model.fit(df)
        labels1 = spec_clust_model.labels_
        labels1= comp(labels1)
        labels1=plotter_inter(df,labels,labels1)

with open('class_labels.txt', 'w') as f:
    for i in labels1:
        i=str(i)+"\n"
        f.write(i)    # best 


print("******** Compelted *******")    