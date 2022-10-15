# @ author Obaid Tamboli


#  Instructions

#  You may have to change the link for test labels"
# to change the classifier...Open the program and call the resp function, SVM is the Default 
#  1 for Logidtic Regression
#  2 for KNN
#  3 for Desicion Tree
#  4 for Naive Bayes
#  SVM is the Default




import csv,os,re,sys,codecs
import numpy as np
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
from sklearn.cluster import KMeans










def KNN(data,label):

    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    print('\n The cluster labels generated by K-means clustering technique is: ') 
    print(kmeans.labels_)

    # Evaluation
    print('\n *************** Confusion Matrix ***************  \n')
    print (confusion_matrix(label, kmeans.labels_)) 

    print('\n ***************  Scores on Test Data  *************** \n ')
    print(classification_report(label, kmeans.labels_, target_names=['0','1' ])) 


def logisticRegerssion(data,label):
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, label, test_size=0.20, random_state=42,stratify=label)
    print('\n\t### Training Logistic Regression Classifier ### \n')
    clf = LogisticRegression(solver='liblinear',class_weight='balanced') 
    clf_parameters = {
            'clf__random_state':(0,10),
            }     
    
    
    pipeline = Pipeline([    
    ('clf', clf),]) 

    #Classificaion
    parameters={**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10) 
    #    grid.fit(trn_data,trn_cat)  
    grid.fit(np.asarray(trn_data),np.asarray(trn_cat))    
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
    return predicted


    # Evaluation
    print('\n *************** Confusion Matrix ***************  \n')
    print (confusion_matrix(tst_cat, predicted)) 
       
          
    print('\n ***************  Scores on Test Data  *************** \n ')
    print(classification_report(tst_cat, predicted, target_names=['0','1'])) 

    # Evaluation
    print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
    print('\n Total documents in the test set: '+str(len(tst_data))+'\n')


    pr=precision_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged Precision:'+str(pr)) 

    rl=recall_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged Recall:'+str(rl))

    #fm=f1_score(tst_cat, predicted, average='macro') 
    #print ('\n Macro Averaged F1-Score :'+str(fm))

    fm=f1_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged F1-Score:'+str(fm)+'\n\n')



def desicionTree(data,label):
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, label, test_size=0.20, random_state=42,stratify=label)
    print('\n\t### Training Desicion Tree Classifier ### \n')
    clf = DecisionTreeClassifier(random_state=40) 
    clf_parameters = {
            'clf__criterion':('gini', 'entropy'), 
            'clf__max_features':('auto', 'sqrt', 'log2'),
            'clf__max_depth':(30,40,35,25),
            'clf__ccp_alpha':(0.0009,0.001,0.005,0.0001),
            }   
    
    
    pipeline = Pipeline([    
    ('clf', clf),]) 

    #Classificaion
    parameters={**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10) 
    #    grid.fit(trn_data,trn_cat)  
    grid.fit(np.asarray(trn_data),np.asarray(trn_cat))    
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
    # return predicted
    # Evaluation
    print('\n *************** Confusion Matrix ***************  \n')
    print (confusion_matrix(tst_cat, predicted)) 
       
          
    print('\n ***************  Scores on Test Data  *************** \n ')
    print(classification_report(tst_cat, predicted, target_names=['0','1'])) 

    # Evaluation
    print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
    print('\n Total documents in the test set: '+str(len(tst_data))+'\n')


    pr=precision_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged Precision:'+str(pr)) 

    rl=recall_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged Recall:'+str(rl))

    #fm=f1_score(tst_cat, predicted, average='macro') 
    #print ('\n Macro Averaged F1-Score :'+str(fm))
#
    fm=f1_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged F1-Score:'+str(fm)+'\n\n')



def naiveBayes(data,label):
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, label, test_size=0.20, random_state=42,stratify=label)
    print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
    clf = MultinomialNB(fit_prior=True, class_prior=None)  
    clf_parameters = {
            'clf__alpha':(0,1),
            }  
    pipeline = Pipeline([    
    ('clf', clf),]) 

    #Classificaion
    parameters={**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10) 
    #    grid.fit(trn_data,trn_cat)  
    grid.fit(np.asarray(trn_data),np.asarray(trn_cat))    
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
    # Evaluation
    print('\n *************** Confusion Matrix ***************  \n')
    print (confusion_matrix(tst_cat, predicted)) 
       
          
    print('\n ***************  Scores on Test Data  *************** \n ')
    print(classification_report(tst_cat, predicted, target_names=['0','1'])) 

    # Evaluation
    print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
    print('\n Total documents in the test set: '+str(len(tst_data))+'\n')


    pr=precision_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged Precision:'+str(pr)) 

    rl=recall_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged Recall:'+str(rl))

    #fm=f1_score(tst_cat, predicted, average='macro') 
    #print ('\n Macro Averaged F1-Score :'+str(fm))
#
    fm=f1_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged F1-Score:'+str(fm)+'\n\n')

    # return predicted





def SVM(data,label):
    trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, label, test_size=0.20, random_state=42,stratify=label)
    print('\n\t### Training SVM Classifier ### \n')
    clf = svm.SVC(class_weight='balanced',probability=True)  
    clf_parameters = {
            'clf__C':(1,0.5,100,200),
            # 'clf__kernel':('linear','rbf','poly','sigmoid'),
            }
    pipeline = Pipeline([    
    ('clf', clf),]) 

    #Classificaion
    parameters={**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10) 
    #    grid.fit(trn_data,trn_cat)  
    grid.fit(np.asarray(trn_data),np.asarray(trn_cat))    
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
    # Evaluation
    print('\n *************** Confusion Matrix ***************  \n')
    print (confusion_matrix(tst_cat, predicted)) 
       
          
    print('\n ***************  Scores on Test Data  *************** \n ')
    print(classification_report(tst_cat, predicted, target_names=['0','1'])) 

    # Evaluation
    print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
    print('\n Total documents in the test set: '+str(len(tst_data))+'\n')


    pr=precision_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged Precision:'+str(pr)) 

    rl=recall_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged Recall:'+str(rl))

    #fm=f1_score(tst_cat, predicted, average='macro') 
    #print ('\n Macro Averaged F1-Score :'+str(fm))
#
    fm=f1_score(tst_cat, predicted, average='macro') 
    print ('\n Macro Averaged F1-Score:'+str(fm)+'\n\n')

    # return predicted   

def SVM1(trn_data,tst_data,trn_cat):
    # trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, label, test_size=0.20, random_state=42,stratify=label)
    print('\n\t### Training SVM Classifier ### \n')
    clf = svm.SVC(class_weight='balanced',probability=True)  
    clf_parameters = {
            'clf__C':(1,0.5,100,200),
            # 'clf__kernel':('linear','rbf','poly','sigmoid'),
            }
    pipeline = Pipeline([    
    ('clf', clf),]) 

    #Classificaion
    parameters={**clf_parameters} 
    grid = GridSearchCV(pipeline,parameters,scoring='f1_micro',cv=10) 
    #    grid.fit(trn_data,trn_cat)  
    grid.fit(np.asarray(trn_data),np.asarray(trn_cat))    
    clf= grid.best_estimator_  
    print('********* Best Set of Parameters ********* \n\n')
    print(clf)
    predicted = clf.predict(tst_data)
    predicted =list(predicted)
    # Evaluation
#     print('\n *************** Confusion Matrix ***************  \n')
#     print (confusion_matrix(tst_cat, predicted)) 
       
          
#     print('\n ***************  Scores on Test Data  *************** \n ')
#     print(classification_report(tst_cat, predicted, target_names=['0','1'])) 

#     # Evaluation
#     print('\n Total documents in the training set: '+str(len(trn_data))+'\n')    
#     print('\n Total documents in the test set: '+str(len(tst_data))+'\n')


#     pr=precision_score(tst_cat, predicted, average='macro') 
#     print ('\n Macro Averaged Precision:'+str(pr)) 

#     rl=recall_score(tst_cat, predicted, average='macro') 
#     print ('\n Macro Averaged Recall:'+str(rl))

#     #fm=f1_score(tst_cat, predicted, average='macro') 
#     #print ('\n Macro Averaged F1-Score :'+str(fm))
# #
#     fm=f1_score(tst_cat, predicted, average='macro') 
#     print ('\n Macro Averaged F1-Score:'+str(fm)+'\n\n')

    return predicted    




print("Instructions:")
print(" You may to change the link for test labels")
print(" to change the classifier...Open the program and call the resp function, SVM is the Default ")
print(" For scores and summary")
print(" 1 for Logidtic Regression")
print(" 2 for KNN")
print(" 3 for Desicion Tree")
print(" 4 for Naive Bayes")
print(" 5 for SVM")

print(" SVM is the Default")

# fl=open("C:/Users/obaid/Downloads/training_data_class_labels.csv", 'r')
fl=open(".Data_sets/training_data_class_labels.csv", 'r')
# f2=open("C:/Users/obaid/Downloads/training_data.csv", 'r')
f2=open(".Data_sets/training_data.csv", 'r')
# f3=open("C:/Users/obaid/Downloads/test_data.csv", 'r')
f3=open(".Data_sets/test_data.csv", 'r')
label =  list(csv.reader(fl,delimiter=';'))
labels=[]
for i in label:
    labels.append(int(i[0]))

data =list(csv.reader(f2,delimiter=','))
test=list(csv.reader(f3,delimiter=','))

# print(labels)
pred=[]
a=input("Enter Your Choice")
if(a.isnumeric()):
    if(int(a)==1):
        pred=logisticRegerssion(data,label)
    elif(int(a)==2):
        pred=KNN(data,labels)  
    elif(int(a)==3):
        pred=desicionTree(data,label)
    elif(int(a)==4):
        pred=naiveBayes(data,label) 
    elif(int(a)==5):
        pred=SVM(data,label)      
    else:
        pred=SVM1(data,test,labels)
    # best Classification was achieved on SVM      
else:
    pred=SVM1(data,test,labels)
    # best Classification was achieved on SVM
print(pred)


with open('test_data_class_labels.txt', 'w') as f:
    for i in pred:
        i=str(i)+"\n"
        f.write(i)

print("******** Compelted *******")        
 