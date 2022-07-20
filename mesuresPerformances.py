import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
import pydot
import json
import os
from sklearn.neighbors import KNeighborsClassifier #the knn classifier
# from sklearn.naive_bayes import CategoricalNB #naive bayes classifier
import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.converters import Loader
import networkx as nx
from graphviz import Source
from networkx.readwrite import json_graph

def matrice_confusion(df,labels=[0,1],class_labl='Class',prd_labbl='predicted',binary=True):
    # class_labl[0] = negative class_labl[1] = positive
    if binary:
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for index, row in df.iterrows():
            if row[prd_labbl] == labels[1] and row[class_labl] == labels[1]:
                TP = TP + 1
            elif row[prd_labbl] == labels[0] and row[class_labl] == labels[0]:
                TN = TN + 1
            elif row[prd_labbl] == labels[1] and row[class_labl] == labels[0]:
                FP = FP +1
            elif row[prd_labbl] == labels[0] and row[class_labl] == labels[1]:
                FN = FN +1
        

        return [TP , FN , FP , TN]
    else:
        raise Exception('multiclass confusion matrix not supported yet.')


def plot_matrice_confusion(df,labels,mtx,binary=True):
    sns.set_theme()
    sns.set(font_scale=2)
    if binary:
        group_names = ['TP','FN','FP','TN']
        data = []
        for v1,v2 in zip(group_names,mtx):
            data.append(f"{v1} \n {v2}")         
        data = np.array(data).reshape(2,2)
        mtx = np.array(mtx).reshape(2,2)        
        ax = sns.heatmap(mtx, annot=data,xticklabels=labels,yticklabels=labels, fmt='', cmap='Blues', cbar=False)
        ax.set(title='Confusion Matrix', xlabel="Predicted label", ylabel="True label")
    else:
        raise Exception('multiclass confusion matrix not supported yet.')
    
    
def recall(mtx,binary=True):
    if binary:
        TP , FN , FP , TN = mtx
        try :
            return ((TP) /(TP+FN))
        except ZeroDivisionError:
            return 0
    else:
        raise Exception('Recall not supported yet.') 

def accuracy(mtx,binary=True):
    if binary:
        TP , FN , FP , TN = mtx
        try: 
            return ((TP+TN) /(TP+FN+FP+TN))
        except ZeroDivisionError:
            return 0
    else:
        raise Exception('Accuracy not supported yet.') 
  
def error(mtx,binary=True):
    if binary:
        TP , FN , FP , TN = mtx
        try: 
            return 1 - ((TP+TN) /(TP+FN+FP+TN))
        except ZeroDivisionError:
            return 0
    else:
        raise Exception('error not supported yet.') 

def precison(mtx,binary=True):
    if binary:
        TP , FN , FP , TN = mtx
        try:
            return ((TP)/(TP+FP))
        except ZeroDivisionError:
            return 0
    else:
        raise Exception('precison not supported yet.') 
  
def f_score(mtx,binary=True):
    if binary:
        TP , FN , FP , TN = mtx
        try:
            return ((2 * precison(mtx,binary) * recall(mtx,binary))/(precison(mtx,binary) + recall(mtx,binary)))
        except ZeroDivisionError:
            return 0
    else:
        raise Exception('F-score not supported yet.') 

def plot_roc(df_train,model,labels,binary=True,class_labl='Class'):
    if binary:
        roc_point = []
        thresholds = list(np.array(list(range(0, 1000+1, 1)))/1000)
        df_train[class_labl].replace(labels, [1, 0], inplace=True)
        for threshold in thresholds:
            df_train['predicted_proba'] = model.predict_proba(df_train.loc[:, df_train.columns != class_labl])
            df_train.loc[(df_train['predicted_proba'] >= threshold),'predicted_proba'] = 1
            df_train.loc[(df_train['predicted_proba'] < threshold),'predicted_proba'] = 0
            df_train['predicted_proba'].replace([1, 0], labels, inplace=True)
            TP , FN , FP , TN = matrice_confusion(df_train,labels,class_labl,prd_labbl='predicted_proba')
            TPR = TP / (TP + FN)
            FPR = FP / (TN + FP)
            roc_point.append([TPR, FPR])          
            df_train.drop('predicted_proba', axis=1, inplace=True)
        
        pivot = pd.DataFrame(roc_point, columns = ["x", "y"])
        pivot["threshold"] = thresholds
        plt.plot(pivot.y, pivot.x)
        plt.plot([0, 1])
        plt.xlabel(f"FALSE POSITIVE RATE POSITIVE LABEL ({labels[0]})")
        plt.ylabel(f"TRUE POSITIVE RATE NEGATIVE  LABEL ({labels[1]})")
        auc = round(abs(np.trapz(pivot.x, pivot.y)), 4)
        plt.title(f"AUC {auc}", loc='center')


    else:
        raise Exception('multiclass Roc Curve not supported yet.')
     


class KNN():
    def __init__(self,k,train_x,train_y):
        self.model = KNeighborsClassifier(n_neighbors = k,n_jobs=None, metric='euclidean',algorithm='brute')
        self.model.fit(train_x,train_y)
    
    def predict(self,test_x):
        return self.model.predict(test_x)


class NB():
    def __init__(self):
        self.model = Classifier(classname="weka.classifiers.bayes.NaiveBayes")   
    
    def fit(self,data):
        new_data = self.dataframe_to_csv(data)
        self.model.build_classifier(new_data)
        
    
    def predict(self,test_data):
        new_data = self.dataframe_to_csv(test_data)
        labt = test_data['Class'].sort_values().unique().tolist()
        labp = [i for i in range(len(labt))]
        lab = dict(zip(labp, labt))
        p = []
        for index, inst in enumerate(new_data):
            p.append(lab[self.model.classify_instance(inst)])
                
        return p
    
    def predict_for_one_instance(self,inst):
        return self.model.classify_instance(inst)

    def dataframe_to_csv(self,dataframe):
        try:
            os.mkdir('tempDir')
        except:
            pass
        dataframe.to_csv('tempDir/temp_Data.csv', encoding='utf-8', index=False)
        loader = Loader(classname="weka.core.converters.CSVLoader")
        dataset = loader.load_file('tempDir/temp_Data.csv')
        dataset.class_is_last()
        return dataset


class C45():
    def __init__(self):
        self.model = Classifier(classname="weka.classifiers.trees.J48")
    
    def fit(self,data):
        newdata = self.dataframe_to_csv(data)
        self.model.build_classifier(newdata)
        self.graph = self.model.graph
        # print('here')
    def predict(self,test_data):
        new_data = self.dataframe_to_csv(test_data)
        labt = test_data['Class'].sort_values().unique().tolist()
        labp = [i for i in range(len(labt))]
        lab = dict(zip(labp, labt))
        p = []
        for index, inst in enumerate(new_data):
            p.append(lab[self.model.classify_instance(inst)])
                
        return p

    def veiw_tree(self):
        s = Source(self.model.graph, filename="tree.gv", format="png")
        s.view()
    
    def dataframe_to_csv(self,dataframe):
        try:
            os.mkdir('tempDir')
        except:
            pass
        dataframe.to_csv('tempDir/temp_Data.csv', encoding='utf-8', index=False)
        loader = Loader(classname="weka.core.converters.CSVLoader")
        dataset = loader.load_file('tempDir/temp_Data.csv')
        dataset.class_is_last()
        return dataset
# class NB():
#     def __init__(self,train_x,train_y):
#         self.model = CategoricalNB()
#         self.model.fit(train_x,train_y)
    
#     def predict(self,test_x):
#         return self.model.predict(test_x)
   