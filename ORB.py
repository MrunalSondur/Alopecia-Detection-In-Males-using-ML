#!/usr/bin/env python
# coding: utf-8

# ### ---------------------------------------------------------------------------------------------------------
# ### Using Cropping: 

# In[25]:


import cv2
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as snss

from sklearn.cluster import KMeans
from sklearn import svm, datasets
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


# In[26]:


### Add paths: 
input0 = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_1\T1_AugGen"
input1 = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_2\T2_AugGen"
input2 = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_3\T3_AugGen"
input3 = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_4\T4_AugGen"


# In[29]:


#Applying ORB descriptor on folder1: TYPE 
i=0
for filename in os.listdir(input0):
    #path
    path=os.path.join(input0,filename)
    a=cv2.imread(path)
    
#     #resize image
#     resize=(224, 224)
#     img=cv2.resize(a,resize)
    
#     #gray image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(a,(3,3),0)

    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) # detects vertical edges
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) # Detects horizontal edges
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

    gX = cv2.convertScaleAbs(img_prewittx)
    gY = cv2.convertScaleAbs(img_prewitty)
    # combine the gradient representations into a single image
    gXY = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gXY, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    print("descriptor shape ",i," : ", out.shape)
    i=i+1
  
    #drop first coloumn as it's the no of feature detected. Not required.
    #append to the csv file
    csv_data=out.to_csv('i1Type1.csv', mode='a', header=False,index=False)


# In[30]:


#reading previously saved feature descriptor csv file of folder1 and save it into a dataframe
data1 = pd.read_csv('i1Type1.csv',header=None,dtype='uint8')
data1=data1.astype(np.uint8) 
data1


# In[31]:


#Applying ORB descriptor on folder2: Type2
i=0
for filename in os.listdir(input1):
    #path
    path=os.path.join(input1,filename)
    a=cv2.imread(path)
    
#     #resize image
#     resize=(224, 224)
#     img=cv2.resize(a,resize)
    
#     #gray image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Initiate FAST detector
#     star = cv2.xfeatures2d.StarDetector_create()
#     # Initiate BRIEF extractor
#     brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#     # find the keypoints with STAR
#     kp = star.detect(img,None)
#     # compute the descriptors with BRIEF
#     keypoints, descriptors = brief.compute(gray, kp)
    img_gaussian = cv2.GaussianBlur(a,(3,3),0)

    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) # detects vertical edges
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) # Detects horizontal edges
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

    gX = cv2.convertScaleAbs(img_prewittx)
    gY = cv2.convertScaleAbs(img_prewitty)
    # combine the gradient representations into a single image
    gXY = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gXY, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    print("descriptor shape ",i," : ", out.shape)
    i=i+1
  
    #drop first coloumn as it's the no of feature detected. Not required.
    #append to the csv file
    csv_data=out.to_csv('i2Type2.csv', mode='a', header=False,index=False)


# In[33]:


#reading previously saved feature descriptor csv file of folder1 and save it into a dataframe
data2 = pd.read_csv('i2Type2.csv',header=None,dtype='uint8')
data2=data2.astype(np.uint8) 
data2


# In[34]:


#Applying ORB descriptor on folder3: Type3
i=0
for filename in os.listdir(input2):
    #path
    path=os.path.join(input2,filename)
    a=cv2.imread(path)
    
#     #resize image
#     resize=(224, 224)
#     img=cv2.resize(a,resize)
    
#     #gray image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Initiate FAST detector
#     star = cv2.xfeatures2d.StarDetector_create()
#     # Initiate BRIEF extractor
#     brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#     # find the keypoints with STAR
#     kp = star.detect(img,None)
#     # compute the descriptors with BRIEF
#     keypoints, descriptors = brief.compute(gray, kp)
    img_gaussian = cv2.GaussianBlur(a,(3,3),0)

    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) # detects vertical edges
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) # Detects horizontal edges
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

    gX = cv2.convertScaleAbs(img_prewittx)
    gY = cv2.convertScaleAbs(img_prewitty)
    # combine the gradient representations into a single image
    gXY = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(a, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    print("descriptor shape ",i," : ", out.shape)
    i=i+1
  
    #drop first coloumn as it's the no of feature detected. Not required.
    #append to the csv file
    csv_data=out.to_csv('i3Type3.csv', mode='a', header=False,index=False)


# In[35]:


#reading previously saved feature descriptor csv file of folder1 and save it into a dataframe
data3 = pd.read_csv('i3Type3.csv',header=None,dtype='uint8')
data3=data3.astype(np.uint8) 
data3


# In[36]:


#Applying ORB descriptor on folder4: Type4 
i=0
for filename in os.listdir(input3):
    #path
    path=os.path.join(input3,filename)
    a=cv2.imread(path)
    
#     #resize image
#     resize=(224, 224)
#     img=cv2.resize(a,resize)
    
#     #gray image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Initiate FAST detector
#     star = cv2.xfeatures2d.StarDetector_create()
#     # Initiate BRIEF extractor
#     brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#     # find the keypoints with STAR
#     kp = star.detect(img,None)
#     # compute the descriptors with BRIEF
#     keypoints, descriptors = brief.compute(gray, kp)
    img_gaussian = cv2.GaussianBlur(a,(3,3),0)

    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]]) # detects vertical edges
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]) # Detects horizontal edges
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

    gX = cv2.convertScaleAbs(img_prewittx)
    gY = cv2.convertScaleAbs(img_prewitty)
    # combine the gradient representations into a single image
    gXY = cv2.addWeighted(img_prewittx, 0.5, img_prewitty, 0.5, 0)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gXY, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    print("descriptor shape ",i," : ", out.shape)
    i=i+1
  
    #drop first coloumn as it's the no of feature detected. Not required.
    #append to the csv file
    csv_data=out.to_csv('i4Type4.csv', mode='a', header=False,index=False)


# In[37]:


#reading previously saved feature descriptor csv file of folder1 and save it into a dataframe
data4 = pd.read_csv('i4Type4.csv',header=None,dtype='uint8')
data4=data4.astype(np.uint8) 
data4


# In[38]:


#append all the class wise feature descriptor data into one data frame
data=data1.append(data2)
data=data.append(data3)
data=data.append(data4)
data 


# In[39]:


#save appended data into a csv file
csv_data=data.to_csv('HB_finalData.csv', mode='a', header=False,index=False)


# In[40]:


#read the data from the previously saved csv file
data = pd.read_csv('HB_finalData.csv',header=None)
data


# In[41]:


#Applying Kmeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)


# In[42]:


### Converting Python Object to byte stream 
import pickle


# In[43]:


### data file created by the Statistical Package for the Social Sciences(SPSS)
filename = 'KmeansModel.pkl'
pickle.dump(kmeans, open(filename, 'wb'))


# In[44]:


hist=np.histogram(kmeans.labels_,bins=[0,1,2,3,4])


print('histogram of trained kmeans')
print(hist,"\n")


# In[45]:


inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data4)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[46]:


inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# ## ------------------------------------------------------------------------------->>> 
# ### Performing Kmeans Prediction 

# In[47]:


#performing kmeans prediction on the folder1 with the pretrained kmeans model

#initialising i=0; as it is the first class
i=0
data=[]
#k=0

for filename in os.listdir(input0):
    #path
    path=os.path.join(input0,filename)
    a=cv2.imread(path)
    
#     #resize image
#     resize=(500,500)
#     img=cv2.resize(a,resize)
    
#     #gray image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Initiate FAST detector
#     star = cv2.xfeatures2d.StarDetector_create()
#     # Initiate BRIEF extractor
#     brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#     # find the keypoints with STAR
#     kp = star.detect(img,None)
#     # compute the descriptors with BRIEF
#     keypoints, descriptors = brief.compute(gray, kp)
    
#     out=pd.DataFrame(descriptors)
    
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(a, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    
    array_double = np.array(out, dtype=np.double)
    try:
        a=kmeans.predict(array_double)
    except:
        print(filename)
    hist=np.histogram(a,bins=[0,1,2,3,4])
    
    #append the dataframe into the array 
    data.append(hist[0])
    #k=k+1
    
#convert Array to Dataframe and append to the list
Output = pd.DataFrame(data)
#add row class 
Output["Class"] = i 
csv_data=Output.to_csv('F_HB_T1.csv', mode='a',header=False,index=False)


# In[48]:


#Displaying the kmeans predicted data of folder1
print("HB_Type1")
dat1= pd.read_csv('F_HB_T1.csv',header=None)
print(dat1)


# In[49]:


#performing kmeans prediction on the folder1 with the pretrained kmeans model

#initialising i=0; as it is the first class
i=1
data=[]
#k=0

for filename in os.listdir(input1):
    #path
    path=os.path.join(input1,filename)
    a=cv2.imread(path)
    
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(a, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    
    array_double = np.array(out, dtype=np.double)
    try:
        a=kmeans.predict(array_double)
    except:
        print(filename)
    hist=np.histogram(a,bins=[0,1,2,3,4])
    
    #append the dataframe into the array 
    data.append(hist[0])
    #k=k+1
    
#convert Array to Dataframe and append to the list
Output = pd.DataFrame(data)
#add row class 
Output["Class"] = i 
csv_data=Output.to_csv('F_HB_T2.csv', mode='a',header=False,index=False)


# In[50]:


#Displaying the kmeans predicted data of folder1
print("HB_Type2")
dat2= pd.read_csv('F_HB_T2.csv',header=None)
print(dat2)


# In[51]:


#performing kmeans prediction on the folder1 with the pretrained kmeans model

#initialising i=0; as it is the first class
i=2
data=[]
#k=0

for filename in os.listdir(input2):
    #path
    path=os.path.join(input2,filename)
    a=cv2.imread(path)
    
#     #resize image
#     resize=(500,500)
#     img=cv2.resize(a,resize)
    
#     #gray image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Initiate FAST detector
#     star = cv2.xfeatures2d.StarDetector_create()
#     # Initiate BRIEF extractor
#     brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#     # find the keypoints with STAR
#     kp = star.detect(img,None)
#     # compute the descriptors with BRIEF
#     keypoints, descriptors = brief.compute(gray, kp)
    
#     out=pd.DataFrame(descriptors)
    
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(a, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    
    array_double = np.array(out, dtype=np.double)
    try:
        a=kmeans.predict(array_double)
    except:
        print(filename)
    hist=np.histogram(a,bins=[0,1,2,3,4])
    
    #append the dataframe into the array 
    data.append(hist[0])
    #k=k+1
    
#convert Array to Dataframe and append to the list
Output = pd.DataFrame(data)
#add row class 
Output["Class"] = i 
csv_data=Output.to_csv('F_HB_T3.csv', mode='a',header=False,index=False)


# In[52]:


#Displaying the kmeans predicted data of folder1
print("HB_Type3")
dat3= pd.read_csv('F_HB_T3.csv',header=None)
print(dat3)


# In[53]:


#performing kmeans prediction on the folder1 with the pretrained kmeans model

#initialising i=0; as it is the first class
i=3
data=[]
#k=0

for filename in os.listdir(input3):
    #path
    path=os.path.join(input3,filename)
    a=cv2.imread(path)
    
#     #resize image
#     resize=(500,500)
#     img=cv2.resize(a,resize)
    
#     #gray image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Initiate FAST detector
#     star = cv2.xfeatures2d.StarDetector_create()
#     # Initiate BRIEF extractor
#     brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#     # find the keypoints with STAR
#     kp = star.detect(img,None)
#     # compute the descriptors with BRIEF
#     keypoints, descriptors = brief.compute(gray, kp)
    
#     out=pd.DataFrame(descriptors)
    
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(a, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    
    array_double = np.array(out, dtype=np.double)
    try:
        a=kmeans.predict(array_double)
    except:
        print(filename)
    hist=np.histogram(a,bins=[0,1,2,3,4])
    
    #append the dataframe into the array 
    data.append(hist[0])
    #k=k+1
    
#convert Array to Dataframe and append to the list
Output = pd.DataFrame(data)
#add row class 
Output["Class"] = i 
csv_data=Output.to_csv('F_HB_T4.csv', mode='a',header=False,index=False)


# In[54]:


#Displaying the kmeans predicted data of folder1
print("HB_Type4")
dat4= pd.read_csv('F_HB_T4.csv',header=None)
print(dat4)


# In[55]:


#appending All kmeans predicted data into 1 dataframe
A = dat1.append(dat2)
A = A.append(dat3)
A = A.append(dat4)
A


# In[56]:


#save the predicted data into csv file
csv_data=A.to_csv('_FHB_Final.csv', mode='a',header=False,index=False)


# In[57]:


#read the data from the previously saved csv file
A = pd.read_csv("_FHB_Final.csv",header=None)
A


# In[58]:


#assigning x the columns from 0 to 2 for training
x = A.iloc[:,0:4].values ## Seperating column from the rows 
x

#performing Normalization 
from sklearn import preprocessing
a=preprocessing.normalize(x, axis=0)
scaled_df= pd.DataFrame(a)
x=scaled_df


# In[59]:


#assigning y with the column 3 as target variable
y = A.iloc[:,4].values
y


# In[60]:


## HBty = hair bald Types
## Ft = Final types
from sklearn.preprocessing import StandardScaler
HBty = StandardScaler()
HBty_f = HBty.fit_transform(x)
HBty_f


# In[61]:


### Algorithm for reducing the dimensionality of the data - Unsupervised Approach 
from sklearn.decomposition import PCA


# In[62]:


pca = PCA(n_components=None)
pca.fit(HBty_f)


# In[63]:


### set of observations of possibly correlated variables into a set of values 
#of linearly uncorrelated variables called principal components.
Fty= pca.transform(HBty_f)
Fty


# In[64]:


### A Pandas DataFrame is a 2 dimensional data structure, like a 2 dimensional array, or a table with rows and columns.
Fty = pd.DataFrame(Fty)
Fty


# In[65]:


Fty.shape


# In[66]:


### used to get the ration of variance (eigenvalue / total eigenvalues)
print(pca.explained_variance_ratio_) 


# In[67]:


#select the number of components such that the amount of variance that needs to be explained
pca = PCA(n_components=4)
pca.fit(HBty_f)


# In[68]:


import pickle


# In[287]:


# save the model to disk
import pickle

filename = 'PCA_Model.sav'
pickle.dump(kmeans, open(filename, 'wb'))


# In[69]:


Fty = pca.transform(HBty_f)
Fty


# In[289]:


print(pca.explained_variance_ratio_) 


# In[70]:


Fty = pd.DataFrame(Fty)
Fty


# In[72]:


B = pd.concat([Fty, pd.DataFrame(y)],axis=1)
B


# ## -------------------------------------------------------------------------------->> 
# ## PCA_Final CSV_

# In[73]:


csv_data=B.to_csv('FS_PCA_Final.csv', mode='a',header=False,index=False)


# In[74]:


data= pd.read_csv('FS_PCA_Final.csv',header=None)

data


# ## ------------------------------------------------------------------>>> 
# ### Training and Testing of Data 

# In[75]:


#assigning x the columns from 1 to 128 for training
x = data.iloc[:,0:4].values
print("X values")
print(x)

#assigning y with the column "Class" as target variable
y = data.iloc[:,4]
print("Y values")
print(y)


# In[76]:


#Dataset split into train and test with 80% Training and 20% Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=8)


# # ------------------------------------------------------------>> 
# ## Classifiers 

# In[83]:


#KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
 
knn = KNeighborsClassifier(n_neighbors=3)
     
knn.fit(x_train, y_train)
knn.fit(x_test, y_test) 
y_pred = knn.predict(x_test)

# Calculate the accuracy of the model
print("KNN Results")
print("KNN Accuracy: ",knn.score(x_test, y_test)*100,"%")


# In[84]:


# Performance Metrics >>> KNN 
from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[85]:


#Decision Tree Classifier
from sklearn.metrics import accuracy_score
#Assign model with Decision Tree classifier
model1 = DecisionTreeClassifier(max_depth=13)
filename = 'model1.sav'
pickle.dump(kmeans, open(filename, 'wb'))
#training the model with the Training Variables 
model1.fit(x_train, y_train)
#predicting the traget variable using testing variables
y_pred1 = model1.predict(x_test)
#Results
print("Decision Tree Results")
print("Decision Tree Accuracy: ",accuracy_score(y_test, y_pred1)*100,"%")

# Performance Metrics >> Decisison tree 
from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[87]:


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#Assign model with Decision Tree classifier
model4 = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None, min_samples_split=2,
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='log2',
                                max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, 
                                n_jobs=None, random_state=2, verbose=0, warm_start=False, class_weight=None, 
                                ccp_alpha=0.0, max_samples=None)
filename = 'model4.sav'
pickle.dump(kmeans, open(filename, 'wb'))
#training the model with the Training Variables 
model4.fit(x_train, y_train)
#predicting the traget variable using testing variables
y_pred4 = model4.predict(x_test)
#Results
print("RandomF Results")
print("RandomF Accuracy: ",accuracy_score(y_test, y_pred4)*100,"%")

from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[88]:


from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier

subclassifier = SVC(kernel='poly')
classifier_SVM = OneVsOneClassifier(estimator=subclassifier)

#classifier_SVM = SVC(kernel='rbf')
#classifier_SVM = SVC(kernel='linear') #accuracy -- 83.079
#classifier_SVM = SVC(kernel='poly') #accuracy -- 83.079
#classifier_SVM = SVC(kernel='sigmoid') #accuracy -- 83.079


# In[90]:


classifier_SVM.fit(x_train, y_train)


# In[91]:


predictions = classifier_SVM.predict(x_test)
#cm = metrics.confusion_matrix(y_test, predictions)
ac = (accuracy_score(y_test,y_pred)*100)
#print(cm)
print(ac)

from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[92]:


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

model = XGBClassifier(max_depth=11)
# sc = StandardScaler()
# train_X = sc.fit_transform(train_X)
# test_X = sc.transform(test_X)
model.fit(x_train, y_train)
print(model)
from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


# # ------------------------------------------------------------>>
# ## Plotting of Classifiers

# In[93]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline


# In[67]:


Final = [
    ['Name', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
    ["KNN", 0.81, 0.77, 0.67, 0.77],
    ["SVM Poly", 0.81, 0.77, 0.65, 0.71],
    ["SVM RBF", 0.78, 0.65, 0.55, 0.65],
]


# In[68]:


df = pd.DataFrame(Final[1:],columns=Final[0]).set_index('Name')
df


# In[69]:


fig, ax  = plt.subplots(1,1, figsize = (10,5))

df.plot.bar(ax = ax)
ax.legend(loc = 'best')
ax.set_xlabel("Results for ORB")


# # ------------------------------------------------------------>>
# ## Keypoint detection of Feature Extractors (SIFT, ORB)

# In[20]:


import cv2


# In[94]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#initialise sift descriptor
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

sift_image = cv2.drawKeypoints(gray, keypoints, img)
cv2.imwrite('C:\\Users\\DELL\\OneDrive\\Desktop\\Mrunal\\TY_S1\\CV\\CV_CP\\CV_CP_Fdataset\\NDS_1\\img_SIFT.jpg', orb_image)


# In[23]:


resize=(512,512)
img=cv2.resize(cv2.imread('C:\\Users\\DELL\\OneDrive\\Desktop\\Mrunal\\TY_S1\\CV\\CV_CP\\CV_CP_Fdataset\\NDS_1\\TYPE_1\\T1_HairBald\\1_jpg.rf.3ec7d4131510ceea456be72e9065ce9d - Copy (2) - Copy.jpg'),resize)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#initialise ORB descriptor
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)
orb_image = cv2.drawKeypoints(gray, keypoints, img)
cv2.imwrite('C:\\Users\\DELL\\OneDrive\\Desktop\\Mrunal\\TY_S1\\CV\\CV_CP\\CV_CP_Fdataset\\NDS_1\\img_ORB.jpg', orb_image)

