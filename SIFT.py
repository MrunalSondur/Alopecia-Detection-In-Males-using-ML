#!/usr/bin/env python
# coding: utf-8

# ### ---------------------------------------------------------------------------------------------------------
# ### Using Cropping: 

# In[6]:


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
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn import svm, datasets
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


# In[9]:


### Add paths: 
input0 = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_1\T1_AugGen"
input1 = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_2\T2_AugGen"
input2 = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_3\T3_AugGen"
input3 = r"C:\Users\DELL\OneDrive\Desktop\Mrunal\TY_S1\CV\CV_CP\CV_CP_Fdataset\NDS_1\TYPE_4\T4_AugGen"


# In[10]:


#Applying SIFT descriptor on folder1: TYPE 
i=0
for filename in os.listdir(input0):
    #path
    path=os.path.join(input0,filename)
    a=cv2.imread(path)
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(a, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    print("descriptor shape ",i," : ", out.shape)
    i=i+1
  
    #drop first coloumn as it's the no of feature detected. Not required.
    #append to the csv file
    csv_data=out.to_csv('Si1Type1.csv', mode='a', header=False,index=False)


# In[11]:


#reading previously saved feature descriptor csv file of folder1 and save it into a dataframe
data1 = pd.read_csv('Si1Type1.csv',header=None,dtype='uint8')
data1=data1.astype(np.uint8) 
data1


# In[12]:


#Applying ORB descriptor on folder2: Type2
i=0
for filename in os.listdir(input1):
    #path
    path=os.path.join(input1,filename)
    a=cv2.imread(path)
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(a, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    print("descriptor shape ",i," : ", out.shape)
    i=i+1
  
    #drop first coloumn as it's the no of feature detected. Not required.
    #append to the csv file
    csv_data=out.to_csv('Si2Type2.csv', mode='a', header=False,index=False)


# In[13]:


#reading previously saved feature descriptor csv file of folder1 and save it into a dataframe
data2 = pd.read_csv('Si2Type2.csv',header=None,dtype='uint8')
data2=data2.astype(np.uint8) 
data2


# In[14]:


#Applying ORB descriptor on folder3: Type3
i=0
for filename in os.listdir(input2):
    #path
    path=os.path.join(input2,filename)
    a=cv2.imread(path)
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(a, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    print("descriptor shape ",i," : ", out.shape)
    i=i+1
  
    #drop first coloumn as it's the no of feature detected. Not required.
    #append to the csv file
    csv_data=out.to_csv('Si3Type3.csv', mode='a', header=False,index=False)


# In[15]:


#reading previously saved feature descriptor csv file of folder1 and save it into a dataframe
data3 = pd.read_csv('Si3Type3.csv',header=None,dtype='uint8')
data3=data3.astype(np.uint8) 
data3


# In[16]:


#Applying ORB descriptor on folder4: Type4 
i=0
for filename in os.listdir(input3):
    #path
    path=os.path.join(input3,filename)
    a=cv2.imread(path)
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(a, None)
    
    #convert the descriptor array into a dataframe format
    out=pd.DataFrame(descriptors)
    print("descriptor shape ",i," : ", out.shape)
    i=i+1
  
    #drop first coloumn as it's the no of feature detected. Not required.
    #append to the csv file
    csv_data=out.to_csv('Si4Type4.csv', mode='a', header=False,index=False)


# In[17]:


#reading previously saved feature descriptor csv file of folder1 and save it into a dataframe
data4 = pd.read_csv('Si4Type4.csv',header=None,dtype='uint8')
data4=data4.astype(np.uint8) 
data4


# In[20]:


#append all the class wise feature descriptor data into one data frame
data=data1.append(data2)
data=data.append(data3)
data=data.append(data4)
data 


# In[21]:


#save appended data into a csv file
csv_data=data.to_csv('HB_finalDataSIFT.csv', mode='a', header=False,index=False)


# In[22]:


#read the data from the previously saved csv file
data = pd.read_csv('HB_finalDataSIFT.csv',header=None)
data


# In[23]:


# inertias = []
# for i in range(1,11):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(data1)
#     inertias.append(kmeans.inertia_)

# plt.plot(range(1,11), inertias, marker='o')
# plt.title('Elbow method')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()


# In[24]:


# #### --------------- Applying Kmeans ----------------------

# #FOR TYPE 1
# kmeans0 = KMeans(n_clusters=3)
# kmeans0.fit(data1)

# #FOR TYPE 2
# kmeans1 = KMeans(n_clusters=3)
# kmeans1.fit(data2)

# #FOR TYPE 3
# kmeans2 = KMeans(n_clusters=3)
# kmeans2.fit(data3)

# #FOR TYPE 4
# kmeans3 = KMeans(n_clusters=3)
# kmeans3.fit(data4)


# In[26]:


#Applying Kmeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)


# In[27]:


### Converting Python Object to byte stream 
import pickle


# In[28]:


### data file created by the Statistical Package for the Social Sciences(SPSS)
filename = 'KmeansModel.sav'
pickle.dump(kmeans, open(filename, 'wb'))


# In[29]:


hist=np.histogram(kmeans.labels_,bins=[0,1,2,3,4])


print('histogram of trained kmeans')
print(hist,"\n")


# ## ------------------------------------------------------------------------------->>> 
# ### Performing Kmeans Prediction 

# In[30]:


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


# In[31]:


#Displaying the kmeans predicted data of folder1
print("HB_Type1")
dat1= pd.read_csv('F_HB_T1.csv',header=None)
print(dat1)


# In[32]:


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


# In[33]:


#Displaying the kmeans predicted data of folder1
print("HB_Type2")
dat2= pd.read_csv('F_HB_T2.csv',header=None)
print(dat2)


# In[34]:


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


# In[35]:


#Displaying the kmeans predicted data of folder1
print("HB_Type3")
dat3= pd.read_csv('F_HB_T3.csv',header=None)
print(dat3)


# In[36]:


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


# In[37]:


#Displaying the kmeans predicted data of folder1
print("HB_Type4")
dat4= pd.read_csv('F_HB_T4.csv',header=None)
print(dat4)


# In[38]:


#appending All kmeans predicted data into 1 dataframe
A = dat1.append(dat2)
A = A.append(dat3)
A = A.append(dat4)
A


# In[39]:


#save the predicted data into csv file
csv_data=A.to_csv('_FHB_Final.csv', mode='a',header=False,index=False)


# In[40]:


#read the data from the previously saved csv file
A = pd.read_csv("_FHB_Final.csv",header=None)
A


# In[41]:


#assigning x the columns from 0 to 2 for training
x = A.iloc[:,0:4].values ## Seperating column from the rows 
x

#performing Normalization 
from sklearn import preprocessing
a=preprocessing.normalize(x, axis=0)
scaled_df= pd.DataFrame(a)
x=scaled_df


# In[42]:


#assigning y with the column 3 as target variable
y = A.iloc[:,4].values
y


# In[43]:


## HBty = hair bald Types
## Ft = Final types
from sklearn.preprocessing import StandardScaler
HBty = StandardScaler()
HBty_f = HBty.fit_transform(x)
HBty_f


# In[44]:


### Algorithm for reducing the dimensionality of the data - Unsupervised Approach 
from sklearn.decomposition import PCA


# In[45]:


pca = PCA(n_components=None)
pca.fit(HBty_f)


# In[46]:


### set of observations of possibly correlated variables into a set of values 
#of linearly uncorrelated variables called principal components.
Fty= pca.transform(HBty_f)
Fty


# In[47]:


### A Pandas DataFrame is a 2 dimensional data structure, like a 2 dimensional array, or a table with rows and columns.
Fty = pd.DataFrame(Fty)
Fty


# In[48]:


Fty.shape


# In[49]:


### used to get the ration of variance (eigenvalue / total eigenvalues)
print(pca.explained_variance_ratio_) 


# In[50]:


#select the number of components such that the amount of variance that needs to be explained
pca = PCA(n_components=4)
pca.fit(HBty_f)


# In[51]:


# save the model to disk
import pickle

filename = 'PCA_Model.sav'
pickle.dump(kmeans, open(filename, 'wb'))


# In[52]:


Fty = pca.transform(HBty_f)
Fty


# In[53]:


print(pca.explained_variance_ratio_) 


# In[54]:


Fty = pd.DataFrame(Fty)
Fty


# In[55]:


B = pd.concat([Fty, pd.DataFrame(y)],axis=1)
B


# ## -------------------------------------------------------------------------------->> 
# ## PCA_Final CSV_

# In[56]:


csv_data=B.to_csv('FS_PCA_Final.csv', mode='a',header=False,index=False)


# In[7]:


data= pd.read_csv('FS_PCA_Final.csv',header=None)

data


# ## ------------------------------------------------------------------>>> 
# ### Training and Testing of Data 

# In[8]:


#assigning x the columns from 1 to 128 for training
x = data.iloc[:,0:4].values
print("X values")
print(x)

#assigning y with the column "Class" as target variable
y = data.iloc[:,4]
print("Y values")
print(y)


# In[9]:


#Dataset split into train and test with 80% Training and 20% Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=0)


# # ------------------------------------------------------------>> 
# ## Classifiers 

# In[76]:


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


# In[61]:


# Performance Metrics >>> KNN 
from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[62]:


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


# In[63]:


# Performance Metrics >> Decisison tree 
from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[77]:


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

# Performance Metrics >> Decisison tree 
from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[78]:


from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier

subclassifier = SVC(kernel='rbf')
classifier_SVM = OneVsOneClassifier(estimator=subclassifier)

#classifier_SVM = SVC(kernel='rbf')
#classifier_SVM = SVC(kernel='linear') #accuracy -- 77.079
#classifier_SVM = SVC(kernel='poly') #accuracy -- 77.079


# In[79]:


classifier_SVM.fit(x_train, y_train)


# In[80]:


predictions = classifier_SVM.predict(x_test)
#cm = confusion_matrix(y_test, predictions)
ac = (accuracy_score(y_test,y_pred)*100)
#print(cm)
print(ac)

# Performance Metrics >> Decisison tree 
from sklearn import metrics

print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


# In[1]:


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

model = XGBClassifier(max_depth=8)
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


# In[ ]:





# In[ ]:


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


# In[ ]:


Final = [
    ['Name', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
    ["K-Nearest Neigbours", 0.8975694444444444, 0.904694444444444, 0.86756900144444, 0.884567000000000],
    ["XgBoost Classfier", 0.921875, 0.9224170445678528, 0.922408816583574, 0.9217210144927536],
    ["SVM Linear", 0.9305555555555556, 0.933284724753919, 0.9321525408904051, 0.9302496018276717],
    ["SVM Sigmoid", 0.8576388888888888, 0.8567416736219573, 0.8590483687571067, 0.9302496018276717],
    ["SVM RBF", 0.9097222222222222, 0.9143692768924953, 0.9127350651622496, 0.909346872345222],
]


# In[ ]:


df = pd.DataFrame(Final[1:],columns=Final[0]).set_index('Name')
df


# In[ ]:


fig, ax  = plt.subplots(1,1, figsize = (10,5))

df.plot.bar(ax = ax)
ax.legend(loc = 'best')
ax.set_xlabel("Results for ORB")

