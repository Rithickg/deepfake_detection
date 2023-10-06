import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

both_lab=pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/both_Label_dataset.csv")

both_lab.head()
both_lab.tail()
both_lab.describe()
both_lab.info()
both_lab['label'].value_counts()
both_lab['label'].value_counts().plot.bar(figsize=(5,5))


labels=both_lab['label']
labels=np.array(labels)
labels
len(labels)

unique_labels=np.unique(labels)
unique_labels
len(unique_labels)

print(labels[0])
labels[0]==unique_labels
boolean_labels=[label==unique_labels for label in labels]
boolean_labels
len(boolean_labels)


print(labels[0])
print(np.where(unique_labels == labels[0]))
print(boolean_labels[0].astype(int))
print(boolean_labels[2].astype(int))
boolean_labels= np.array(boolean_labels).astype('int')
boolean_labels

# Creating Validation set fot the dataset
# setup x and y variable
x=both_lab['file_path'].values
y=boolean_labels
x,len(x),y,len(y)

#Setup number of images for experimentation
num_images =1000 #@param {type:'slider', min:1000,max:10000,step:100}