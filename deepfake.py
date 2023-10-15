#Import TensorFlow and TensorFlow Hub 
import tensorflow as tf
import tensorflow_hub as hub


import os
import pandas as pd

# Creating CSV file for the data 

def create_dataframe(image_folder, label):
    # Get the list of image file paths in the folder
    image_files = os.listdir(image_folder)
    image_paths = [os.path.join(image_folder, file) for file in image_files]

    # Create a list of image names by extracting the filename without the directory path and extension
    image_names = [os.path.splitext(os.path.basename(path))[0] for path in image_paths]

    # Create a list of corresponding labels
    labels = [label] * len(image_paths)

    # Create a DataFrame with 'filename', 'image_name', 'label', 'binary_label', and 'file_path' columns
    df = pd.DataFrame({'filename': image_files, 'image_name': image_names, 'label': labels, 'file_path': image_paths})

    return df

if __name__ == "__main__":
    # Replace 'path_to_real_folder' and 'path_to_deepfake_folder' with the actual paths to your image folders
    real_folder = '/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/deepfake_database/train/real'
    deepfake_folder = '/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/deepfake_database/train/df'

    # Create DataFrames for real and deepfake images
    real_df = create_dataframe(real_folder, 'real')
    deepfake_df = create_dataframe(deepfake_folder, 'deepfake')

    # Combine the DataFrames for real and deepfake images
    combined_df = pd.concat([real_df, deepfake_df], ignore_index=True)

    # Create a binary label column where 'real' is labeled as 0 and 'deepfake' is labeled as 1
    combined_df['binary_label'] = combined_df['label'].apply(lambda x: 0 if x == 'real' else 1)

    # Optionally, shuffle the dataset
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    # Display the first few rows of the combined DataFrame
    print(combined_df.head())

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv('/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/both_Label_dataset.csv', index=False)


both_lab=pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/both_Label_dataset.csv")
both_lab.head()
both_lab.tail()
both_lab.info()

import numpy as np
labels=both_lab['label']
labels=np.array(labels)
labels
len(labels)


unique_labels=np.unique(labels)
unique_labels

boolean_labels=[label==unique_labels for label in labels]
boolean_labels

boolean_labels= np.array(boolean_labels).astype('int')
boolean_labels

# Creating Validation set fot the dataset
# setup x and y variable
x=both_lab['file_path'].values
y=boolean_labels


#Split the data for training
from sklearn.model_selection import train_test_split

#split the data into training and validation of total images
x_train,x_val,y_train,y_val =train_test_split(x[:num_images],
                                              y[:num_images],
                                              test_size=0.2,
                                              random_state=45)


# Creating a function to convert image to tensor

img_size =224

def process_image(image_path):
  # Take the image path and turn it into tensors
  # Read the image file using image path
  image =tf.io.read_file(image_path)
  # turn jpg image into numerical tensors with 3 color channels
  image=tf.image.decode_jpeg(image,channels=3)
  # convert color channel values from 1-255 to 0-1 values
  image =tf.image.convert_image_dtype(image,tf.float32)
  # resize the image to our desired value(244,244)
  image=tf.image.resize(image,size=[img_size,img_size])

  return image

# Divide the images into batches of 32 image so it could fit into the computers memory and process image faster 
# create a functin to return a tuple(image,label)
def get_image_label(image_path,label):
  image =process_image(image_path)
  return image,label


# Turn all our data into batches of size 32 for both x and y
batch_size=32

# Now create a function to turn data into batches
def create_data_batches(x,y=None,batch_size=batch_size,valid_data=False,test_data=False):
  # if data is test dataset, we don't have labels
  if test_data:
    print("Creating test data batches...")
    data=tf.data.Dataset.from_tensor_slices((tf.constant(x))) #only file path (no labels)
    data_batch =data.map(process_image).batch(batch_size)
    return data_batch

  # if the data is valid dataset ,we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y))) #filepath and labels
    data_batch =data.map(get_image_label).batch(batch_size)
    return data_batch

  else:
    print("Creating training data batches...")
    # Turn filepaths and labels into tensors
    data =tf.data.Dataset.from_tensor_slices((tf.constant(x),tf.constant(y)))
    # Shuffling the pathnames and labels before mapping the image processor function is faster than shuffling images
    data =data.shuffle(buffer_size=len(x))
    # Create (image,label) tuple ,This also turns image path into preprocessed images
    data =data.map(get_image_label)
    # Turn the training data into batches
    data_batch =data.batch(batch_size)
    return data_batch

# Creating training and validation data batches using previously build function with parameters
train_data = create_data_batches(x_train,y_train)
val_data =create_data_batches(x_val,y_val,valid_data=True)


import matplotlib.pyplot as plt

# Creating a function for viewing image in data batches

def show_25_images(images,labels):
  plt.figure(figsize=(15,15))
  # Loop throught 25 images
  for i in range(25):
    ax=plt.subplot(5,5,i+1)
    # Display images
    plt.imshow(images[i])
    # Add image label as title
    plt.title(unique_labels[labels[i].argmax()])
    # Turn the grid line off
    plt.axis("off")
    
    
# setup input shspe to our model same as that of trained model

input_shape=[None,img_size,img_size,3] #batch,height,width,color channel

# setup output shape to our model
output_shape=len(unique_labels)

#setup model url from tensorflow hub
model_url ="https://tfhub.dev/tensorflow/resnet_50/classification/1"


# create a function which builds a keras model
def create_model(input_shape=input_shape,output_shape=output_shape,model_url=model_url):
  print("Building model with:",model_url)

  # setup model layers
  model = tf.keras.Sequential([
      hub.KerasLayer(model_url), #layer 1 (input layer)
      tf.keras.layers.Dense(units=output_shape,
                            activation="softmax") # layer 2 (output layer)
  ])

  # compile the model

  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.Adam(),
      metrics=["accuracy"]
  )

  # build the model
  model.build(input_shape)

  return model

model =create_model()
model.summary()

# Early stopping callback ,stops our model from overfitting

early_stopping =tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                 patience=3)

import datetime
import os

# Creating a function to call tensorboard callback

def create_tensorboard_callback():
  # create log directory for storing tensorboard log data
  logdir = os.path.join('/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/tensorboard_logs',
                        # Making it log datatime to keep track
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return tf.keras.callbacks.TensorBoard(logdir)

# Create a function that trains a model
def train_model():

  # create a model
  model=create_model()

  # Create new tensorboard session everytime we train model
  tensorboard = create_tensorboard_callback()

  # Fit the model
  model.fit(x=train_data,
            epochs=num_epochs,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[tensorboard,early_stopping])

  # Return the fitted model
  return model

# Fit the model to data
model = train_model()

# Make prediction on validation data(Not used to train)
predictions = model.predict(val_data)
predictions


#Create a function to save the model
# save to model to given directory with suffix filename
def save_model(model,suffix=None):
   # create a model with directory pathname with current time
  modeldir=os.path.join("/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/models",
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
    # save the model in formet
  model_path=modeldir + "-" + suffix + ".h5"
  print(f"Saving the model to :{model_path}")
  model.save(model_path)

  return model_path

# Create a function to load our model
def load_model(model_path):
  # load the model from specific path
  print(f"Loading the saved model from :{model_path}")
  model=tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer})

  return model