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


# Create a function to load our model
def load_model(model_path):
  # load the model from specific path
  print(f"Loading the saved model from :{model_path}")
  model=tf.keras.models.load_model(model_path,custom_objects={"KerasLayer":hub.KerasLayer})

  return model


model_with_earlystop=load_model("/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/models/20230820-06421692513779-full-image-set-resnet50-Adam-with-early-stop.h5")


# Custom image prediction from user

custom_path="/content/drive/MyDrive/Colab_Notebooks/Deepfake_Detection/custom_user_data/"
custom_image_path=[custom_path + fname for fname in os.listdir(custom_path)]
custom_image_path


#Create data batches for custom images
custom_data = create_data_batches(custom_image_path,
                                  test_data=True)
custom_data


# Predicting custom images with model
custom_preds = model_with_earlystop.predict(custom_data)
custom_preds

import numpy as np
labels=both_lab['label']
labels=np.array(labels)
labels

unique_labels=np.unique(labels)
unique_labels

# Turn prediction probability into their respective label

def get_pred_label(prediction_probabilities):
  return unique_labels[np.argmax(prediction_probabilities)]


# Get custom image prediction label
custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
custom_pred_labels