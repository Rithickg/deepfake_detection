#Split the data for training
from sklearn.model_selection import train_test_split

#split the data into training and validation of total images
x_train,x_val,y_train,y_val =train_test_split(x[:num_images],
                                              y[:num_images],
                                              test_size=0.2,
                                              random_state=45)
len(x_train),len(y_train),len(x_val),len(y_val)

x_train[:3],y_train[:3]

# First Convert Images to Numpy array
from matplotlib.pyplot import imread
image=imread(x[8])
image

image.shape