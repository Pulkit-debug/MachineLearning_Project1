#!/usr/bin/env python
# coding: utf-8

# In[1]:



import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# In[2]:


import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline # Only use this if using iPython')
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')


# In[3]:



# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


# In[4]:



# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dense(64, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))


# In[5]:


from keras.optimizers import Adam
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=20)


# In[6]:

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy: %.2%%", test_acc)
predic = model.evaluate(x_test, y_test)

print(predic[1]*100)
#acc = print("%.2%%" % (test_acc*100))
#model.save("cnn_model.h5")
f = open("prediction.txt", "w")
stringparse=str(predic[1]*100)
ac=stringparse[0]+stringparse[1]
f.write(ac)
f.close()

#CNN code
