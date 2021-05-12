import tensorflow as tf
from tensorflow import keras # In this case, keras is an API for the library of tensorflow that allows us to write smaller code
import numpy as np # Use to get the max number of an array
import matplotlib.pyplot as plt # Used to show images of the data set

# One of the most important things in machine learning is to have a good dataset, so in this case 
# we chose to use the dataset called Fashion MNIST which contains a 
# total of 70,000 images of 10 different types of clothing

data = keras.datasets.fashion_mnist

# Now, after having selected our dataset, what we do now is 
# to separate our data in 2, the first one refers to 60,000 images 
# with which our neural network will be training

# While the second part refers to 10,000 images that will be 
# used to evaluate the accuracy with which the network identifies the images presented

# This is done to prevent the neural network from "cheating" and memorizing the answers

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Regarding the train_labels as well as the test_labels, both refer to the names of each image (remember that there are 10), so here 
# we created a list that indicates the name of the image according to the index of the above mentioned labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Now, before we can train our neural network with our data set, we first have to preprocess them, so if we analyze how our images are composed, 
# we will realize that they have a size of 28x28 pixels, and that they range from 0 to 255 (with respect to color)


# If we decide to work with that range of colors, we would get a little complicated 
# because there are many, so a strategy that is usually taken is to preprocess the 
# data to be within a range of 0 and 1, so that our images are in black and white contrast

train_images = train_images/255.0
test_images = test_images/255.0 

# The architecture of our model is now defined in a sequential way

model = keras.Sequential([

    # If we abstract a little bit the concept of what a neuron is, we can say that it is like a 
    # function (it receives something and delivers something), so in this case our input values 
    # are the ones we have just processed, which are the pixels of each image, which have a size of 28x28

    # However, we can't pass this entire two-dimensional array to a single neuron, so we have to 
    # flatten it so that we can work with a one-dimensional array, so that each neuron will 
    # work with each pixel, resulting in a total of 784 neurons for our input layer

    keras.layers.Flatten(input_shape=(28,28)), 

    # Subsequently we must add a layer to our network that can perform the learning analysis process, in 
    # this case the number of neurons to occupy can be determined little by little, sometimes it 
    # is usually used half or a quarter of what they are in the initial layer, I was testing various numbers 
    # of neurons, from 1 to 360, and I realized that the lower the number of neurons, the greater the chances 
    # that the network is wrong, although there is no need to exaggerate, in the end we chose to use 130 neurons

    # Then, we also have to indicate the activation function that will occupy our neurons, although there are 
    # several such as Tanh, Sigmoid or linear (which in reality has no case to use because in the end everything 
    # would be reduced to the same), we chose to use the relu function because this is one of 
    # the most popular and best activation functions that has given the best results in recent years

    keras.layers.Dense(130, activation='relu'),

    # After this, we have our output layer, which is made up of 10 neurons, the reason for this is because depending 
    # on the neuron, it will determine which type of clothing was recognized (remember that we have 10 labels for the 
    # 10 types of clothing that comprise our 70,000 data set)



    # Then, we have another activation function called softmax which practically allows us to take all the resulting values 
    # that will be within a range of 0 and 1 and distribute them in such a way that when we add them all together we get 1, so 
    # that the largest value will be the one that the neural network has selected

    keras.layers.Dense(10, activation='softmax')

    # And finally, with the use of the Dense method, it means that all the neurons in the network will 
    # be fully connected from layer to layer
])


# Now we have to define a couple of things in order to start the training process of our network, first we will 
# start defining our optimization function which will allow us to iteratively adjust the weights of the network 
# based on the training data, for this we selected the function called "adam" which can be used instead of the 
# classical stochastic gradient descent procedure.

# Then we come to the loss function whose purpose is to calculate the amount that a model should try to minimize 
# during training, in this case we selected the function named as "sparse_categorical_crossentropy" which produces a 
# category index of the most likely match category

# Finally we have the metrics which are used to control the training and evaluation process, in this case 
# we used the one called accuracy which calculates the frequency with which the predictions coincide with the labels

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Now we have to train our neural network, for this we use the function "fit" that receives as parameters our 
# preprocessed data set that we defined as "train_images" and "train_labels", besides this we have to 
# define how many times it will train or practice with our data set within the parameter called epochs, 
# this will be done randomly to have the greatest possible variety, in this case it was defined to occur 6 times

model.fit(train_images, train_labels, epochs=6)


# The following 3 lines of code were only used to adjust both the number of internal neurons and the epochs

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Tested Accuracy is: {test_acc}')
print(f'Tested Loss is: {test_loss}')

# Now comes the moment of truth where we verify how well our neural network can predict or identify 
# the test images (contained in the vairble test_images)


# The test will be performed against 5 images, which will be displayed on the screen with two texts, the 
# first one indicating the prediction made by the neural network (located at the top) and the second one 
# indicating the current label of the image being displayed (located at the bottom)

prediction = model.predict(test_images)
for i in range(5): 
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Actual Image: {class_names[test_labels[i]]}")
    plt.title(f"Prediction: {class_names[np.argmax(prediction[i])]}")
    plt.show()


# Note, this code was made possible thanks to the official tensorflow tutorial, here the reference: https://www.tensorflow.org/tutorials/keras/classification?hl=en