# BAH_TechEx_Guided_Project_Phase2
This is a deliverable for the Guided Project Phase 2 during the BAH Technology Excellence Program

**Instructions**
This project focuses on the development and understanding of a Convolutional Neural Network (CNN)
 for recognizing handwritten digits using the MNIST dataset. 

- Setting up a Python project environment
- Performing data collection and preprocessing
- Designing and refining a CNN architecture
- Conducting model training and evaluation
- Executing results analysis and model optimization
- Preparing the model for deployment


**Overview**
This repository contains the implementation of a Convolutional Neural Network (CNN) architecture for Image Classification. This README provides the results of running a baseline model, an improved model and a best model post adjustment using the parameters from a Hyper Parameterization exercise.

**Architecture**
A Sequential model with regular densely-connected Neural Network (NN) layers was built. The input layer was assigned as layers.Input(shape=(28,28,1)). 
The following are the list of Hidden Layers that were used:

    layers.Conv2D(128, kernel_size=(3, 3), activation="relu", name='CL1'),
    layers.MaxPooling2D(pool_size=(2, 2), name='MPL1'),

    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name='CL2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='MPL2'),

    layers.Flatten(name='FL'),
    layers.Dropout(0.2),

  Finally, the model's output layer was given as: layers.Dense(num_classes, name="OUTL", activation='softmax') where num_classes was earlier determined as 10.
  
**Dataset**
The MNIST (Modified National Institute of Standards and Technology database) dataset contains a training set of 60,000 images and a test set of 10,000 images of handwritten digits. The handwritten digit images have been size-normalized and centered in a fixed size of 28×28 pixels.

**Pre-Processing Steps**
The MNIST dataset was split into four datasets of Training (X_train,y_train), Validation and Testing (X_test and y_test) after which they were normalized (dividing by 255), reshaped and augmented with ImageDataGenerator. The Pre-Processed data was saved as CNN_DATA.npz file.

**Training**
Three (3) different models were built, compiled and trained. These are the model's names:(1) build_model() (2) build_model (same as the first) and (3) best_model(). The first model used the normalized data within X_train_scaled2. The second model used the normalized AND augmented data within the train_generator and validation_generator. The final model used the recommended parameter choices from the hyperopt HyperParameter functions to run a final adjusted model. The epochs used for the first two models were 5 each but 10 for the last model. The model.fit was used for the first model with a batch_size set to 128 and verbose =1. When compiling this model, the Optimizer used was the Stochastic Gradient Descent(SGD) with a learning ratee of 1e-2, momentum =0.9 and nesterov=True.

For the five epochs, here were the results:
**Epoch 1/5 --> loss: 0.4090 - sparse_categorical_accuracy: 0.8769
**Epoch 2/5 --> loss: 0.1201 - sparse_categorical_accuracy: 0.9638
**Epoch 3/5 --> loss: 0.0927 - sparse_categorical_accuracy: 0.9712
**Epoch 4/5 --> loss: 0.0778 - sparse_categorical_accuracy: 0.9763
**Epoch 5/5 --> loss: 0.0690 - sparse_categorical_accuracy: 0.9791

Afterwards, the first model was evaluated using the model.evaluate function leveraging the X_test_scaled and y_test datasets. That in turn yielded a loss of: 0.05009949207305908 and an accuracy of 0.984099984169006. Continuing on, predictions were made using the model.predict() function while viewing the shape as well. After this the labels of the predicted variable (y_pred_labels) were aligned.

Using model.fit_generator, the second model was run just like the first model and used the same parameters. The five Epochs yielded these results.
Epoch 1/5 --> loss: 1.5474 - sparse_categorical_accuracy: 0.4643 
Epoch 2/5 --> loss: 0.8297 - sparse_categorical_accuracy: 0.7334
Epoch 3/5 --> loss: 0.6564 - sparse_categorical_accuracy: 0.7912
Epoch 4/5 --> loss: 0.5595 - sparse_categorical_accuracy: 0.8240
Epoch 5/5 --> loss: 0.5162 - sparse_categorical_accuracy: 0.8395
**Evaluation**
The augmented images did NOT seem to help the model. The model lost accuracy from 0.9791 to become 0.8395. Also the loss was smaller 0.0569 compared to the increased loss of 0.5162 when the images were augmented. This means more model tuning was necessary.
With the help of TensorBoard results yielding various graphs and charts and scalars, the models were further evaluated.



**Results**
The augmented images did NOT seem to help the model. The model lost accuracy from 0.9825 to become 0.8395. Also the loss was smaller 0.0569 compared to the increased loss of 0.5162 when the images were augmented. This means more model tuning was necessary.

**Usage**
In loading the data and splitting the data, these snippets of python code were used:
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

In making predictions, these snippets of python code were used.
y_pred_prob = model.predict(X_test_scaled)
y_pred_prob.shape
np.argmax(y_pred_prob[0]) == y_test[0] == 7
y_pred_labels = np.zeros(len(y_pred_prob), dtype=int)
for i in range(len(y_pred_prob)):
  y_pred_labels[i] = np.argmax(y_pred_prob[i])

print (y_pred_labels[:31])
print ('-' * 60)
print (y_test[:31])


**Dependencies**
Developed/run in Google colab notebook
Dependencies can be found and installed with requirements.txt


**References**
Several web searches were conducted and online articles reviewed to supplement knowledge gained in training to accomplish this project.

**Contributors**
Afia Owusu-Forfie and Martin Moreno

**Advisor**
Arvind Krishnan

**Instructor**
Armando Galeana



