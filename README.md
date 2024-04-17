# BAH_TechEx_Guided_Project_Phase2
This is a deliverable for the Guided Project Phase 2 during the BAH Technology Excellence Program

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

Using model.fit_generator, the second model was run as follows.
[Describe the training process here. Include details such as the optimizer used, learning rate, batch size, number of epochs, any data augmentation techniques applied, etc.]

**Evaluation**
With the help of TensorBoard results yielding various graphs and charts and scalars, the models were evaluated.
[Describe how to evaluate the performance of the CNN model. Include metrics used for evaluation (e.g., accuracy, precision, recall), any validation techniques applied (e.g., k-fold cross-validation), etc.]

**Results**
[Present the results of the trained model. Include metrics such as accuracy, loss, and any other relevant metrics. If applicable, provide visualizations (e.g., confusion matrix, learning curve) to further analyze the performance of the model.]

Usage
[Provide instructions on how to use the trained model for inference. Include code snippets or examples demonstrating how to load the model and perform predictions on new data.]

Dependencies
[List any dependencies required to run the code (e.g., Python libraries, frameworks) and how to install them.]

References
[Include any references to papers, articles, or other resources that inspired or were used in the development of the CNN architecture.]

**Contributors**
Afia Owusu-Forfie and Martin Moreno

**Advisor**
Arvind Krishnan

**Instructor***
Armando Galeana

License
[Specify the license under which the code is distributed.]


