# BAH_TechEx_Guided_Project_Phase2
This is a deliverable for the Guided Project Phase 2 during the BAH Technology Excellence Program

**Overview**
This repository contains the implementation of a Convolutional Neural Network (CNN) architecture for image classification. This README provides the results of running a baseline model, an improved model and a best model post adjustment using the parameters from a Hyper Parameterization exercise.

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
[If applicable, describe the dataset used for training and evaluation. Include details such as the size of the dataset, number of classes, preprocessing steps, etc.]

Training
[Describe the training process here. Include details such as the optimizer used, learning rate, batch size, number of epochs, any data augmentation techniques applied, etc.]

Evaluation
[Describe how to evaluate the performance of the CNN model. Include metrics used for evaluation (e.g., accuracy, precision, recall), any validation techniques applied (e.g., k-fold cross-validation), etc.]

Results
[Present the results of the trained model. Include metrics such as accuracy, loss, and any other relevant metrics. If applicable, provide visualizations (e.g., confusion matrix, learning curve) to further analyze the performance of the model.]

Usage
[Provide instructions on how to use the trained model for inference. Include code snippets or examples demonstrating how to load the model and perform predictions on new data.]

Dependencies
[List any dependencies required to run the code (e.g., Python libraries, frameworks) and how to install them.]

References
[Include any references to papers, articles, or other resources that inspired or were used in the development of the CNN architecture.]

Contributors
[List the contributors to this project, along with their respective roles.]

License
[Specify the license under which the code is distributed.]


