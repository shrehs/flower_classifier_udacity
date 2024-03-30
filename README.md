# flower_classifier_udacity
# Flower Classification Deep Learning Project

In this project, our objective is to develop a flower classification deep learning network using transfer learning. Initiated within Udacity's GPU-enabled workspace, the project leverages pre-trained classifiers from the PyTorch package. However, due to workspace constraints, the source files are not provided. Nevertheless, our primary goal is to customize the classifier attribute of each imported model to suit the requirements of our flower classification task.

## Project Breakdown

The project is structured into several key components:

1. **Creating the Datasets**: Importing images provided by Udacity, applying appropriate transformations, and splitting them into training, validation, and testing datasets.

2. **Creating the Architecture**: Utilizing pre-trained models from PyTorch's torchvision package to establish different classifier parameters suitable for our datasets. This involves defining an NLL Loss criterion and an Adam optimizer.

3. **Training the Model**: Leveraging PyTorch and Udacity's GPU-enabled platform to train our model on the training and validation datasets, aiming to create an optimal flower classification model.

4. **Saving / Loading the Model**: Exporting the trained model to a 'checkpoint.pth' file for future use and demonstrating reloading/rebuilding the model in another file.

5. **Class Prediction**: Using the trained model to predict the class of a flower given a testing input image.

## Files Included

Here's a brief overview of the files included in the project:

- **Image Classifier Project.ipynb**: This Jupyter notebook contains all project activities, including more than what's covered in the predict.py and train.py files.

- **Image Classifier Project.html**: Similar to the above notebook but in HTML format.

- **train.py**: This file accepts inputs from the command line prompt and handles tasks such as dataset creation, architecture setup, model training, and model saving.

- **predict.py**: This file accepts inputs from the command line prompt and handles tasks related to loading the model and making predictions on new images.

## Package Imports

All necessary packages and modules are imported in the first cell of the notebook. These include:

- Data augmentation modules from torchvision.transforms.
- DataLoader and ImageFolder from torchvision.
- Pretrained models such as VGG16 from torchvision.models.
- Necessary libraries for training, validation, and testing processes.

## Command Line Application

This part of the project focuses on creating a command-line interface for training and predicting with the model. Key specifications include:

- **Training a Network**: The train.py script trains a new network on a dataset of images and saves the model to a checkpoint.

- **Training Validation Log**: Prints out the training loss, validation loss, and validation accuracy as the network trains.

- **Model Architecture**: Users can choose from at least two different architectures available from torchvision.models.

- **Model Hyperparameters**: Users can set hyperparameters for learning rate, number of hidden units, and training epochs.

- **Training with GPU**: Users can choose to train the model on a GPU if available.

- **Predicting Classes**: The predict.py script reads in an image and a checkpoint, then prints the most likely image class and its associated probability.

- **Top K Classes**: Allows users to print out the top K classes along with associated probabilities.

- **Displaying Class Names**: Users can load a JSON file that maps the class values to other category names.

- **Predicting with GPU**: Users can utilize the GPU to calculate predictions.

## Conclusion

This README provides an overview of the Flower Classification Deep Learning Project, outlining its objectives, project breakdown, included files, and specifications for training and prediction using command-line interfaces. For further details, please refer to the provided files and documentation.
