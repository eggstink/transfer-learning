# Reinforcement Learning with Drone vs Bird Dataset

This repository contains a project that utilizes reinforcement learning to distinguish between drones and birds using the [Drone vs Bird dataset](https://www.kaggle.com/datasets/muhammadsaoodsarwar/drone-vs-bird) from Kaggle. We employ the Xception model from Keras to perform this classification task.

## Dataset

The dataset used in this project is the Drone vs Bird dataset from Kaggle. It contains images of drones and birds that are used to train and evaluate the model.

- **Dataset URL**: [Drone vs Bird Dataset](https://www.kaggle.com/datasets/muhammadsaoodsarwar/drone-vs-bird)

## Model

We use the Xception model from Keras for our classification task. Xception is a deep convolutional neural network architecture that involves depthwise separable convolutions and has been shown to achieve better performance compared to other models in image classification tasks.

### Xception Model

The Xception model is initialized with pre-trained weights from ImageNet and fine-tuned on the Drone vs Bird dataset. The architecture of the model is as follows:

- **Input Layer**: Takes input images of size 299x299x3(but used 224x224x3 in the notebook).
- **Base Model**: Xception model with pre-trained weights.
- **Output Layer**: A dense layer with a softmax activation function for binary classification (Drone vs Bird).

## Requirements

The project was ran using the following dependencies:

- Python 3.12.6
- TensorFlow
- Keras
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
