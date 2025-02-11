# Notes

## New Concepts Forum 1

**MLP**

MLP stands for "Multilayer Perceptron." It is a class of feedforward artificial neural network (ANN) that consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Each node, except for the input nodes, is a neuron that uses a nonlinear activation function. MLPs are widely used for solving problems that require supervised learning.


**Data Augmentation**

Data augmentation is a technique used to increase the diversity of your training data without actually collecting new data. It involves creating modified versions of the existing data by applying various transformations such as rotation, translation, scaling, flipping, and adding noise. This helps improve the robustness and generalization ability of machine learning models, especially in tasks like image classification, where having a large and varied dataset is crucial.


## Activities for the Assignment
### Generate a Multilayer Perceptron (MLP) for MNIST
- Run the example and obtain results

'''
# Hyperparameters

num_classes = 10  # this is the number of digits
num_epochs = 5
batch_size = 100
learning_rate = 0.001
hidden_size = 16
'''

Test accuracy: 89.56%

![alt text](image.png)


- Design the architecture (Increase the hidden_layer size)
- Train the model modify the number of epochs for training
- Evaluate the model on a validation set
- Report accuracy and discuss any challenges encountered

'''
# Hyperparameters only modifying the hidden_size

num_classes = 10  # this is the number of digits
num_epochs = 5
batch_size = 100
learning_rate = 0.001
hidden_size = 32
'''

Test accuracy: 90.44%

![alt text](image-1.png)


'''
# Hyperparameters doubling the hidden_size + epochs

num_classes = 10  # this is the number of digits
num_epochs = 10
batch_size = 100
learning_rate = 0.001
hidden_size = 32
'''

Test accuracy: 90.37% >> worse accuracy

'''
# Hyperparameters doubling the hidden_size + epochs

num_classes = 10  # this is the number of digits
num_epochs = 30
batch_size = 100
learning_rate = 0.001
hidden_size = 32
'''

Test accuracy: 90.41% >> worse accuracy


Setting num_epochs too low might result in underfitting, where the model does not learn enough from the data. Conversely, setting it too high might lead to overfitting, where the model learns the training data too well, including its noise, and performs poorly on unseen data. Therefore, choosing an appropriate value for num_epochs is essential for achieving a good balance between underfitting and overfitting.


![alt text](image-2.png)


**Tested with 2 hidden layers and the accuracy dropped to Test accuracy: 30.98%**

**Tested with 50 hidden layers and the accuracy went up Test accuracy: 90.71%**

![alt text](image-3.png)


**Tested with 250 hidden layers and num_epochs = 200 the accuracy went up Test accuracy: 91.19%**


### Questions to Answer

1. How long have you trained the network?
2. What accuracy do you obtain with this program?
3. What do you see analyzing the confusion matrix?
4. Do you think the program is overfitting by looking at the Loss/Accuracy plots?


---

Analyzing the confusion matrix can provide insights into the performance of your classification model. Here are some key points to consider:

1. **True Positives (TP)**: The number of correct predictions for each class.
2. **False Positives (FP)**: The number of incorrect predictions where the model predicted a class that was not the true class.
3. **False Negatives (FN)**: The number of incorrect predictions where the model failed to predict the true class.
4. **True Negatives (TN)**: The number of correct predictions where the model correctly identified that a sample does not belong to a specific class.

By examining these values, you can determine:
- **Which classes are being confused with each other**: High values in off-diagonal cells indicate that the model is confusing one class with another.
- **Class imbalance issues**: If certain classes have significantly more samples than others, it might affect the model's performance.
- **Model's strengths and weaknesses**: Identify which classes the model predicts well and which it struggles with.

For example, if the confusion matrix shows that the model frequently misclassifies digit '3' as digit '8', you might need to investigate why these digits are being confused and consider ways to improve the model's ability to distinguish between them.

Would you like to see an example of a confusion matrix and how to interpret it?

