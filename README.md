# Data_mining

This is my python code for Data Mining Project.

## Text recognition (train.py)

First, the dataset is split into training and testing sets, and the text data is transformed using TF-IDF vectorization and standardized. Next, a logistic regression model is trained on the training data and used to make predictions on the test data, generating a classification report to evaluate the model's performance. Subsequently, a random forest model is trained and evaluated in a similar manner to compare the performance of different models. To address class imbalance in the dataset, the SMOTE technique is applied to the training data for oversampling, and the logistic regression model is retrained and re-evaluated. Finally, the code inputs new text into the model, processes it with the same vectorization and standardization steps, and uses the trained logistic regression model to make predictions, outputting the predicted results to demonstrate the model's effectiveness in real-world applications.

## Facial recognition based on Vggnet and resnet using CK+ datasets

First, data augmentation methods are used to prepare the training and testing data, where the training data is enhanced through random cropping and horizontal flipping, while the testing data employs 10-crop augmentation. Then, the loss function is defined as cross-entropy loss (nn.CrossEntropyLoss), and the optimizer is set as stochastic gradient descent (optim.SGD) with momentum.

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

During training, the learning rate is dynamically adjusted, forward propagation is performed to compute the loss, followed by backpropagation to update the model parameters, while recording the training loss and accuracy. In the testing phase, the inputs are reshaped to fit the 10-crop augmentation, forward propagation is conducted to calculate the loss and average predictions, and the testing loss and accuracy are recorded. Finally, the best-performing model is saved.
