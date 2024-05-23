# Data_mining

This is my python code for Facial recognition based on CK+

First, data augmentation methods are used to prepare the training and testing data, where the training data is enhanced through random cropping and horizontal flipping, while the testing data employs 10-crop augmentation. Then, the loss function is defined as cross-entropy loss (nn.CrossEntropyLoss), and the optimizer is set as stochastic gradient descent (optim.SGD) with momentum.

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

During training, the learning rate is dynamically adjusted, forward propagation is performed to compute the loss, followed by backpropagation to update the model parameters, while recording the training loss and accuracy. In the testing phase, the inputs are reshaped to fit the 10-crop augmentation, forward propagation is conducted to calculate the loss and average predictions, and the testing loss and accuracy are recorded. Finally, the best-performing model is saved.
