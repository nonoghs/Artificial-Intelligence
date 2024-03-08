import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training=True):
    """
        TODO: implement this function.

        INPUT:
            An optional boolean argument (default value is True for training dataset)

        RETURNS:
            Dataloader for the training set (if training = True) or the test set (if training = False)
        """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.FashionMNIST('./data', train=training, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=training)
    return loader


def build_model():
    """
        TODO: implement this function.

        INPUT:
            None

        RETURNS:
            An untrained neural network model
        """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
        TODO: implement this function.

        INPUT:
            model - the model produced by the previous function
            train_loader  - the train DataLoader produced by the first function
            criterion   - cross-entropy
            T - number of epochs for training

        RETURNS:
            None
        """

    # This line is from tutorial
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Training start
    model.train()
    for epoch in range(T):
        total_loss = 0.0  # Initialize total_loss to 0 at the start of each epoch
        correct = 0
        total = 0
        # Most of codes below in this class are modified from tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
        for data in train_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Accumulate the total loss
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # I asked chatgpt for revise code here, it was running_loss and I forget to divide by total.
        average_loss = total_loss / total  # Calculate the average loss per epoch
        print('Train Epoch: {} Accuracy: {}/{} ({:.2f}%) Loss: {:.3f}'.format(
            epoch, correct, total, 100. * correct / total, average_loss))


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
        TODO: implement this function.

        INPUT:
            model - the the trained model produced by the previous function
            test_loader    - the test DataLoader
            criterion   - cropy-entropy

        RETURNS:
            None
        """

    model.eval()
    singleloss = 0
    correct = 0
    total = len(test_loader.dataset)  # Define total to store the total number of samples
    # The codes below are modified from tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
    # I also ask chatgpt to revise the code for me.
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            singleloss += criterion(outputs, labels).item() * images.size(0)  # Multiply by batch size
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    # Chatgpt helps me find the bug here. Same promblem as in train model.
    average_loss = singleloss / total  # Calculate average loss
    # I use if-else initially, chatgpt helps me revise my code.
    if show_loss:
        print('Average loss: {:.4f}'.format(average_loss))
    print('Accuracy: {:.2f}%'.format(100 * (correct / total)))


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT:
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1

    RETURNS:
        None
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    model.eval()
    image = test_images[index]
    image = image.unsqueeze(0)  # Add batch dimension
    output = model(image)
    prob = F.softmax(output, dim=1)
    top3_prob, top3_label = torch.topk(prob, 3)

    for i in range(3):
        label = class_names[top3_label[0][i].item()]
        probability = top3_prob[0][i].item() * 100
        print('{}: {:.2f}%'.format(label, probability))


if __name__ == '__main__':
    # Define loss criterion
    criterion = nn.CrossEntropyLoss()
    # Get data loaders
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    # Build model
    model = build_model()
    print(model)
    # Start train
    train_model(model, train_loader, criterion, 5)
    # Start evaluate
    evaluate_model(model, test_loader, criterion)

    # Load a small batch of test images
    test_images, _ = next(iter(test_loader))

    # Predict label for a specific image index
    index = 1 # Defined in pdf
    predict_label(model, test_images, index)
