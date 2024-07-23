import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
import matplotlib.pyplot as plt

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        # out = inputs

        g_output = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(g_output))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out

def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # Compute the loss with regularization
            # loss = torch.sum((output - target) ** 2.0) + lamb * model.get_weight_norm()

            loss = torch.sum((output - target) ** 2.0) + (lamb / 2) * model.get_weight_norm()  # Added regularization term
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

            

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t Valid Acc: {}".format(
                epoch, train_loss, valid_acc
            )
        )
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    # k = None
    # model = None

    # ks = [10, 50, 100, 200, 500]
    # best_k = ks[0]
    # best_acc = 0
    # best_model = None

    # # Set optimization hyperparameters.
    # lr = 0.01
    # num_epoch = 50
    # lamb = 0.001

    # # train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    # # Next, evaluate your network on validation/test data
    # for k in ks:
    #     print(f'Training model with k={k}')
    #     model = AutoEncoder(num_question=train_matrix.shape[1], k=k)
        
    #     training_losses = []
    #     validation_accuracies = []

    #     for epoch in range(num_epoch):
    #         model.train()
    #         train_loss = 0.0

    #         for user_id in range(train_matrix.shape[0]):
    #             inputs = Variable(zero_train_matrix[user_id]).unsqueeze(0)
    #             target = inputs.clone()

    #             optimizer = optim.Adam(model.parameters(), lr=lr)
    #             optimizer.zero_grad()
    #             output = model(inputs)

    #             # Mask the target to only compute the gradient of valid entries.
    #             nan_mask = np.isnan(train_matrix[user_id].numpy())
    #             target[0][nan_mask] = output[0][nan_mask]

    #             # Compute the loss with regularization
    #             loss = torch.sum((output - target) ** 2.0) + lamb * model.get_weight_norm()
    #             loss.backward()

    #             train_loss += loss.item()
    #             optimizer.step()

    #         training_losses.append(train_loss)
    #         valid_acc = evaluate(model, zero_train_matrix, valid_data)
    #         validation_accuracies.append(valid_acc)

    #         print(
    #             "Epoch: {} \tTraining Cost: {:.6f}\t Valid Acc: {}".format(
    #                 epoch, train_loss, valid_acc
    #             )
    #         )
        
    #     valid_acc = evaluate(model, zero_train_matrix, valid_data)
    #     print(f'Validation accuracy for k={k}: {valid_acc}')
        
    #     if valid_acc > best_acc:
    #         best_acc = valid_acc
    #         best_k = k
    #         best_model = model
    #         best_train_losses = training_losses
    #         best_valid_accuracies = validation_accuracies
    
    # print(f'Best k: {best_k} with validation accuracy: {best_acc}')

    # # Evaluate on the test data with the best model
    # test_acc = evaluate(best_model, zero_train_matrix, test_data)
    # print(f'Test accuracy with best k={best_k}: {test_acc}')

    # # Plot training losses and validation accuracies
    # epochs = range(num_epoch)
    # plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, best_train_losses, label='Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss vs Epochs')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, best_valid_accuracies, label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Validation Accuracy vs Epochs')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

    ks = [10, 50, 100, 200, 500]
    best_k = ks[0]
    best_acc = 0
    best_model = None

    # Set optimization hyperparameters.
    lr = 0.001
    num_epoch = 50
    lambdas = [0.001, 0.01, 0.1, 1]

    best_train_losses = None
    best_valid_accuracies = None

    for k in ks:
        for lamb in lambdas:
            print(f'Training model with k={k} and lambda={lamb}')
            model = AutoEncoder(num_question=train_matrix.shape[1], k=k)

            training_losses = []
            validation_accuracies = []

            for epoch in range(num_epoch):
                model.train()
                train_loss = 0.0

                for user_id in range(train_matrix.shape[0]):
                    inputs = Variable(zero_train_matrix[user_id]).unsqueeze(0)
                    target = inputs.clone()

                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    optimizer.zero_grad()
                    output = model(inputs)

                    # Mask the target to only compute the gradient of valid entries.
                    nan_mask = np.isnan(train_matrix[user_id].numpy())
                    target[0][nan_mask] = output[0][nan_mask]

                    # Compute the loss with regularization
                    loss = torch.sum((output - target) ** 2.0) + (lamb / 2) * model.get_weight_norm()
                    loss.backward()

                    train_loss += loss.item()
                    optimizer.step()

                training_losses.append(train_loss)
                valid_acc = evaluate(model, zero_train_matrix, valid_data)
                validation_accuracies.append(valid_acc)

                print(
                    "Epoch: {} \tTraining Cost: {:.6f}\t Valid Acc: {}".format(
                        epoch, train_loss, valid_acc
                    )
                )

            valid_acc = evaluate(model, zero_train_matrix, valid_data)
            print(f'Validation accuracy for k={k}, lambda={lamb}: {valid_acc}')

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_k = k
                best_lambda = lamb
                best_model = model
                best_train_losses = training_losses
                best_valid_accuracies = validation_accuracies

    print(f'Best k: {best_k}, best lambda: {best_lambda} with validation accuracy: {best_acc}')

    # Evaluate on the test data with the best model
    test_acc = evaluate(best_model, zero_train_matrix, test_data)
    print(f'Test accuracy with best k={best_k}, best lambda={best_lambda}: {test_acc}')

    # Plot training losses and validation accuracies
    epochs = range(num_epoch)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, best_train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, best_valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

