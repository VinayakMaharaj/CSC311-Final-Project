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
        out = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(out))
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
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0)
            
            # Add L2 regularization term
            regularization = (model.get_weight_norm() / 2) * lamb
            loss += regularization

            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t " "Valid Acc: {}".format(
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

    # # Set optimization hyperparameters.
    # lr = None
    # num_epoch = None
    # lamb = None

    # train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    # # Next, evaluate your network on validation/test data

#####################################################################

    # # Set model hyperparameters.
    # latent_dims = [10, 50, 100, 200, 500]
    # best_k = None
    # best_model = None
    # best_valid_acc = 0

    # # Set optimization hyperparameters.
    # lr = 0.01
    # num_epoch = 50
    # lamb = 0.001

    # for k in latent_dims:
    #     print(f"Training with latent dimension k={k}")
    #     model = AutoEncoder(train_matrix.shape[1], k)
    #     train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
    #     valid_acc = evaluate(model, zero_train_matrix, valid_data)
    #     print(f"Validation Accuracy for k={k}: {valid_acc:.4f}")

    #     if valid_acc > best_valid_acc:
    #         best_valid_acc = valid_acc
    #         best_k = k
    #         best_model = model

    # print(f"Best latent dimension k*: {best_k} with validation accuracy: {best_valid_acc:.4f}")

    # # Next, evaluate your network on test data
    # test_acc = evaluate(best_model, zero_train_matrix, test_data)
    # print(f"Test Accuracy for k*={best_k}: {test_acc:.4f}")    

#####################################################################
#   # Set model hyperparameters.
#     k = 50
#     model = AutoEncoder(train_matrix.shape[1], k)

#     # Set optimization hyperparameters.
#     lr = 0.01
#     num_epoch = 50
#     lamb = 0.001

#     # Lists to store training loss and validation accuracy
#     train_losses = []
#     valid_accuracies = []

#     # Define optimizer
#     optimizer = optim.SGD(model.parameters(), lr=lr)
#     num_student = train_matrix.shape[0]

#     # Training loop
#     for epoch in range(num_epoch):
#         train_loss = 0.0
#         model.train()  # Set model to training mode

#         for user_id in range(num_student):
#             inputs = Variable(zero_train_matrix[user_id]).unsqueeze(0)
#             target = inputs.clone()

#             optimizer.zero_grad()
#             output = model(inputs)

#             # Mask the target to only compute the gradient of valid entries.
#             nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
#             target[nan_mask] = output[nan_mask]

#             loss = torch.sum((output - target) ** 2.0)
#             loss.backward()

#             train_loss += loss.item()
#             optimizer.step()

#         train_losses.append(train_loss)
#         valid_acc = evaluate(model, zero_train_matrix, valid_data)
#         valid_accuracies.append(valid_acc)

#         print(f"Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tValidation Accuracy: {valid_acc:.4f}")

#     # Plotting training loss and validation accuracy
#     epochs = range(1, num_epoch + 1)
#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, train_losses, label='Training Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training Loss per Epoch')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, valid_accuracies, label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Validation Accuracy per Epoch')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()

#     # Evaluate on test data
#     test_acc = evaluate(model, zero_train_matrix, test_data)
#     print(f"Test Accuracy for k*={k}: {test_acc:.4f}")

#####################################################################
    # Set model hyperparameters.
    k = 50
    lambdas = [0.001, 0.01, 0.1, 1]
    best_lambda = None
    best_valid_acc = 0
    best_model = None

    best_train_losses = []
    best_valid_accuracies = []

    for lamb in lambdas:
        print(f"Training with lambda={lamb}")
        model = AutoEncoder(train_matrix.shape[1], k)

        # Set optimization hyperparameters.
        lr = 0.01
        num_epoch = 50

        # Lists to store training loss and validation accuracy
        train_losses = []
        valid_accuracies = []

        # Define optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr)
        num_student = train_matrix.shape[0]

        # Training loop
        for epoch in range(num_epoch):
            train_loss = 0.0
            model.train()  # Set model to training mode

            for user_id in range(num_student):
                inputs = Variable(zero_train_matrix[user_id]).unsqueeze(0)
                target = inputs.clone()

                optimizer.zero_grad()
                output = model(inputs)

                # Mask the target to only compute the gradient of valid entries.
                nan_mask = np.isnan(train_matrix[user_id].unsqueeze(0).numpy())
                target[nan_mask] = output[nan_mask]

                loss = torch.sum((output - target) ** 2.0)

                # Add L2 regularization term
                regularization = (model.get_weight_norm() / 2) * lamb
                loss += regularization

                loss.backward()

                train_loss += loss.item()
                optimizer.step()

            train_losses.append(train_loss)
            valid_acc = evaluate(model, zero_train_matrix, valid_data)
            valid_accuracies.append(valid_acc)

            print(f"Epoch: {epoch+1} \tTraining Loss: {train_loss:.6f} \tValidation Accuracy: {valid_acc:.4f}")

        final_valid_acc = valid_accuracies[-1]
        print(f"Final Validation Accuracy for lambda={lamb}: {final_valid_acc:.4f}")

        if final_valid_acc > best_valid_acc:
            best_valid_acc = final_valid_acc
            best_lambda = lamb
            best_model = model
            best_train_losses = train_losses
            best_valid_accuracies = valid_accuracies

    # Plotting training loss and validation accuracy for the best lambda
    epochs = range(1, num_epoch + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, best_train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, best_valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Evaluate the best model on test data
    test_acc = evaluate(best_model, zero_train_matrix, test_data)
    print(f"Best lambda: {best_lambda}")
    print(f"Final Validation Accuracy for best lambda={best_lambda}: {best_valid_acc:.4f}")
    print(f"Test Accuracy for best lambda={best_lambda}: {test_acc:.4f}")



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()