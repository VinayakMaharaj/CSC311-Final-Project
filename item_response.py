from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    log_lklihood = 0.0
    epsilon = 1e-10  # to prevent a log error

    for n in range(len(data["user_id"])):
        i, j, c = data["user_id"][n], data["question_id"][n], data["is_correct"][n]
        sig = sigmoid(theta[i] - beta[j])
        log_lklihood += c * np.log(sig + epsilon) + (1 - c) * np.log(1 - sig + epsilon)

    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    # gradient for parameters
    d_theta = np.zeros(len(theta))
    d_beta = np.zeros(len(beta))

    # determining the gradient
    for n in range(len(data["user_id"])):
        i, j, c = data["user_id"][n], data["question_id"][n], data["is_correct"][n]
        sig = sigmoid(theta[i] - beta[j])
        d_theta[i] -= (c - sig)
        d_beta[j] -= (sig - c)

    # performing the update
    theta -= lr * d_theta
    beta -= lr * d_beta

    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # initialize theta and beta
    theta = np.zeros(len(set(data["user_id"])))
    beta = np.zeros(len(set(data["question_id"])))

    val_acc_lst = []
    tr_lld = []
    val_lld = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        tr_lld.append(-1 * neg_lld)
        val_lld.append(-1 * neg_log_likelihood(data=val_data, theta=theta, beta=beta))
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, tr_lld, val_lld


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # Hyperparameters
    lr = 0.002
    iterations = 50

    # Initialize theta and beta
    theta, beta, val_acc_lst, tr_lld, val_lld = irt(train_data, val_data, lr, iterations)

    # Plotting the log-likelihoods
    plt.figure(figsize=(10, 5))
    plt.plot(range(iterations), tr_lld, label='Training Log-Likelihood')
    plt.plot(range(iterations), val_lld, label='Validation Log-Likelihood')
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log-Likelihood')
    plt.title('Training and Validation Log-Likelihood vs. Iterations')
    plt.legend()
    plt.show()

    # Final evaluation
    val_acc = evaluate(val_data, theta, beta)
    test_acc = evaluate(test_data, theta, beta)
    print(f"Final validation accuracy: {val_acc}")
    print(f"Final test accuracy: {test_acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()

