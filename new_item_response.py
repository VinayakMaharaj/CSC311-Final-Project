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

def neg_log_likelihood(data, theta, beta, alpha):
    """Compute the negative log-likelihood.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    log_lklihood = 0.0
    epsilon = 1e-10  # to prevent a log error

    for n in range(len(data["user_id"])):
        i, j, c = data["user_id"][n], data["question_id"][n], data["is_correct"][n]
        sig = sigmoid((theta[i] + alpha[i]) - beta[j])
        log_lklihood += c * np.log(sig + epsilon) + (1 - c) * np.log(1 - sig + epsilon)

    return -log_lklihood

def update_theta_beta(data, lr, theta, beta, alpha):
    """Update theta, beta, and alpha using gradient descent.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: tuple of vectors
    """
    # gradient for parameters
    d_theta = np.zeros(len(theta))
    d_beta = np.zeros(len(beta))
    d_alpha = np.zeros(len(alpha))

    # determining the gradient
    for n in range(len(data["user_id"])):
        i, j, c = data["user_id"][n], data["question_id"][n], data["is_correct"][n]
        sig = sigmoid((theta[i] + alpha[i]) - beta[j])
        d_theta[i] -= (c - sig)
        d_beta[j] -= (sig - c)
        d_alpha[i] -= (c - sig)

    # performing the update
    theta -= lr * d_theta
    beta -= lr * d_beta
    alpha -= lr * d_alpha

    return theta, beta, alpha

def irt(data, val_data, lr, iterations):
    """Train IRT model.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, alpha, val_acc_lst, tr_acc_lst)
    """
    # initialize theta, beta, and alpha
    theta = np.zeros(len(set(data["user_id"])))
    beta = np.zeros(len(set(data["question_id"])))
    alpha = np.random.randn(len(set(data["user_id"])))

    val_acc_lst = []
    tr_acc_lst = []
    tr_lld = []
    val_lld = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha)
        tr_lld.append(-1 * neg_lld)
        val_lld.append(-1 * neg_log_likelihood(data=val_data, theta=theta, beta=beta, alpha=alpha))
        tr_acc = evaluate(data=data, theta=theta, beta=beta, alpha=alpha)
        val_acc = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha)
        tr_acc_lst.append(tr_acc)
        val_acc_lst.append(val_acc)
        print("NLLK: {} \t Training Score: {} \t Validation Score: {}".format(neg_lld, tr_acc, val_acc))
        theta, beta, alpha = update_theta_beta(data, lr, theta, beta, alpha)

    return theta, beta, alpha, val_acc_lst, tr_acc_lst, tr_lld, val_lld

def evaluate(data, theta, beta, alpha):
    """Evaluate the model given data and return the accuracy.
    
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = ((theta[u] + alpha[u]) - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])

def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    num_iter = 50
    lr = 0.003
    theta, beta, alpha, val_acc_lst, tr_acc_lst, tr_lld, val_lld = irt(train_data, val_data, lr, num_iter)
    
    plt.figure(figsize=(12, 5))

    # Plot log-likelihoods
    plt.subplot(1, 2, 1)
    x = np.arange(num_iter)
    plt.plot(x, tr_lld, label='Training Log-Likelihood')
    plt.plot(x, val_lld, label='Validation Log-Likelihood')
    plt.xlabel('Iterations')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    plt.title('Log-Likelihoods as a Function of Iteration')

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(x, tr_acc_lst, label='Training Accuracy')
    plt.plot(x, val_acc_lst, label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracies as a Function of Iteration')

    plt.tight_layout()
    plt.show()

    val_acc = evaluate(val_data, theta, beta, alpha)
    test_acc = evaluate(test_data, theta, beta, alpha)
    training_acc = evaluate(train_data, theta=theta, beta=beta, alpha=alpha)
    # print(f"Validation Accuracy: {val_acc:.4f} \nTest Accuracy: {test_acc:.4f}")
    print("Validation Accuracy: {} \n Test Accuracy: {} \n Training accuracy: {}".format(val_acc, test_acc, training_acc))

    # Plot probability curves for three questions
    questions = np.random.choice(1774, 3, replace=False)
    theta_range = np.linspace(min(theta), max(theta), 100)
    for q in questions:
        probs = sigmoid((theta_range[:, None] + alpha.mean()) - beta[q])
        plt.plot(theta_range, probs, label=f"Question {q} with beta {beta[q]:.2f}")
    plt.xlabel('Theta')
    plt.ylabel('Probability of Correct Response')
    plt.legend()
    plt.title('Probability Curves for Selected Questions')
    plt.show()

if __name__ == "__main__":
    main()