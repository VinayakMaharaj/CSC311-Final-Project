# Machine Learning Final Project

## Overview

This repository contains the final project for the Machine Learning course, where we explore various machine learning algorithms, with a focus on **Neural Networks** for question 3, and **Item Response Theory (IRT) Models** with modifications for question 1. The IRT model has been enhanced by adding a new parameter to account for student effort, leading to improved performance in predictive tasks.

## Key Files

- **final_report.pdf**: The main project report detailing the models and their performance.
- **item_response.py**: The original IRT model script.
- **new_item_response.py**: The modified IRT model that includes a new parameter αi to account for varying levels of student effort. This modification helped improve model accuracy and log-likelihood.
- **neural_network.py**: Implementation of the neural network for question 3.
- **ensemble.py**, **knn.py**, **matrix_factorization.py**, **majority_vote.py**: Other machine learning algorithms explored as part of the project.

## How to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/VinayakMaharaj/CSC311-Final-Project

Install the necessary dependencies:

```bash

pip install -r requirements.txt
```
Run the Neural Network model for question 3:

```bash
python neural_network.py
```
To explore other models, use:

For k-NN:
```bash
python knn.py
```

For ensemble methods:
```bash
python ensemble.py
```

For matrix factorization:
```bash
python matrix_factorization.py
```

Results
The results of the neural network and other models can be found in the generated output files and graphs, specifically:

final project q3 graph.png and final project q3 graph 2.png: Visualizations showing the model’s performance.
final_report.pdf: Contains detailed analysis and discussion of the results.

Conclusion
This project covers multiple machine learning algorithms, with a special focus on the neural network model for question 3 and the modified IRT model for question 1. The addition of the student effort parameter in the IRT model has provided a significant improvement in the accuracy and log-likelihood compared to the baseline model. Further improvements and experimentation with regularization and parameter tuning can enhance model performance even more.
