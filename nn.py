import numpy as np
import random
from scipy.special import expit as activation_function
from scipy.stats import truncnorm
from sklearn.metrics import classification_report, accuracy_score


def tnd(mean: float, standard_deviation: float, lower_bound: float, upp: float):
    return truncnorm(
        (lower_bound - mean) / standard_deviation,
        (upp - mean) / standard_deviation, loc=mean,
        scale=standard_deviation
    )


class MLP:
    def __init__(self, input_nodes_count: int, output_nodes_count: int, hidden_nodes_count: int, learning_rate: float):
        self.input_nodes_count = input_nodes_count
        self.output_nodes_count = output_nodes_count
        self.no_of_hidden_nodes = hidden_nodes_count
        self.learning_rate = learning_rate
        input_weights_count = self.input_nodes_count * self.no_of_hidden_nodes

        distribution = tnd(0, 1, -1, 1)
        self.weights_input = distribution.rvs(input_weights_count).reshape(
            (self.no_of_hidden_nodes, self.input_nodes_count))
        output_weights_count = self.no_of_hidden_nodes * self.output_nodes_count

        distribution = tnd(0, 1, -1, 1)
        self.weights_output = distribution.rvs(output_weights_count).reshape(
            (self.output_nodes_count, self.no_of_hidden_nodes))

    def dropout_weight_matrices(self, active_input_percentage: float, active_hidden_percentage: float):
        self.weights_input_save = self.weights_input.copy()
        self.weights_output_save = self.weights_output.copy()
        self.input_nodes_save = self.input_nodes_count
        self.hidden_nodes_save = self.no_of_hidden_nodes

        active_input_nodes = int(self.input_nodes_count * active_input_percentage)
        active_input_indices = sorted(random.sample(range(self.input_nodes_count), active_input_nodes))
        active_hidden_nodes = int(self.no_of_hidden_nodes * active_hidden_percentage)
        active_hidden_indices = sorted(random.sample(range(self.no_of_hidden_nodes), active_hidden_nodes))

        self.weights_input = self.weights_input[:, active_input_indices][active_hidden_indices]
        self.weights_output = self.weights_output[:, active_hidden_indices]

        self.no_of_hidden_nodes = active_hidden_nodes
        self.input_nodes_count = active_input_nodes
        return active_input_indices, active_hidden_indices

    def weight_matrices_reset(self, active_input_indices, active_hidden_indices):
        temp = self.weights_input_save.copy()[:, active_input_indices]
        temp[active_hidden_indices] = self.weights_input
        self.weights_input_save[:, active_input_indices] = temp
        self.weights_input = self.weights_input_save.copy()

        self.weights_output_save[:, active_hidden_indices] = self.weights_output
        self.weights_output = self.weights_output_save.copy()
        self.input_nodes_count = self.input_nodes_save
        self.no_of_hidden_nodes = self.hidden_nodes_save

    def train_single(self, input_vector, target_vector):
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.weights_input, input_vector)
        output_vector_hidden = activation_function(output_vector1)

        output_vector2 = np.dot(self.weights_output, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)

        output_errors = target_vector - output_vector_network
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_output += tmp

        hidden_errors = np.dot(self.weights_output.T, output_errors)
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        x = np.dot(tmp, input_vector.T)
        self.weights_input += self.learning_rate * x

    def train(self, data_array,
              labels_one_hot_array,
              iterations=1,
              active_input_percentage=0.70,
              active_hidden_percentage=0.70,
              no_of_dropout_tests=10):

        partition_length = int(len(data_array) / no_of_dropout_tests)

        for epoch in range(iterations):
            for start in range(0, len(data_array), partition_length):
                indices = self.dropout_weight_matrices(active_input_percentage, active_hidden_percentage)
                active_in_indices, active_hidden_indices = indices
                for i in range(start, start + partition_length):
                    self.train_single(data_array[i][active_in_indices], labels_one_hot_array[i])

                self.weight_matrices_reset(active_in_indices, active_hidden_indices)

    def predict(self, input_vector):
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.weights_input, input_vector)
        output_vector = activation_function(output_vector)

        output_vector = np.dot(self.weights_output, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector

    def get_classification_metrics(self, test_images, test_labels):
        predictions = [output_function(self.predict(img)) for img in test_images]
        labels = [x[0] for x in test_labels]

        metrics_report = classification_report(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        return metrics_report, accuracy


def output_function(predictions) -> int:
    return predictions.argmax()
