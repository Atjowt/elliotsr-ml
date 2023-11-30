#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "MNIST_for_C/mnist.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 10
#define BATCH_SIZE 64

/// Returns a random value between 0.0 and 1.0 inclusive.
static inline double nrand(void) {
    return (double)rand() / (double)RAND_MAX;
}

// Returns the maximum of a and b.
static inline double max(double a, double b) {
    return a > b ? a : b;
}

// Initializes weights and biases.
void initialize_parameters(double weights_ih[HIDDEN_SIZE][INPUT_SIZE], double biases_ih[HIDDEN_SIZE],
                           double weights_ho[OUTPUT_SIZE][HIDDEN_SIZE], double biases_ho[OUTPUT_SIZE]) {
                            
    // Initialize weights and biases with random values
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        biases_ih[h] = nrand() - 0.5;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            weights_ih[h][i] = nrand() - 0.5;
        }
    }

    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        biases_ho[o] = nrand() - 0.5;
        for (int h = 0; h < HIDDEN_SIZE; ++h) {
            weights_ho[o][h] = nrand() - 0.5;
        }
    }
}

// ReLU activation function
double reLU(double x) {
    return max(0, x);
}

// Derivative of the ReLU function
double reLU_derivative(double x) {
    return x < 0 ? 0 : 1;
}

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    double y = sigmoid(x);
    return y * (1.0 - y);
}

// Forward pass
void forward_pass(double input[INPUT_SIZE], double weights_ih[HIDDEN_SIZE][INPUT_SIZE], double biases_ih[HIDDEN_SIZE],
                  double hidden[HIDDEN_SIZE], double weights_ho[OUTPUT_SIZE][HIDDEN_SIZE], double biases_ho[OUTPUT_SIZE],
                  double output[OUTPUT_SIZE]) {

    // Compute hidden layer activations
    for (int h = 0; h < HIDDEN_SIZE; ++h) {
        // Compute weighted sum
        hidden[h] = 0;
        for (int j = 0; j < INPUT_SIZE; ++j) {
            hidden[h] += input[j] * weights_ih[h][j];
        }
        hidden[h] += biases_ih[h]; // Add bias
        hidden[h] = reLU(hidden[h]); // Pass through activation function
    }

    // Compute output layer activations
    for (int o = 0; o < OUTPUT_SIZE; ++o) {
        // Compute weighted sum
        output[o] = 0;
        for (int h = 0; h < HIDDEN_SIZE; ++h) {
            output[o] += hidden[h] * weights_ho[o][h];
        }
        output[o] += biases_ho[o]; // Add bias
        output[o] = reLU(output[o]); // Pass through activation function
    }
}

// Backward pass
void backward_pass(double input[INPUT_SIZE], double hidden[HIDDEN_SIZE], double output[OUTPUT_SIZE],
                    double target[OUTPUT_SIZE], double weights_ih[HIDDEN_SIZE][INPUT_SIZE],
                    double biases_ih[HIDDEN_SIZE], double weights_ho[OUTPUT_SIZE][HIDDEN_SIZE],
                    double biases_ho[OUTPUT_SIZE]) {

    // Compute output layer gradients
    double output_gradients[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        output_gradients[i] = (output[i] - target[i]) * reLU_derivative(output[i]);
    }

    // Update output layer parameters
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        biases_ho[i] -= LEARNING_RATE * output_gradients[i];
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            weights_ho[i][j] -= LEARNING_RATE * output_gradients[i] * hidden[j];
        }
    }

    // Compute hidden layer gradients
    double hidden_gradients[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        hidden_gradients[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            hidden_gradients[i] += output_gradients[j] * weights_ho[j][i];
        }
        hidden_gradients[i] *= reLU_derivative(hidden[i]);
    }

    // Update hidden layer parameters
    for (int i = 0; i < HIDDEN_SIZE; ++i) {
        biases_ih[i] -= LEARNING_RATE * hidden_gradients[i];
        for (int j = 0; j < INPUT_SIZE; ++j) {
            weights_ih[i][j] -= LEARNING_RATE * hidden_gradients[i] * input[j];
        }
    }
}

// Cross-entropy loss
double calculate_cost(double output[OUTPUT_SIZE], double target[OUTPUT_SIZE]) {
    double cost = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        cost += target[i] * log(output[i] + 1e-10); // Add small epsilon to avoid log(0)
    }
    return -cost;
}


int main() {

    // Load MNIST dataset
    load_mnist();
    
    // Initialize random seed
    srand(69420);

    double weights_ih[HIDDEN_SIZE][INPUT_SIZE];
    double biases_ih[HIDDEN_SIZE];
    double weights_ho[OUTPUT_SIZE][HIDDEN_SIZE];
    double biases_ho[OUTPUT_SIZE];

    // Initialize parameters
    initialize_parameters(weights_ih, biases_ih, weights_ho, biases_ho);

    double input[INPUT_SIZE];
    double target[OUTPUT_SIZE];
    double output[OUTPUT_SIZE];
    double hidden[HIDDEN_SIZE];

    // Train for a set number of epochs
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {

        for (int b = 0; b < NUM_TRAIN; b += BATCH_SIZE) {

            // Forward and backward pass for each digit sample
            double cost = 0.0;
            for (int i = 0; i < BATCH_SIZE; ++i) {
                
                double* image = train_image[b + i];
                int label = train_label[b + i];

                double target[OUTPUT_SIZE] = { 0 };
                target[label] = 1.0;

                forward_pass(input, weights_ih, biases_ih, hidden, weights_ho, biases_ho, output);
                backward_pass(input, hidden, output, target, weights_ih, biases_ih, weights_ho, biases_ho);
                cost += calculate_cost(output, target);
            }

            int batch = b / BATCH_SIZE;
            printf("Batch: %d, Epoch: %d, Cost: %f, Average: %f\n", batch, epoch, cost, cost / (double)BATCH_SIZE);
        }

    }

    return 0;
}
