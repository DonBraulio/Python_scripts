"""
Solve Homework#1 from the course "Learning from data" (edX, Caltech)
Implement a perceptron to distinguish between two classes of linearly separable points, in the [-1, 1]x[-1, 1] square.
Generate random points in the square and separate them with a random line that divides the square in two parts.
Use the PLA (Perceptron Learning Algorithm) to distinguish between the two classes of points.
"""
from random import uniform, seed
import numpy as np
import matplotlib.pyplot as plt
from time import time


# Generate a random point in the square [-1, 1]x[-1, 1]
def get_random_point():
    return uniform(-1, 1), uniform(-1, 1)


def get_random_line():
    # Initialize pseudo random generator with timestamp microseconds
    seed(time())

    # Pick two random points to generate a line that separates the space [-1, 1]x[-1, 1] in two sets
    p1 = (0, 0)  # random_point() # Use (0, 0) as one of the points if equal areas are preferred
    p2 = get_random_point()
    # Convert to y = a*x + b
    a = (p1[1] - p2[1]) / (p1[0] - p2[0])
    b = p1[1] - a*p1[0]

    # Return line equation
    return lambda x: a * x + b


def init_perceptron(N):
    # Re-initialize random generator to avoid uniformity with the 2 line points
    seed(time()+1)
    X = [get_random_point() for i in range(N)]
    # Weights vector [w0, w1, w2] -> Note that w0 is the threshold
    W = np.array([0.0, 0.0, 0.0])
    return X, W


# Target function is: True if the point is below separator_line, False if above
def target_function(point, separator_line): return point[1] < separator_line(point[0])


# Perceptron Output = sum( w0 + w1*x + w2*y ) . Note that w0 is the threshold.
# When the W vector is calculated correctly, perceptron_output shall be equal to target_function.
def perceptron_output(point, W):
    return np.sum(W * ((1,) + point)) > 0  # [1] is the threshold's artificial coordinate


# This line was calculated clearing out the "y" from the perceptron equation
def perceptron_line(x, W):
    return (-W[0] - W[1]*x)/W[2]


def plot_perceptron(X, W, misclassified_points, separator_line):
    plt.figure()
    x = np.array([-1, 1])
    plt.plot(x, separator_line(x), '-r', label='Target line')
    plt.plot(x, perceptron_line(x, W), '-g', label='Perceptron line')
    plt.legend()
    for x, y in X:
        if (x, y) in misclassified_points:
            style = 'xk'
        elif perceptron_output((x, y), W):
            style = 'or'
        else:
            style = 'ob'
        plt.plot(x, y, style)
    plt.gca().set_ylim([-1, 1])
    # Uncomment to compare the perceptron line equation with perceptron output of many random points
    # X_test = [random_point() for i in range(500)]
    # for x, y in X_test:
    #     if perceptron_output((x, y)):
    #         style = 'xr'
    #     else:
    #         style = 'xb'
    #     plt.plot(x, y, style)
    plt.show()


# Approximate numerically P[target(x) != perceptron(x)]. The greater N_test (number of points), the better accuracy.
def calculate_error_probability(N_test, W, separator_line):
    N_errors = 0.0
    X_test = [get_random_point() for i in range(N_test)]
    for x, y in X_test:
        if perceptron_output((x, y), W) != target_function((x, y), separator_line):
            N_errors += 1
    return N_errors / N_test


if __name__ == '__main__':
    N = 100  # number of sample points

    # Run the whole algorithm several times to get an average measure of P_error and number of iterations.
    total_P_error = 0.0
    total_iterations = 0.0
    n_attempts = 3
    for attempt in range(n_attempts):
        # Implementation of the PLA algorithm
        X, W = init_perceptron(N)
        misclassified_points = set(X)  # All points are misclassified at first
        separator_line = get_random_line()
        i = 0
        while len(misclassified_points) > 0 and i < 1000:
            # Pick a misclassified point
            point = misclassified_points.pop()
            # Update weights vector
            W += np.array(((1,) + point))*(1 if target_function(point, separator_line) else -1)
            # Update misclassified points
            for point in X:
                if target_function(point, separator_line) != perceptron_output(point, W):
                    misclassified_points.add(point)
            # Uncomment to plot evolution of PLA
            # if i % 10 == 0:
            #    plot_perceptron()
            i += 1

        # Show plot for the last attempt only
        if attempt == n_attempts - 1:
            plot_perceptron(X, W, misclassified_points, separator_line)

        print("Iterations: %d | Misclassified points: %d" % (i, len(misclassified_points)))

        # Calculate error probability
        P_error = calculate_error_probability(5000, W, separator_line)
        print("Error probability: P[target(x) != perceptron(x)] = %.4f" % P_error)

        total_P_error += P_error
        total_iterations += i

    print("Number of PLA runs: %d | Average number of iterations: %.2f | Average P_error: %.4f" %
          (n_attempts, total_iterations / n_attempts, total_P_error / n_attempts))
