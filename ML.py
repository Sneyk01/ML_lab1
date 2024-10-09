import numpy as np


def debug(text: str):
    debug_enable = True
    if debug_enable:
        print(f'dbg: {text}')


class NeuralNetwork:
    def __init__(self, layers_num: int, perceptron_count: [int]):
        self.layers = []
        self.layers_num = layers_num

        for layer_idx in range(self.layers_num):
            self.layers.append(perceptron_count[layer_idx])

        # rows - input neurons; cols - output neurons
        self.weight_matrix = []
        self.weight_matrix = [np.matrix([np.random.rand(self.layers[matrix_index + 1])
                                         for _ in range(self.layers[matrix_index])])
                              for matrix_index in range(self.layers_num - 1)]

        self.displacement_vectors = []
        self.displacement_vectors = [np.random.rand(self.layers[layer_idx]) for layer_idx in range(1, self.layers_num)]

    # ===================================
    # ===Work with displacement vector===
    # ===================================
    def get_displacement_vector(self, vector_idx: int):
        return self.displacement_vectors[vector_idx]

    def save_displacement_vector(self):
        f = open('parameters/dv_weights.csv', 'w')
        vector_idx = 0

        for vector in self.displacement_vectors:
            f.write(f'dv for {vector_idx} - {vector_idx + 1} layer \n')
            vector_idx += 1

            for vector_element in vector:
                f.write(f'{vector_element};')
            f.write('\n')

        f.close()

    def load_displacement_vector(self, path='parameters/dv_weights.csv'):
        self.displacement_vectors = []

        f = open(path, 'r')
        lines = [line.rstrip() for line in f]
        for line in lines:
            if ';' in line:
                vector = []
                for val in line.split(';'):
                    if val != '':
                        vector.append(float(val))
                self.displacement_vectors.append(vector)
        f.close()

    # ===================================
    # ==========Work with matrix=========
    # ===================================
    def get_weight_matrix(self, matrix_index: int):
        return self.weight_matrix[matrix_index]

    def save_weights(self):
        f = open('parameters/matrix_weights.csv', 'w')
        matrix_index = 0

        for matrix in self.weight_matrix:
            f.write(f'Matrix for {matrix_index} - {matrix_index + 1} layer \n')
            matrix_index += 1

            for matrix_string in matrix.getA():
                for matrix_element in matrix_string:
                    f.write(f'{matrix_element};')
                f.write('\n')
            f.write('\n')
        f.close()

    def load_weights(self, path='parameters/matrix_weights.csv'):
        self.weight_matrix = []
        matrix_rows = []

        f = open(path, 'r')
        lines = [line.rstrip() for line in f]
        for line in lines:
            # If line is matrix row
            if ';' in line:
                matrix_row = []
                for val in line.split(';'):
                    if val != '':
                        matrix_row.append(float(val))
                matrix_rows.append(matrix_row)

            # If empty string (next matrix)
            if line == '':
                self.weight_matrix.append(np.matrix(matrix_rows))
                matrix_rows = []
        f.close()

    # ===================================
    # ==========Basic functions==========
    # ===================================
    @staticmethod
    def softmax(vector: np.array):
        max_vector_value = np.max(vector)
        temp_vector = [elem - max_vector_value for elem in vector]  # minimize vector
        exp_vector = np.exp(temp_vector)
        return exp_vector / np.sum(exp_vector)

    @staticmethod
    def relu(vector: np.array):
        return np.maximum(0, vector)

    @staticmethod
    def d_relu(vector: np.array):
        return (vector > 0).astype(float)

    def predict(self, input_data: np.array):
        output = [] * self.layers[self.layers_num - 1]  # last layer size
        selected_class = 0

        # t - matrix operations res
        # h - layer res (f(t))
        layer_input = np.matrix(input_data)
        for layer_idx in range(self.layers_num - 1):
            t = layer_input * self.get_weight_matrix(layer_idx) + self.get_displacement_vector(layer_idx)

            # If not last layer
            if layer_idx != self.layers_num - 2:
                h = self.relu(t.getA())
                layer_input = np.matrix(h)
                debug(f'Layer:{layer_idx} is complete')
            else:
                output = self.softmax(t.getA()[0])
                selected_class = np.argmax(output)
                debug(f'Final layer:{layer_idx} is complete')
        return output, selected_class


a = NeuralNetwork(3, [4096, 512, 10])
a.load_displacement_vector()
a.load_weights()
# res = a.predict(np.random.randint(0, 2, 4096))
res, selected = a.predict([0 for i in range(4096)])
print(res, selected)
