import random
import numpy as np
from PIL import Image

DEBUG_ENABLE = True


def debug(text: str):
    if DEBUG_ENABLE:
        print(f'dbg: {text}')


class DataSet:
    def __init__(self, path, img_size):
        self.answer_array = {}
        answer_id = 0

        self.image_size = img_size
        self.dataset_array = []

        f = open(path, 'r')
        lines = [line.rstrip() for line in f]
        for line in lines:
            line_elements = line.split(';')
            img_collection = {'Id': int(line_elements[-2]), 'Name': line_elements[-3][1::],
                              'Data': [int(i) for i in line_elements[0:img_size]]}
            self.dataset_array.append(img_collection)

            # Form correct answer list
            if img_collection['Name'] not in self.answer_array:
                self.answer_array[img_collection['Name']] = answer_id
                answer_id += 1

        self.dataset_size = len(lines)
        f.close()

    def get_dataset_element(self, dataset_id: int):
        return self.dataset_array[dataset_id]

    def get_correct_answer(self, img_type: str):
        answer_id = self.answer_array[img_type]
        result = np.zeros(len(self.answer_array))
        result[answer_id] = 1

        return result

    def get_correct_answer_idx(self, img_type: str):
        return self.answer_array[img_type]

    @staticmethod
    # Method return prepared img object for recognizing
    def prepare_img_for_recognize(path: str, white_threshold=255):
        # Image must have valid size
        im = Image.open(path)
        pixels = np.asarray(im)
        pixels2 = np.ndarray((64, 64, 3), dtype='uint8')
        bin_img = []
        for pxl_row_idx in range(im.height):
            j = 0
            for px in pixels[pxl_row_idx]:
                # White is 0, black is 1
                color = 255 if (px[0] >= white_threshold and px[1] >= white_threshold and px[2] >= white_threshold) else 0
                pixels2[pxl_row_idx, j] = (color, color, color)
                j += 1
                bin_img.append(int(not (px[0] >= white_threshold and px[1] >= white_threshold and px[2] >= white_threshold)))

        im.close()
        im2 = Image.fromarray(pixels2, 'RGB')
        im2.show()
        return {'Id': -1, 'Name':'None', 'Data': bin_img}


class NeuralNetwork:
    def __init__(self, layers_num: int, perceptron_count: [int], rounds_num=30, learning_speed=0.001):
        self.layers = []
        self.layers_num = layers_num

        for layer_idx in range(self.layers_num):
            self.layers.append(perceptron_count[layer_idx])

        # rows - input neurons; cols - output neurons
        self.weight_matrix = [np.matrix([np.random.rand(self.layers[matrix_index + 1])
                                         for _ in range(self.layers[matrix_index])])
                              for matrix_index in range(self.layers_num - 1)]

        self.displacement_vectors = [np.matrix(np.random.rand(self.layers[layer_idx])) for layer_idx in range(1, self.layers_num)]

        self.layers_input_data = [None] * (self.layers_num - 1)
        self.layers_t_data = [None] * (self.layers_num - 1)

        # Parameters
        self.sigmoid_a = 1
        self.sigmoid_max_val = 10
        self.rounds_num = rounds_num
        self.learning_speed = learning_speed

    # ===================================
    # ===Work with displacement vector===
    # ===================================
    def get_displacement_vector(self, vector_idx: int):
        return self.displacement_vectors[vector_idx]

    def save_displacement_vector(self, path='parameters/dv_weights.csv'):
        f = open(path, 'w')
        vector_idx = 0

        for vector in self.displacement_vectors:
            f.write(f'dv for {vector_idx} - {vector_idx + 1} layer \n')
            vector_idx += 1

            # TODO: fix different types from load and random generate vectors
            vector = np.matrix(vector)

            for vector_string in vector.getA():
                for vector_element in vector_string:
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
    # ==========Work with weights=========
    # ===================================
    def get_weight_matrix(self, matrix_index: int):
        return self.weight_matrix[matrix_index]

    def save_weights(self, path='parameters/matrix_weights.csv'):
        f = open(path, 'w')
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
    def softmax(matrix: np.array, big_nums=False):
        def min_max_scaler(data, new_min=-20, new_max=20):
            min_vals = np.min(data)
            max_vals = np.max(data)

            scaled_data = new_min + (data - min_vals) / (max_vals - min_vals) * (
                    new_max - new_min)

            return scaled_data

        if big_nums:
            # max_matrix_value = np.max(matrix)
            # matrix = [elem - max_matrix_value for elem in matrix]  # minimize vector
            matrix = min_max_scaler(matrix)

        exp_matrix = np.exp(matrix)
        return exp_matrix / np.sum(exp_matrix)

    @staticmethod
    def relu(matrix: np.array):
        return np.maximum(0, matrix)

    @staticmethod
    def d_relu(matrix: np.array):
        return (matrix >= 0).astype(float)

    def sigmoid(self, matrix: np.array):
        float_coefficient = self.sigmoid_max_val / np.max(matrix)
        matrix *= self.sigmoid_a * (-1) * float_coefficient
        return 1/(1+np.exp(matrix))

    def d_sigmoid(self, matrix: np.array):
        f = self.sigmoid(matrix)
        for i_f in range(f.shape[0]):
            for j_f in range(f.shape[1]):
                matrix[i_f, j_f] = f[i_f, j_f] * (1 - f[i_f, j_f])
        return matrix

    @staticmethod
    def cross_entropy(z: np.array, y: np.array):
        return np.sum(y * np.log(z)) * (-1)

    @staticmethod
    def sparse_cross_entropy(z: np.array, y_idx: int):
        return np.log(z[y_idx]) * (-1)

    @staticmethod
    def d_last_layer(z, y):
        return z - y

    def d_e_respect_to_d_w(self, d_e_respect_to_d_t: np.array, layer_idx: int):
        # Decrement idx because zero layer don't change data
        layer_idx -= 1

        layer_input = self.layers_input_data[layer_idx]
        return np.transpose(layer_input) * d_e_respect_to_d_t

    @staticmethod
    def d_e_respect_to_d_vector(d_e_respect_to_d_t: np.array):
        return d_e_respect_to_d_t

    def d_e_respect_to_d_h(self, d_e_respect_to_d_t: np.array, prev_layer_index: int):
        # Decrement idx because parameters trail by one idx (3 layers == 2 parameters)
        prev_layer_index -= 1

        return d_e_respect_to_d_t * np.transpose(self.get_weight_matrix(prev_layer_index))

    def d_e_respect_to_d_t(self, d_e_respect_to_d_h: np.array, prev_layer_index: int):
        # Decrement idx because zero layer don't change data
        prev_layer_index -= 1

        prev_t = self.layers_t_data[prev_layer_index]
        return d_e_respect_to_d_h.getA() * self.d_relu(prev_t).getA()

    # ===================================
    # ==========Main functions===========
    # ===================================

    def predict(self, input_data: np.array):
        output = [] * self.layers[self.layers_num - 1]  # last layer size
        selected_class = 0

        # t - matrix operations res
        # h - layer res (f(t))
        layer_input = np.matrix(input_data)
        for layer_idx in range(self.layers_num - 1):
            # Save layer input for back propagation
            self.layers_input_data[layer_idx] = layer_input

            t1 = layer_input * self.get_weight_matrix(layer_idx)
            t = t1 + self.get_displacement_vector(layer_idx)
            self.layers_t_data[layer_idx] = t

            # If not last layer
            if layer_idx != self.layers_num - 2:
                layer_input = self.relu(t.getA())
                # debug(f'Layer:{layer_idx} is complete')
            else:
                output = self.softmax(t.getA()[0], True)
                selected_class = np.argmax(output)
                # debug(f'Final layer:{layer_idx} is complete')
        return output, selected_class

    def update_params(self, d_w: np.array, d_vector: np.array, parameters_idx: int):
        # Decrement idx because parameters trail by one idx (3 layers == 2 parameters)
        parameters_idx -= 1

        self.weight_matrix[parameters_idx] -= self.learning_speed * d_w
        self.displacement_vectors[parameters_idx] -= self.learning_speed * d_vector

    def backward_propagation(self, neural_res: np.array, correct_answer: np.array):
        # Special action for first (in fact last) layer
        last_layer_idx = self.layers_num - 1
        d_e_to_d_t = self.d_last_layer(neural_res, correct_answer)

        d_e_to_d_vector = self.d_e_respect_to_d_vector(d_e_to_d_t)
        d_e_to_d_input = self.d_e_respect_to_d_h(d_e_to_d_t, last_layer_idx)
        d_e_to_d_w = self.d_e_respect_to_d_w(d_e_to_d_t, last_layer_idx)

        self.update_params(d_e_to_d_w, d_e_to_d_vector, last_layer_idx)

        # Start back propagation
        for layer_idx in range(last_layer_idx - 1, 0, -1):
            d_e_to_d_t = self.d_e_respect_to_d_t(d_e_to_d_input, layer_idx)
            d_e_to_d_vector = self.d_e_respect_to_d_vector(d_e_to_d_t)
            d_e_to_d_w = self.d_e_respect_to_d_w(d_e_to_d_t, layer_idx)

            self.update_params(d_e_to_d_w, d_e_to_d_vector, layer_idx)
            if layer_idx != 0:
                d_e_to_d_input = self.d_e_respect_to_d_h(d_e_to_d_t, layer_idx)

    def network_education(self, dataset_path: str):
        img_size = self.layers[0]
        debug(f'Img_size:{img_size}')
        dataset = DataSet(dataset_path, img_size)

        for round_idx in range(self.rounds_num):
            random.shuffle(dataset.dataset_array)

            debug(f'Start {round_idx} round')
            for img_i in range(dataset.dataset_size):
                dataset_object = dataset.get_dataset_element(img_i)

                object_data = dataset_object['Data']
                predict_res, _ = self.predict(object_data)

                correct_answer_idx = dataset.get_correct_answer_idx(dataset_object['Name'])
                correct_answer = dataset.get_correct_answer(dataset_object['Name'])

                predict_err = self.sparse_cross_entropy(predict_res, correct_answer_idx)

                if img_i % 100 == 0:
                    debug(f'For Round:{round_idx} image:{img_i} err:{predict_err}')

                self.backward_propagation(predict_res, correct_answer)


if __name__ == "__main__":
    # ===========Hyper params=========== #
    #  INPUT_DIM = 64*64                 #
    #  OUTPUT_DIM = 10                   #
    #  LAYERS_NUM = 3                    #
    #  NEURONS_NUM_IN_FIRST_LAYER - 512  #
    # ================================== #

    # Prepare DataSet
    dataSet = DataSet('dataSet/Test/annotation.csv', 64*64)

    # Prepare NeuralNetwork
    a = NeuralNetwork(3, [4096, 64, 10], rounds_num=30, learning_speed=0.0008)
    a.load_displacement_vector('dataSet/Learning/vectors64_2.csv')
    a.load_weights('dataSet/Learning/matrix64_2.csv')

    # Educate NeuralNetwork
    # a.network_education('dataSet/Learning/annotation.csv')

    # Save results
    # a.save_displacement_vector('dataSet/Learning/vectors64_2.csv')
    # a.save_weights('dataSet/Learning/matrix64_2.csv')

    # Check accuracy
    counter = 0
    for i in range(dataSet.dataset_size):
        predict_element = dataSet.get_dataset_element(i)
        res, selected = a.predict(predict_element['Data'])
        debug(f'{selected} | {dataSet.get_correct_answer_idx(predict_element["Name"])}')
        if selected == dataSet.get_correct_answer_idx(predict_element['Name']):
            counter += 1

    print(counter / dataSet.dataset_size)

    img_obj = dataSet.prepare_img_for_recognize('temp_center_resize.jpg', white_threshold=250)
    res, selected = a.predict(img_obj['Data'])

    print(selected)

    for key, val in dataSet.answer_array.items():
        if val == selected:
            print(key)
            break