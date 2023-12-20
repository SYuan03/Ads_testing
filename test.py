from ads import dave_dropout
from keras.models import Model
import numpy as np
import pickle
import os
import cv2
import imageio

weights_path = "./ads/Autopilot.h5"
img_path = "./ads/data/sunny/"


def keras_process_image(img):
    frame = imageio.imread(img, pilmode="RGB")
    gray = cv2.resize((cv2.cvtColor(frame, cv2.COLOR_RGB2HSV))[:, :, 1], (40, 40))
    image_x = 40
    image_y = 40
    gray = cv2.resize(gray, (image_x, image_y))
    gray = np.array(gray, dtype=np.float32)
    gray = np.reshape(gray, (-1, image_x, image_y, 1))
    return gray


def update_bounds(intermediate_layer_outputs, layer_bounds_):
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        output = intermediate_layer_output[0]
        layer = layer_to_compute[i]
        if layer_bounds_[layer.name] is None:
            low_bound = [np.inf] * output.shape[-1]
            high_bound = [-np.inf] * output.shape[-1]
        else:
            (low_bound, high_bound) = layer_bounds_[layer.name]
        for id_neuron in range(output.shape[-1]):
            val = np.mean(output[..., id_neuron])
            if val < low_bound[id_neuron]:
                low_bound[id_neuron] = val
            if val > high_bound[id_neuron]:
                high_bound[id_neuron] = val
        layer_bounds_[layer.name] = (low_bound, high_bound)


def init_bounds(img_paths, layer_bounds_):
    img_list = os.listdir(img_paths)
    for i, img_path_ in enumerate(img_list):
        if i % 5 != 0:
            continue
        print(i)
        img = keras_process_image(img_paths + img_path_)
        internal_outputs = intermediate_model.predict(img)
        intermediate_outputs = internal_outputs[0:-1]
        update_bounds(intermediate_outputs, layer_bounds_)

    with open('./cache/Dave_dropout/train_outputs/layer_bounds.pkl', 'wb') as file:
        pickle.dump(layer_bounds, file, pickle.HIGHEST_PROTOCOL)

    for layer in layer_to_compute:
        (low_bound, high_bound) = layer_bounds_[layer.name]
        layer_bounds_bin[layer.name] = [np.linspace(low_bound[i], high_bound[i], 1000 + 1)
                                        for i in range(len(high_bound))]
    with open('./cache/Dave_dropout/train_outputs/layer_bounds_bin.pkl', 'wb') as file:
        pickle.dump(layer_bounds_bin, file, pickle.HIGHEST_PROTOCOL)


def update_nbc(intermediate_layer_outputs, layer_bounds_):
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        output = intermediate_layer_output[0]
        layer = layer_to_compute[i]
        (low_bound, high_bound) = layer_bounds_[layer.name]
        for id_neuron in range(output.shape[-1]):
            val = np.mean(output[..., id_neuron])
            if val < low_bound[id_neuron] and not nbc_cov_dict[layer.name][id_neuron][0]:
                nbc_cov_dict[layer.name][id_neuron][0] = True
            elif val > high_bound[id_neuron] and not nbc_cov_dict[layer.name][id_neuron][1]:
                nbc_cov_dict[layer.name][id_neuron][1] = True
            else:
                continue


def current_nbc_coverage():
    """
    Calculate the current Neuron Boundary Coverage
    :return:
    """
    covered = 0
    total = 0
    for layer in layer_to_compute:
        covered = covered + np.count_nonzero(nbc_cov_dict[layer.name])
        total = total + np.size(nbc_cov_dict[layer.name])
    return covered / float(total)


def update_knc(intermediate_layer_outputs):
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        layer = layer_to_compute[i]
        bins = layer_bounds_bin[layer.name]
        output = intermediate_layer_output[0]
        for id_neuron in range(output.shape[-1]):
            _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
            if _bin >= 1000 or knc_cov_dict[layer.name][id_neuron][_bin]:
                continue
            knc_cov_dict[layer.name][id_neuron][_bin] = True


def current_knc_coverage():
    """
    Calculate the current K-Multi Section Neuron Coverage
    :return:
    """
    covered = 0
    total = 0
    for layer in layer_to_compute:
        covered = covered + np.count_nonzero(knc_cov_dict[layer.name])
        total = total + np.size(knc_cov_dict[layer.name])
    return covered / float(total)


def init_cov(img_paths, layer_bounds_):
    img_list = os.listdir(img_paths)
    preds = []
    for i, img_path_ in enumerate(img_list):
        if i % 5 != 0:
            continue
        print(i)
        img = keras_process_image(img_paths + img_path_)
        internal_outputs = intermediate_model.predict(img)
        intermediate_outputs = internal_outputs[0:-1]
        preds.append(internal_outputs[-1][0][0])
        update_knc(intermediate_outputs)
        update_nbc(intermediate_outputs, layer_bounds_)

    print(current_knc_coverage())
    print(current_nbc_coverage())

    with open('./cache/Dave_dropout/test_outputs/knc_coverage.pkl', 'wb') as file:
        pickle.dump(knc_cov_dict, file, pickle.HIGHEST_PROTOCOL)

    with open('./cache/Dave_dropout/test_outputs/nbc_coverage.pkl', 'wb') as file:
        pickle.dump(nbc_cov_dict, file, pickle.HIGHEST_PROTOCOL)

    with open('./cache/Dave_dropout/test_outputs/steering_angles.pkl', 'wb') as file:
        pickle.dump(preds, file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    model, _ = dave_dropout(load_weights=True, weights_path=weights_path)
    model.summary()
    layer_to_compute = [layer for layer in model.layers
                        if all(ex not in layer.name for ex in ['input', 'flatten', 'pool', 'dropout'])][0:-2]
    outputs_layer = [layer.output for layer in layer_to_compute]
    outputs_layer.append(model.layers[-1].output)
    intermediate_model = Model(inputs=model.input, outputs=outputs_layer)

    layer_bounds = {}
    layer_bounds_bin = {}
    for layer in layer_to_compute:
        layer_bounds[layer.name] = None
    os.makedirs('./cache/Dave_dropout/train_outputs', exist_ok=True)
    os.makedirs('./cache/Dave_dropout/test_outputs', exist_ok=True)
    init_bounds(img_path, layer_bounds)
    with open('./cache/Dave_dropout/train_outputs/layer_bounds.pkl', 'rb') as f:
        layer_bounds = pickle.load(f)

    with open('./cache/Dave_dropout/train_outputs/layer_bounds_bin.pkl',
              'rb') as f:
        layer_bounds_bin = pickle.load(f)

    knc_cov_dict = {}
    nbc_cov_dict = {}
    for layer in layer_to_compute:
        knc_cov_dict[layer.name] = np.zeros((layer.output.get_shape()[-1], 1000), dtype='bool')
        nbc_cov_dict[layer.name] = np.zeros((layer.output.get_shape()[-1], 2), dtype='bool')
    init_cov(img_path, layer_bounds)
