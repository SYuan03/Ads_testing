from ads import dave_dropout
from keras.models import Model
from keras.utils import image_utils
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import pickle
import os

weights_path = "./ads/Autopilot.h5"

def preprocess_image(img_path, target_size=(100, 100)):
    img = image_utils.load_img(img_path, target_size=target_size)
    input_img_data = image_utils.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data

def update_bounds(intermediate_layer_outputs, layer_bounds):
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        output = intermediate_layer_output[0]
        layer = layer_to_compute[i]
        if layer_bounds[layer.name] is None:
            low_bound = [np.inf] * output.shape[-1]
            high_bound = [-np.inf] * output.shape[-1]
        else:
            (low_bound, high_bound) = layer_bounds[layer.name]
        for id_neuron in range(output.shape[-1]):
            val = np.mean(output[..., id_neuron])
            if val < low_bound[id_neuron]:
                low_bound[id_neuron] = val
            if val > high_bound[id_neuron]:
                high_bound[id_neuron] = val
        layer_bounds[layer.name] = (low_bound, high_bound)
def init_bounds(img_paths):
    for i, img_path in enumerate(img_paths):
        print(i)
        img = preprocess_image(img_path)
        internal_outputs = intermediate_model.predict(img)
        intermediate_outputs = internal_outputs[0:-1]
        update_bounds(intermediate_outputs)

    with open('/cache/Dave_dropout/train_outputs/layer_bounds.pkl', 'wb') as f:
        pickle.dump(layer_bounds, f, pickle.HIGHEST_PROTOCOL)

    for layer in layer_to_compute:
        (low_bound, high_bound) = layer_bounds[layer.name]
        layer_bounds_bin[layer.name] = [np.linspace(low_bound[i], high_bound[i], 1000 + 1)
                                                 for i in range(len(high_bound))]
    with open('/home/test/program/self-driving/testing/cache/Dave_dropout/train_outputs/layer_bounds_bin.pkl', 'wb') as f:
        pickle.dump(layer_bounds_bin, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    model = dave_dropout(load_weights=True, weights_path=weights_path)
    model.summary()
    layer_to_compute = [layer for layer in model.layers
                        if all(ex not in layer.name for ex in ['flatten', 'input', 'pool', 'dropout'])][0:-2]
    outputs_layer = [layer.output for layer in layer_to_compute]
    outputs_layer.append(model.layers[-1].output)
    intermediate_model = Model(input=model.input, output=outputs_layer)

    layer_bounds = {}
    layer_bounds_bin = {}
    for layer in layer_to_compute:
        layer_bounds[layer.name] = None


