import logging
import os
import pickle
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from torch.autograd import Variable
from torchvision import transforms

from ads.TrainModel import dave_dropout
from engine import EAEngine
from munit.MUNIT.trainer import MUNIT_Trainer
from munit.MUNIT.utils import get_config

K.set_learning_phase(0)
####################
# parameters
####################
# the image path
# train_image_paths = '/home/test/program/self-driving/dataset/train/center/'
test_image_path = './ads/data/sunny/'
# munit model path
# config_path = '/home/test/program/self-driving/munit/configs/snowy.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/snowy/gen_01000000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/night.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/night/gen_01000000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/rainy.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/rainy/gen_01000000.pt'
# config_path = '/home/test/program/self-driving/munit/configs/sunny.yaml'
# checkpoint_path = '/home/test/program/self-driving/munit/checkpoints/sunny/gen_01250000.pt'
config_path = 'munit/MUNIT/configs/rainy.yaml'
checkpoint_path = 'munit/MUNIT/models/rainy.pt'
# the self-driving system's weight file
weights_path = './ads/Autopilot.h5'
target_size = (40, 40)
nb_part = 1000

###################
# set logger
###################
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
os.makedirs('./logger', exist_ok=True)
handler = logging.FileHandler("./logger/knc_rainy_ES_time_cost.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

model, callback_list = dave_dropout(load_weights=True, weights_path=weights_path)
model.summary()

layer_to_compute = [layer for layer in model.layers
                    if all(ex not in layer.name for ex in ['flatten', 'input', 'pool', 'dropout'])][0:-2]

outputs_layer = [layer.output for layer in layer_to_compute]
outputs_layer.append(model.layers[-1].output)
intermediate_model = Model(inputs=model.input, outputs=outputs_layer)

with open('cache/Dave_dropout/train_outputs/layer_bounds_bin.pkl', 'rb') as f:
    layer_bounds_bins = pickle.load(f)
with open('cache/Dave_dropout/test_outputs/knc_coverage.pkl', 'rb') as f:
     knc_cov_dict = pickle.load(f)
#with open('cache/Dave_dropout/test_outputs/knc_coverage_cache_snowy_random.pkl', 'rb') as f:
 #   knc_cov_dict = pickle.load(f)
with open('cache/Dave_dropout/test_outputs/steering_angles.pkl', 'rb') as f:
    original_steering_angles = pickle.load(f)

#####################
# build MUNIT model
#####################
config = get_config(config_path)

munit = MUNIT_Trainer(config)

try:
    state_dict = torch.load(checkpoint_path)
    munit.gen_a.load_state_dict(state_dict['a'])
    munit.gen_b.load_state_dict(state_dict['b'])
except Exception:
    raise RuntimeError('load model failed')

munit.cuda()
new_size = config['new_size']  # the GAN's input size is 256*256
style_dim = config['gen']['style_dim']
encode = munit.gen_a.encode
style_encode = munit.gen_b.encode
decode = munit.gen_b.decode


# process the munit's input
transform = transforms.Compose([transforms.Resize(new_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# function generator use to generate transformed images by MUNIT
def generator(img, style):
    with torch.no_grad():
        img = Variable(transform(img).unsqueeze(0).cuda())
        s = Variable(style.unsqueeze(0).cuda())
        content, _ = encode(img)

        outputs = decode(content, s)
        outputs = (outputs + 1) / 2.
        del img
        del s
        del content
        return outputs.data


# process the generated image from munit
def preprocess_transformed_images(original_image):
    tensor = original_image.view(1, original_image.size(0), original_image.size(1), original_image.size(2))
    tensor = tensor.clone()

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    def norm_range(img):
        norm_ip(img, float(img.min()), float(img.max()))

    norm_range(tensor)
    tensor = tensor.squeeze()
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(ndarr)
    img = img.resize((target_size[1], target_size[0])).convert('L')
    input_img_data = image.image_utils.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data, mode='tf')
    return input_img_data


######################
# create some functions to calculate
######################
def calculate_uncovered_knc_sections():
    """
    Calculate the number of uncovered sections on the KNC criterion
    :return:
    """
    return np.sum([np.count_nonzero(knc_cov_dict[layer.name] == 0) for layer in layer_to_compute])


def get_new_covered_knc_sections(intermediate_layer_outputs, cov_dict):
    # cov_dict = deepcopy(knc_cov_dict)
    new_covered_sections = 0
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        output = intermediate_layer_output[0]
        layer = layer_to_compute[i]
        bins = layer_bounds_bins[layer.name]
        for id_neuron in range(output.shape[-1]):
            _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
            if _bin >= nb_part or cov_dict[layer.name][id_neuron][_bin]:
                continue
            new_covered_sections = new_covered_sections + 1
            cov_dict[layer.name][id_neuron][_bin] = True
    return new_covered_sections


def update_knc(intermediate_layer_outputs):
    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        layer = layer_to_compute[i]
        bins = layer_bounds_bins[layer.name]
        output = intermediate_layer_output[0]
        for id_neuron in range(output.shape[-1]):
            _bin = np.digitize(np.mean(output[..., id_neuron]), bins[id_neuron]) - 1
            if _bin >= nb_part or knc_cov_dict[layer.name][id_neuron][_bin]:
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


# the fitness function
def fitness_function(original_images, original_preds, theoretical_uncovered_sections, style):
    preds = []
    cov_dict = deepcopy(knc_cov_dict)
    new_covered_sections = 0
    logger.info("do prediction")
    for img_num, img in enumerate(original_images):
        if img_num % 5 != 0:
            continue
        if img_num % 100 == 0:
            print("the {i}th image".format(i=img_num))
        logger.info("generate driving scenes {i}".format(i=img_num))
        transformed_image = generator(img, style)[0]
        transformed_image = preprocess_transformed_images(transformed_image)
        # logger.info("finish generating driving scenes")

        logger.info("obtain internal outputs")
        internal_outputs = intermediate_model.predict(transformed_image)
        intermediate_outputs = internal_outputs[0:-1]
        preds.append(internal_outputs[-1][0][0])
        # logger.info("finish obtaining internal outputs")

        logger.info("calculate coverage")
        new_covered_sections += get_new_covered_knc_sections(intermediate_outputs, cov_dict)
        # logger.info("finish calculating coverage")
    # for img in original_images:
    #     transformed_image = generator(img, style)[0]
    #     transformed_image = preprocess_transformed_images(transformed_image)
    #
    #     internal_outputs = intermediate_model.predict(transformed_image)
    #     intermediate_outputs = internal_outputs[0:-1]
    #     preds.append(internal_outputs[-1][0][0])
    #
    #     new_covered_sections += get_new_covered_knc_sections(intermediate_outputs, cov_dict)

    logger.info(new_covered_sections)
    # logger.info(len(transformed_preds))
    transformed_preds = np.asarray(preds)
    o1 = float(new_covered_sections) / float(theoretical_uncovered_sections)
    o2 = np.average(np.abs(transformed_preds - original_preds))
    o2 = o2 / (o2 + 1)  # normalize
    logger.info("the o1 is {}".format(o1))
    logger.info("the o2 is {}".format(o2))
    del new_covered_sections
    del transformed_preds
    del cov_dict
    return o1 + o2


# the wrapper of fitness function
def fitness_function_wrapper(original_images, original_preds, theoretical_uncovered_sections):
    def decorator(func):
        def wrapper(*args, **kwargs):
            return fitness_function(original_images, original_preds, theoretical_uncovered_sections, *args, **kwargs)
        return wrapper
    return decorator


def count_error_behaviors(ori_preds, preds):
    count = 0
    for i in range(len(ori_preds)):
        if np.abs(ori_preds[i] - preds[i]) > 0.2 \
                or (ori_preds[i] < 0 and preds[i] > 0) or (ori_preds[i] > 0 and preds[i] < 0):
            count = count + 1
    return count


def update_history(original_images, style):
    preds = []
    for img_num, img in enumerate(original_images):
        if img_num % 5 != 0:
            continue
        if img_num % 100 == 0:
            print("update the {i}th image".format(i=img_num))
        transformed_image = generator(img, style)[0]
        transformed_image = preprocess_transformed_images(transformed_image)

        internal_outputs = intermediate_model.predict(transformed_image)
        intermediate_outputs = internal_outputs[0:-1]
        preds.append(internal_outputs[-1][0][0])

        update_knc(intermediate_outputs)
    return preds


def testing():
    images_path = [(test_image_path + image_file) for image_file in sorted(os.listdir(test_image_path))
                   if image_file.endswith(".jpg")]
    orig_images_for_transform = [Image.open(path).convert('RGB') for path in images_path]

    iteration = 0

    print('## current_knc_coverage: {}'.format(current_knc_coverage()))

    while True:
        logger.info("the {nb_iter} begin".format(nb_iter=iteration))
        # theoretical_uncovered_sections = nb_neurons * nb_images
        nb_uncovered_sections = calculate_uncovered_knc_sections()
        theoretical_uncovered_sections = calculate_uncovered_knc_sections()

        theoretical_uncovered_sections = theoretical_uncovered_sections \
            if theoretical_uncovered_sections <= nb_uncovered_sections else nb_uncovered_sections

        print('## theoretical_uncovered_sections: {}'.format(theoretical_uncovered_sections))

        @fitness_function_wrapper(orig_images_for_transform, original_steering_angles, theoretical_uncovered_sections)
        def fitness(style):
            pass

        search_handler = EAEngine(style_dim=style_dim, fitness_func=fitness, logger=logger)
        print('best')
        best = search_handler.run(64)

        transformed_preds = update_history(orig_images_for_transform, best)

        logger.info("the {nb_iter} finish".format(nb_iter=iteration))
        print("current k-multi section coverage is {}".format(current_knc_coverage()))
        logger.info("current k-multi section coverage is {}".format(current_knc_coverage()))
        # logger.info("current neuron boundary coverage is {}".format(current_knc_coverage()))
        logger.info("the best style code is {}".format(best))
        logger.info("the number of error behaviors is {}".format(count_error_behaviors(original_steering_angles, transformed_preds)))

        with open('cache/Dave_dropout/test_outputs/knc_coverage_cache_rainy_random.pkl', 'wb') \
                as f:
            pickle.dump(knc_cov_dict, f, pickle.HIGHEST_PROTOCOL)

        iteration += 1

        if iteration == 4 or current_knc_coverage() >= 0.80:
            # Terminate the algorithm when the coverage is greater than threshold
            # or the number of iterations is euqal to four
            break


if __name__ == '__main__':
    testing()
