from __future__ import print_function, division, absolute_import

import argparse
import random 
import json
from collections import OrderedDict

import cv2
import numpy as np
import torch as th
from sklearn.neighbors import KNeighborsClassifier

from models.learner import SRL4robotics
from preprocessing.utils import deNormalize
from preprocessing.data_loader import preprocessImage
from utils import detachToNumpy

VALID_MODELS = ["forward", "inverse", "reward", "priors", "episode-prior", "reward-prior", "triplet",
                "autoencoder", "vae", "dae", "random"]
AUTOENCODERS = ['autoencoder', 'vae', 'dae']


def getState(srl_model, obs, device):
    """
    Gets an image by using the decoder of a SRL model
    (when available)

    :param srl_model: (Pytorch model)
    :param state: ([float]) the state vector from latent space
    :param device: (pytorch device)
    :return: ([float])
    """
    obs = preprocessImage(obs)
    obs = th.tensor(obs.reshape((1,) + obs.shape).transpose(0, 3, 2, 1))
    with th.no_grad():
        obs = obs.to(device)
        net_out = srl_model.getStates(obs)
        state = detachToNumpy(net_out)[0].T

    return state

def getImage(srl_model, state, device):
    """
    Gets an image by using the decoder of a SRL model
    (when available)

    :param srl_model: (Pytorch model)
    :param state: ([float]) the state vector from latent space
    :param device: (pytorch device)
    :return: ([float])
    """
    with th.no_grad():
        state = th.from_numpy(np.array(state).reshape(1, -1)).float()
        state = state.to(device)

        net_out = srl_model.decode(state)
        img = detachToNumpy(net_out)[0].T

    img = deNormalize(img, mode="image_net")
    return img[:, :, ::-1]

def getNextState(srl_model, state, action, device):
    with th.no_grad():
        state = th.from_numpy(np.array(state).reshape(1, -1)).float()
        action = th.from_numpy(np.array(action).reshape(1, -1)).long()

        state = state.to(device)
        action = action.to(device)

        net_out = srl_model.forwardModel(state, action)
        state = detachToNumpy(net_out)[0]
        print ("Next State : " + str(state))

    return state

def main():
    parser = argparse.ArgumentParser(description="latent space enjoy")
    parser.add_argument('--log-dir', default='', type=str, help='directory to load model')
    parser.add_argument('--no-cuda', default=False, action="store_true")

    args = parser.parse_args()
    use_cuda = not args.no_cuda
    device = th.device("cuda" if th.cuda.is_available() and use_cuda else "cpu")

    srl_model, exp_config = SRL4robotics.loadSavedModel(args.log_dir, VALID_MODELS, cuda=use_cuda)

    losses = exp_config['losses']
    state_dim = exp_config['state-dim']

    split_dimensions = exp_config.get('split-dimensions')
    loss_dims = OrderedDict()
    n_dimensions = 0
    if split_dimensions is not None and isinstance(split_dimensions, OrderedDict):
        for loss_name, loss_dim in split_dimensions.items():
            print(loss_name, loss_dim)
            if loss_dim > 0 or len(split_dimensions) == 1:
                loss_dims[loss_name] = loss_dim

    if len(loss_dims) == 0:
        print(losses)
        loss_dims = {losses[0]: state_dim}

    # Load all the states and images
    data = json.load(open(args.log_dir + 'image_to_state.json'))
    X = np.array(list(data.values())).astype(float)
    y = list(data.keys())

    bound_max, bound_min, fig_names = {}, {}, {}
    start_indices, end_indices = {}, {}
    start_idx = 0

    should_exit = False

    cv2.namedWindow("Dream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dream", 500, 500)

    num_steps = 1000

    initial_img = cv2.imread("data/"+y[random.randint(0, len(y))]+".jpg")
    inital_state = getState(srl_model.model, initial_img, device)
    state = inital_state
    # print ("First state : " + str(state))

    obs = initial_img
    for step in range(num_steps):
        cv2.imshow("Dream", obs)
        k = cv2.waitKey(150) & 0xFF
        if k == 27:
            break

        action = random.randint(0, 5)
        # print ("Selected action : " + str(action))
        next_state_pred = getNextState(srl_model.model, state, action,
                device)
        obs = getImage(srl_model.model.model, next_state_pred, device)

        state = next_state_pred

        # Next state comes from the encoding of the current observation
        # state = getState(srl_model.model, img, device) 

    # gracefully close
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
