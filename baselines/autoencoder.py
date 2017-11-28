from __future__ import print_function, division, absolute_import

import time
import argparse

import numpy as np
import torch as th
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import parseDataFolder
from preprocessing.data_loader import AutoEncoderDataLoader
from preprocessing.preprocess import INPUT_DIM
from preprocessing.utils import deNormalize
from models.base_learner import BaseLearner
from models.models import LinearAutoEncoder, DenseAutoEncoder, CNNAutoEncoder
from plotting.representation_plot import plot_representation, plt, plot_image

# Python 2/3 compatibility
try:
    input = raw_input
except NameError:
    pass

DISPLAY_PLOTS = True
EPOCH_FLAG = 1  # Plot every 1 epoch
BATCH_SIZE = 32
NOISE_FACTOR = 0.1
TEST_BATCH_SIZE = 512


class AutoEncoderLearning(BaseLearner):
    """
    :param state_dim: (int)
    :param model_type: (str) one of "cnn" or "mlp"
    :param seed: (int)
    :param learning_rate: (float)
    :param cuda: (bool)
    """

    def __init__(self, state_dim, model_type="cnn", log_folder="logs/default",
                 seed=1, learning_rate=0.001, cuda=False):

        super(AutoEncoderLearning, self).__init__(state_dim, BATCH_SIZE, seed, cuda)

        if model_type == "cnn":
            self.model = CNNAutoEncoder(self.state_dim)
        elif model_type == "mlp":
            self.model = DenseAutoEncoder(INPUT_DIM, self.state_dim)
            # self.model = LinearAutoEncoder(INPUT_DIM, self.state_dim)
        else:
            raise ValueError("Unknown model: {}".format(model_type))
        print("Using {} model".format(model_type))

        if cuda:
            self.model.cuda()
        learnable_params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizer = th.optim.Adam(learnable_params, lr=learning_rate)

        self.log_folder = log_folder

    def _predFn(self, observations, restore_train=True):
        """
        Predict states in test mode given observations
        :param observations: (PyTorch Variable)
        :param restore_train: (bool) whether to restore training mode after prediction
        :return: (numpy tensor)
        """
        # Switch to test mode
        self.model.eval()
        states, _ = self.model(observations)
        if restore_train:
            # Restore training mode
            self.model.train()
        if self.cuda:
            # Move the tensor back to the cpu
            return states.data.cpu().numpy()
        return states.data.numpy()

    def learn(self, images_path, rewards):
        """
        Learn a state representation
        :param images_path: (numpy 1D array)
        :param rewards: (numpy 1D array)
        :return: (numpy tensor) the learned states for the given observations
        """
        x_indices = np.arange(len(images_path)).astype(np.int64)

        # Split into train/validation set
        x_train, x_val = train_test_split(x_indices, test_size=0.33, random_state=self.seed)

        train_loader = AutoEncoderDataLoader(x_train, images_path,
                                             batch_size=self.batch_size,
                                             noise_factor=NOISE_FACTOR)
        val_loader = AutoEncoderDataLoader(x_val, images_path,
                                           batch_size=TEST_BATCH_SIZE,
                                           noise_factor=NOISE_FACTOR, is_training=False)
        # For plotting
        data_loader = AutoEncoderDataLoader(x_indices, images_path, batch_size=TEST_BATCH_SIZE,
                                            no_targets=True, is_training=False)

        # TRAINING -----------------------------------------------------------------------------------------------------
        criterion = nn.MSELoss(size_average=True)
        best_error = np.inf
        best_model_path = "{}/srl_ae_model.pth".format(self.log_folder)
        print("Training...")
        self.model.train()
        start_time = time.time()
        for epoch in range(N_EPOCHS):
            # In each epoch, we do a full pass over the training data:
            train_loss, val_loss = 0, 0
            train_loader.resetAndShuffle()
            pbar = tqdm(total=len(train_loader))
            for batch_idx, (noisy_obs, obs) in enumerate(train_loader):
                if self.cuda:
                    noisy_obs, obs = noisy_obs.cuda(), obs.cuda()

                _, decoded = self.model(noisy_obs)
                self.optimizer.zero_grad()
                loss = criterion(decoded, obs)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.data[0]
                pbar.update(1)
            pbar.close()

            train_loss /= len(train_loader)

            self.model.eval()
            val_loader.resetIterator()
            # Pass on the validation set
            for noisy_obs, obs in val_loader:
                if self.cuda:
                    noisy_obs, obs = noisy_obs.cuda(), obs.cuda()

                _, decoded = self.model(noisy_obs)
                loss = criterion(decoded, obs)
                val_loss += loss.data[0]

            val_loss /= len(val_loader)
            # Plot Reconstructed Image
            plot_image(deNormalize(noisy_obs[0].data.cpu().numpy()), "Input Validation Image")
            plot_image(deNormalize(decoded[0].data.cpu().numpy()), "Reconstructed Image")

            self.model.train()  # Restore train mode

            # Save best model
            if val_loss < best_error:
                best_error = val_loss
                th.save(self.model.state_dict(), best_model_path)

            # Then we print the results for this epoch:
            if (epoch + 1) % EPOCH_FLAG == 0:
                print("Epoch {:3}/{}".format(epoch + 1, N_EPOCHS))
                print("train_loss:{:.4f} val_loss:{:.4f}".format(train_loss, val_loss))
                print("{:.2f}s/epoch".format((time.time() - start_time) / (epoch + 1)))
                if DISPLAY_PLOTS:
                    # Optionally plot the current state space
                    plot_representation(self.predStatesWithDataLoader(data_loader), rewards, add_colorbar=epoch == 0,
                                        name="Learned State Representation (Training Data)")
        if DISPLAY_PLOTS:
            plt.close("Learned State Representation (Training Data)")

        # load best model before predicting states
        self.model.load_state_dict(th.load(best_model_path))
        # return predicted states for training observations
        return self.predStatesWithDataLoader(data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised Learning')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch_size (default: 32)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--no-plots', action='store_true', default=False, help='disables plots')
    parser.add_argument('--model_type', type=str, default="cnn", help='Model architecture (default: "cnn")')
    parser.add_argument('--data_folder', type=str, default="", help='Dataset folder', required=True)
    parser.add_argument('--state_dim', type=int, default=2, help='state dimension (default: 2)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and th.cuda.is_available()
    DISPLAY_PLOTS = not args.no_plots
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    args.data_folder = parseDataFolder(args.data_folder)
    log_folder = "logs/{}/baselines".format(args.data_folder)

    print('Log folder: {}'.format(log_folder))

    print('Loading data ... ')
    rewards = np.load("data/{}/preprocessed_data.npz".format(args.data_folder))['rewards']

    ground_truth = np.load("data/{}/ground_truth.npz".format(args.data_folder))

    print('Learning a state representation ... ')
    srl = AutoEncoderLearning(args.state_dim, model_type=args.model_type, seed=args.seed,
                              log_folder=log_folder, learning_rate=args.learning_rate,
                              cuda=args.cuda)
    learned_states = srl.learn(ground_truth['images_path'], rewards)
    srl.saveStates(learned_states, ground_truth['images_path'], rewards, log_folder,
                   name="_autoencoder")

    name = "Learned State Representation - {} \n Autoencoder state_dim={}".format(args.data_folder, args.state_dim)
    path = "{}/learned_states_ae.png".format(log_folder)
    plot_representation(learned_states, rewards, name, add_colorbar=True, path=path)

    if DISPLAY_PLOTS:
        input('\nPress any key to exit.')