from .autoencoders import CNNAutoEncoder, DenseAutoEncoder, LinearAutoEncoder
from .vae import CNNVAE, DenseVAE
from .forward_inverse import BaseForwardModel, BaseInverseModel, BaseRewardModel
from .priors import SRLConvolutionalNetwork, SRLDenseNetwork, SRLLinear
from .triplet import EmbeddingNet
from .models import *

# In case of importing into the SRL repository
try:
    from preprocessing.preprocess import getInputDim
# In case of importing material from modules.py into the external Robotics RL repository,
# consider the relative path to the package
except ImportError:
    from ..preprocessing.preprocess import getInputDim


class SRLModules(BaseForwardModel, BaseInverseModel, BaseRewardModel):
    def __init__(self, state_dim=2, action_dim=6, cuda=False, model_type="custom_cnn", losses=None,
                 inverse_model_type="linear"):
        """
        A model that can combine AE/VAE + Inverse + Forward + Reward models
        :param state_dim: (int)
        :param action_dim: (int)
        :param cuda: (bool)
        :param model_type: (str)
        :param losses: ([str])
        """
        self.model_type = model_type
        self.losses = losses
        BaseForwardModel.__init__(self)
        BaseInverseModel.__init__(self)
        BaseRewardModel.__init__(self)

        self.cuda = cuda

        self.initForwardNet(state_dim, action_dim)
        self.initInverseNet(state_dim, action_dim, model_type=inverse_model_type)
        self.initRewardNet(state_dim)

        # Architecture
        if model_type == "custom_cnn":
            if "autoencoder" in losses or "dae" in losses:
                self.model = CNNAutoEncoder(state_dim)
            elif "vae" in losses:
                self.model = CNNVAE(state_dim)
            else:
                # for losses not depending on specific architecture (supervised, inv, fwd..)
                self.model = CustomCNN(state_dim)

        elif model_type == "mlp":
            if "autoencoder" in losses or "dae" in losses:
                self.model = DenseAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            elif "vae" in losses:
                self.model = DenseVAE(input_dim=getInputDim(),
                                      state_dim=state_dim)
            else:
                # for losses not depending on specific architecture (supervised, inv, fwd..)
                self.model = SRLDenseNetwork(getInputDim(), state_dim, cuda=cuda)

        elif model_type == "linear":
            if "autoencoder" in losses or "dae" in losses:
                self.model = LinearAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            else:
                # for losses not depending on specific architecture (supervised, inv, fwd..)
                self.model = SRLLinear(input_dim=getInputDim(), state_dim=state_dim, cuda=cuda)

        elif model_type == "resnet":
            self.model = SRLConvolutionalNetwork(state_dim, cuda)

        if losses is not None and "triplet" in losses:
            # pretrained resnet18 with fixed weights
            self.model = EmbeddingNet(state_dim)

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.model.getStates(observations)

    def forward(self, x):
        if self.model_type == 'linear' or self.model_type == 'mlp':
            x = x.contiguous()
        return self.model(x)

    def encode(self, x):
        if "triplet" in self.losses:
            return self.model(x)
        else:
            raise NotImplementedError()

    def forwardTriplets(self, anchor, positive, negative):
        """
        Overriding the forward function in the case of Triplet loss
        anchor : anchor observations (th. Tensor)
        positive : positive observations (th. Tensor)
        negative : negative observations (th. Tensor)
        """
        return self.model(anchor), self.model(positive), self.model(negative)


class SRLModulesSplit(BaseForwardModel, BaseInverseModel, BaseRewardModel):
    def __init__(self, state_dim=2, action_dim=6, cuda=False, model_type="custom_cnn",
                 losses=None, split_index=-1, n_hidden_reward=16, inverse_model_type="linear"):
        """
        A model that can split representation, combining
        AE/VAE for the first split with Inverse + Forward in the second split
        Reward model is learned for all the dimensions
        :param state_dim: (int)
        :param action_dim: (int)
        :param cuda: (bool)
        :param model_type: (str)
        :param losses: ([str])
        :param split_index: (int or [int])) Number of dimensions for the different split
        :param n_hidden_reward: (int) Number of hidden units for the reward model
        """

        # TODO: rename split_index -> split_indices + change in RL repo
        if isinstance(split_index, int):
            assert split_index < state_dim, \
                "The second split must be of dim >= 1, consider increasing the state_dim or decreasing the split_index"
            split_indices = [split_index]
        else:
            # TODO: sanity check: split_indices ordered + < state_dim
            split_indices = split_index

        # Compute the number of dimensions for each method
        n_dimensions = [split_indices[0]]
        for i in range(len(split_indices) - 1):
            n_dimensions.append(split_indices[i + 1] - split_indices[i])

        n_dimensions.append(state_dim - split_indices[-1])

        assert sum(n_dimensions) == state_dim
        self.n_dimensions = n_dimensions

        self.model_type = model_type
        self.losses = losses

        BaseForwardModel.__init__(self)
        BaseInverseModel.__init__(self)
        BaseRewardModel.__init__(self)

        self.cuda = cuda
        self.state_dim = state_dim

        self.ae_index = 0
        # self.reward_index = 1
        self.inverse_index = len(split_indices)

        self.initForwardNet(self.state_dim, action_dim)
        self.initInverseNet(self.state_dim, action_dim, model_type=inverse_model_type)
        self.initRewardNet(self.state_dim, n_hidden=n_hidden_reward)

        # Architecture
        if model_type == "custom_cnn":
            if "autoencoder" in losses or "dae" in losses:
                self.model = CNNAutoEncoder(state_dim)
            elif "vae" in losses:
                self.model = CNNVAE(state_dim)
            else:
                self.model = CustomCNN(state_dim)

        elif model_type == "mlp":
            if "autoencoder" in losses or "dae" in losses:
                self.model = DenseAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            elif "vae" in losses:
                self.model = DenseVAE(input_dim=getInputDim(), state_dim=state_dim)
            else:
                self.model = SRLDenseNetwork(getInputDim(), state_dim, cuda=cuda)

        elif model_type == "linear":
            if "autoencoder" in losses:
                self.model = LinearAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            else:
                self.model = SRLLinear(input_dim=getInputDim(), state_dim=state_dim, cuda=cuda)

        elif model_type == "resnet":
            raise ValueError("Resnet not supported for autoencoders")

        if "triplet" in losses:
            raise ValueError("triplet not supported when splitting representation")

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.model.getStates(observations)

    def forward(self, x):
        if "autoencoder" in self.losses or "dae" in self.losses:
            return self.forwardAutoencoder(x)
        elif "vae" in self.losses:
            return self.forwardVAE(x)
        else:
            return self.model.forward(x)

    def detachSplit(self, tensor, index):
        """
        Detach splits from the graph,
        so no gradients are backpropagated
        for those splits part of the states
        :param tensor: (th.Tensor)
        :param index (int) position of the split not to detach
        :return: (th.Tensor)
        """
        tensors = []
        start_idx = 0
        for idx, n_dim in enumerate(self.n_dimensions):
            if idx != index:
                tensors.append(tensor[:, start_idx:start_idx + n_dim].detach())
            else:
                tensors.append(tensor[:, start_idx:start_idx + n_dim])
            start_idx += n_dim

        return th.cat(tensors, dim=1)

    def forwardVAE(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        mu, logvar = self.model.encode(x)
        z = self.model.reparameterize(self.detachSplit(mu, index=0), self.detachSplit(logvar, index=0))
        decoded = self.model.decode(z).view(input_shape)
        return decoded, self.detachSplit(mu, index=0), self.detachSplit(logvar, index=0)

    def forwardAutoencoder(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        encoded = self.model.encode(x)
        decoded = self.model.decode(self.detachSplit(encoded, index=0)).view(input_shape)
        return encoded, decoded

    def inverseModel(self, state, next_state):
        """
        Predict action given current state and next state
        :param state: (th.Tensor)
        :param next_state: (th.Tensor)
        :return: probability of each action
        """
        return self.inverse_net(th.cat((self.detachSplit(state, index=self.inverse_index),
                                        self.detachSplit(next_state, index=self.inverse_index)), dim=1))

    def forwardModel(self, state, action):
        """
        Predict next state given current state and action
        :param state: (th.Tensor)
        :param action: (th Tensor)
        :return: (th.Tensor)
        """
        # Predict the delta between the next state and current state
        concat = th.cat((self.detachSplit(state, index=2), encodeOneHot(action, self.action_dim)), dim=1)
        return self.detachSplit(state, index=2) + self.forward_net(concat)

    def rewardModel(self, state, next_state):
        """
        Predict reward given current state and next state
        :param state: (th.Tensor)
        :param action: (th Tensor)
        :return: (th.Tensor)
        """
        return self.reward_net(th.cat((self.detachSplit(state, index=1),
                                       self.detachSplit(next_state, index=1)), dim=1))
