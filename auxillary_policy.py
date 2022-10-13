
"""
Setup for agent training code w/ auxillary loss.
"""

from typing import Dict, List, Tuple, Type, Union, Optional, Any 
import gym
import torch
from torch import nn
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.distributions import Distribution 

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.latent = nn.Linear(64, 16)
        self.fc_recover = nn.Linear(16, 16)
        self.act_fn=nn.ReLU() 
        self.fc_recover2 = nn.Linear(16,16)


        
    def forward(self, x):
        x=x.reshape((-1,1,4,4))
        x=self.act_fn(self.latent(self.cnn(x)))
        recovered=self.fc_recover2(self.act_fn(self.fc_recover(x)))
        return recovered 


class LanguageEncoder(nn.Module):
    def __init__(self):
        super(LanguageEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.latent = nn.Linear(64, 16)
        self.fc_language = nn.Linear(16, 768)
        self.fc_language2= nn.Linear(768,768) 
        self.act_fn=nn.ReLU()

    def forward(self, x):
        x=x.reshape((-1,1,4,4))
        x=self.act_fn(self.latent(self.cnn(x)))
        language=self.fc_language2(self.act_fn(self.fc_language(x)))
        return language



class ProgramEncoder(nn.Module):
    def __init__(self):
        super(ProgramEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.latent = nn.Linear(64, 16)
        self.fc_program = nn.Linear(16, 64)
        self.fc_program2= nn.Linear(64,64) 
        self.act_fn=nn.ReLU()

    def forward(self, x):
        x=x.reshape((-1,1,4,4))
        x=self.act_fn(self.latent(self.cnn(x)))
        program=self.fc_program2(self.act_fn(self.fc_program(x)))
        return program 



class DualFeatureExtractor(BaseFeaturesExtractor):
    

    def __init__(self, observation_space: gym.spaces.Box):
       
        super(DualFeatureExtractor, self).__init__(observation_space, features_dim=33)

        self.visual_encoder=LanguageEncoder()



    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observation_board=observations[:,768:]
        visual_input=torch.reshape(observation_board[:,:16],(-1,1,4,4))
        prev_output=observation_board[:,16:] 
        visual_output=self.visual_encoder.act_fn(self.visual_encoder.latent(self.visual_encoder.cnn(visual_input)))
        total_output=torch.cat([visual_output,prev_output],dim=1)

        predicted_language=self.visual_encoder.fc_language2(self.visual_encoder.act_fn(self.visual_encoder.fc_language(visual_output))) 
        return total_output, predicted_language 
    




class ProgramDualFeatureExtractor(BaseFeaturesExtractor):
    

    def __init__(self, observation_space: gym.spaces.Box):

        super(ProgramDualFeatureExtractor, self).__init__(observation_space, features_dim=33)

        self.visual_encoder=ProgramEncoder()



    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observation_board=observations[:,64:]
        visual_input=torch.reshape(observation_board[:,:16],(-1,1,4,4))
        prev_output=observation_board[:,16:] 
        visual_output=self.visual_encoder.act_fn(self.visual_encoder.latent(self.visual_encoder.cnn(visual_input)))
        total_output=torch.cat([visual_output,prev_output],dim=1)

        predicted_program=self.visual_encoder.fc_program2(self.visual_encoder.act_fn(self.visual_encoder.fc_program(visual_output))) 
        return total_output, predicted_program 


class AutoDualFeatureExtractor(BaseFeaturesExtractor): 
    

    def __init__(self, observation_space: gym.spaces.Box):

        super(AutoDualFeatureExtractor, self).__init__(observation_space, features_dim=33)

        self.visual_encoder=AutoEncoder()


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observation_board=observations[:,16:] 
        visual_input=torch.reshape(observation_board[:,:16],(-1,1,4,4))
        prev_output=observation_board[:,16:] 
        visual_output=self.visual_encoder.act_fn(self.visual_encoder.latent(self.visual_encoder.cnn(visual_input)))
        total_output=torch.cat([visual_output,prev_output],dim=1)

        predicted_language=self.visual_encoder.fc_recover2(self.visual_encoder.act_fn(self.visual_encoder.fc_recover(visual_output))) 
        #print(predicted_language.shape)
        return total_output, predicted_language 

class DualFeatureRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = DualFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 64,
        n_lstm_layers: int = 1,
        enable_critic_lstm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            enable_critic_lstm,
        )
    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features,features_lang = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Tuple[Distribution, Tuple[torch.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.
        :param obs:
        :return: the action distribution and new hidden states.
        """
        features,features_lang = self.extract_features(obs)
        latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        :param obs:
        :return: the estimated values.
        """
        features,features_lang  = self.extract_features(obs)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: torch.tensor,
        actions: torch.Tensor,
        lstm_states: RNNStates, 
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features,features_lang = self.extract_features(obs)
        latent_pi, _ = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy(), features_lang 


class ProgramDualFeatureRecurrentActorCriticPolicy(RecurrentActorCriticPolicy): 
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = ProgramDualFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 64,
        n_lstm_layers: int = 1,
        enable_critic_lstm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            enable_critic_lstm,
        )
    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features,features_lang = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Tuple[Distribution, Tuple[torch.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.
        :param obs:
        :return: the action distribution and new hidden states.
        """
        features,features_lang = self.extract_features(obs)
        latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        :param obs:
        :return: the estimated values.
        """
        features,features_lang  = self.extract_features(obs)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: torch.tensor,
        actions: torch.Tensor,
        lstm_states: RNNStates, 
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features,features_lang = self.extract_features(obs)
        latent_pi, _ = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy(), features_lang 



class AutoDualFeatureRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """
    MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``torch.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = AutoDualFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lstm_hidden_size: int = 64,
        n_lstm_layers: int = 1,
        enable_critic_lstm: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size,
            n_lstm_layers,
            enable_critic_lstm,
        )
    def forward(
        self,
        obs: torch.Tensor,
        lstm_states: RNNStates,
        episode_starts: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RNNStates]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features,features_lang = self.extract_features(obs)
        # latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, lstm_states_pi = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Re-use LSTM features but do not backpropagate
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def get_distribution(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Tuple[Distribution, Tuple[torch.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.
        :param obs:
        :return: the action distribution and new hidden states.
        """
        features,features_lang = self.extract_features(obs)
        latent_pi, lstm_states = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        return self._get_action_dist_from_latent(latent_pi), lstm_states

    def predict_values(
        self,
        obs: torch.Tensor,
        lstm_states: Tuple[torch.Tensor, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        :param obs:
        :return: the estimated values.
        """
        features,features_lang  = self.extract_features(obs)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            # Use LSTM from the actor
            latent_pi, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        return self.value_net(latent_vf)

    def evaluate_actions(
        self,
        obs: torch.tensor,
        actions: torch.Tensor,
        lstm_states: RNNStates, 
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features,features_lang = self.extract_features(obs)
        latent_pi, _ = self._process_sequence(features, lstm_states.pi, episode_starts, self.lstm_actor)

        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy(), features_lang 
