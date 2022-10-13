
"""
Code to train PPO agent without grounding loss. 
"""
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn 
import gym 
import torch 
import pickle 
from small_env_4x4 import * 
import pickle 
import sys 

rules=sys.argv[1] 
version='orig'  
run=sys.argv[2] 

register_small_env('small-v0','gsp_4x4',hold_out=10,pretrain=0)   
register_small_env('test-v0',rules,hold_out=-1,pretrain=0,max_episode_steps=120)     


hyperparams_dict=pickle.load(open('data/hyperparams_nogrounding.pkl','rb'))   


batch_size=hyperparams_dict['batch_size']
n_steps=hyperparams_dict['n_steps']
gamma=hyperparams_dict['gamma']
learning_rate=hyperparams_dict['learning_rate']
lr_schedule=hyperparams_dict['lr_schedule']
ent_coef=hyperparams_dict['ent_coef']
vf_coef=hyperparams_dict['vf_coef']
clip_range=hyperparams_dict['clip_range']
n_epochs=hyperparams_dict['n_epochs']
gae_lambda=hyperparams_dict['gae_lambda']
max_grad_norm=hyperparams_dict['max_grad_norm']
activation_fn=hyperparams_dict['activation_fn']
activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]
n_lstm=120



def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

if lr_schedule=='linear':
    learning_rate=linear_schedule(learning_rate)








class CNNSkip(BaseFeaturesExtractor): 
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 33):

        super(CNNSkip, self).__init__(observation_space, features_dim)
        n_input_channels = 1
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        self.linear = nn.Sequential(nn.Linear(64, 16), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        visual_input=torch.reshape(observations[:,:16],(-1,1,4,4))
        prev_output=observations[:,16:]
        visual_output=self.linear(self.cnn(visual_input))
        total_output=torch.cat([visual_output,prev_output],dim=1)

        return total_output
pk=dict(features_extractor_class=CNNSkip,n_lstm_layers=1,lstm_hidden_size=n_lstm)
kwargs={
        "policy": "CnnLstmPolicy",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda, 
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": pk, 
        'verbose':1
    }

action_converter=[]
for i in range(7):
        for j in range(7):
                action_converter.append((i,j)) 

if __name__=='__main__':
    num_episodes=1000000

    env=make_vec_env('small-v0',n_envs=1,vec_env_cls=SubprocVecEnv) 
    kwargs.update({'env':env})
    model=RecurrentPPO(**kwargs) 
    


    print("Start")
    model.learn(num_episodes,log_interval=1)    
    print("Saving") 

    model.save("models/ppo_"+rules+"_"+version+"_"+str(run)+"_metalearning.zip")     
    print("loading")     

    obs=env.reset() 
    state=None
    done = [False for _ in range(env.num_envs)]

    reward_buffer=[]
    evals=[]
    num_evals=25
    tot_rewards=[]
    print("Begin")

    env.close()
    env=make_vec_env('test-v0',n_envs=1,vec_env_cls=SubprocVecEnv) 
    obs=env.reset() 
    state=None
    done = [False for _ in range(env.num_envs)]
    episode_start=np.asarray([True for _ in range(env.num_envs)])

    
    num_evals=25
    print("Begin")
    raw_performance=np.zeros((15,25))
    mean_performance=[]
    raw_choices_total=[]
    test_boards=np.load('data/'+rules+"_sample.npy") 
    for i in range(15):
        reward_buffer=[]
        evals=[]
        tot_rewards=[]
        raw_choices_buffer=[]
        tot_raw_choices=[]
        while len(tot_rewards)<num_evals: 
            action, state = model.predict(obs, state=state, episode_start=episode_start)
            if episode_start[0]:
                episode_start[0]=False 

            obs, reward , done, _ = env.step(action)
            reward_buffer.append(reward[0])
            raw_choices_buffer.append(action_converter[action[0]])

            if done[0]: 
                state=None 
                reward_array=np.asarray(reward_buffer)
                raw_choices_array=raw_choices_buffer[:]
                reward_buffer=[]
                raw_choices_buffer=[]
                episode_start[0]=True 



                if reward[0]==5: 
                    print("Finished")
                    raw_performance[i,len(tot_rewards)]=np.sum(reward_array==-1)
                    tot_rewards.append(np.sum(reward_array==-1))
                    tot_raw_choices.append(raw_choices_array)
                else:
                    print("Didnt finish")
                    raw_performance[i,len(tot_rewards)]=16-test_boards[len(tot_rewards)].sum()
                    tot_rewards.append(16-test_boards[len(tot_rewards)].sum())
                    tot_raw_choices.append(raw_choices_array) 

        mean_performance.append(np.mean(tot_rewards))
        raw_choices_total.append(tot_raw_choices)  
    tot_rewards=np.asarray(mean_performance) 
    np.save('data/raw_choices_ppo_orig_agent_'+str(rules)+"_"+str(run)+'.npy',np.asarray(raw_choices_total))  
    np.save('data/raw_performance_ppo_orig_agent_'+str(rules)+"_"+str(run)+".npy",raw_performance)     

 
     
