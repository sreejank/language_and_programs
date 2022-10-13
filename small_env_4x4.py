"""
Original training enviornment for meta-rl agent for the GSP task distribution used in Kumar et al. 2022. We used their enviornment (but turned of training on the control distribution).  
"""

import gym
from gym.utils import seeding
from PIL import Image as PILImage
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
from itertools import product  
from itertools import permutations 
import pickle 




class BattleshipEnv(gym.Env):  
	
	reward_range = (-float('inf'), float('inf'))
	metadata = {'render.modes': ['human', 'rgb_array'],'video.frames_per_second' : 3}
	
	def __init__(self,rules='chain',n_board=4,hold_out=0,permute=0,pretrain=0,render_img=0):
		
		self.viewer = None
		self.seed()
		action_converter=[]
		for i in range(n_board):
			for j in range(n_board): 
				action_converter.append((i,j))
		self.action_converter=np.asarray(action_converter)
		self.n_board=n_board

		self.hold_out=hold_out
		self.rules=rules
		self.pretrain=pretrain 
		self.render_img=render_img

		
		

		if hold_out==-1:
			self.heldout=np.load('data/'+self.rules+'_sample.npy').reshape((-1,4,4))

			self.maze_idx=0 
			self.maze=np.reshape(self.heldout[self.maze_idx],(4,4))
			start=np.load('data/'+self.rules+'_sample_starts.npy')[self.maze_idx]

		else:
			if hold_out>0:
				heldout=np.load('data/'+self.rules+'_sample.npy').reshape((-1,16))

				self.heldout=set([tuple(x) for x in heldout]) 
			
			
			self.full_distribution=np.load('data/'+self.rules+"_full.npy")
			self.full_probs=np.load('data/'+self.rules+"_full_probs.npy")
			if hold_out>0:
				for board in heldout:
					boarda=np.asarray(board)
					for i in range(self.full_distribution.shape[0]):
						if np.sum(self.full_distribution[i]==boarda)==16:
							self.full_probs[i]=0.0
							break
						elif np.sum(self.full_distribution[i])==0:
							self.full_probs[i]=0.0   

				for i in range(self.full_distribution.shape[0]):
					if np.sum(self.full_distribution[i])==0:
						self.full_probs[i]=0.0   
				self.full_probs=self.full_probs/np.sum(self.full_probs)
			elif hold_out==0:
				self.heldout=set() 

			
			if 'null' not in self.rules:
				r_idx=np.random.choice(np.arange(self.full_distribution.shape[0]),p=self.full_probs)
				grid=self.full_distribution[r_idx].reshape((4,4))
				reds=np.vstack(np.where(grid==1)).T 
				start=reds[np.random.choice(np.arange(reds.shape[0]))]
				gen=(grid,start) 
			else:
				raise Exception('No training on control distributions.')
				"""
				network.load_state_dict(torch.load("data/"+self.rules+"_fc_generator.pt"))
				network.eval()
				self.size_buffer=1000 
				self.gibbs_buffer=batch_gibbs(S=4,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,4,4))
				self.gibbs_idx=0 

				grid=self.null_sample()
				reds=np.vstack(np.where(grid==1)).T 
				start=reds[np.random.choice(np.arange(reds.shape[0]))]
				gen=(grid,start) 
				"""
			while tuple(gen[0].flatten()) in self.heldout:
				if 'null' not in self.rules:
					r_idx=np.random.choice(np.arange(self.full_distribution.shape[0]),p=self.full_probs)
					grid=self.full_distribution[r_idx].reshape((4,4))
					reds=np.vstack(np.where(grid==1)).T 
					start=reds[np.random.choice(np.arange(reds.shape[0]))]
					gen=(grid,start) 
				else:
					raise Exception('No training on control distributions.')
					"""
					network.load_state_dict(torch.load("data/"+self.rules+"_fc_generator.pt"))
					network.eval()
					self.size_buffer=1000 
					self.gibbs_buffer=batch_gibbs(S=4,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,4,4))
					self.gibbs_idx=0 

					grid=self.null_sample()
					reds=np.vstack(np.where(grid==1)).T 
					start=reds[np.random.choice(np.arange(reds.shape[0]))]
					gen=(grid,start) 
					"""



			self.maze=grid

		self.board=np.ones(self.maze.shape)*-1
		self.current_position=start 
		self.board[self.current_position[0],self.current_position[1]]=1
		self.num_hits=0
		self.self_hits={}
		if self.render_img:
			self.observation_space = Box(low=-1, high=1, shape=(n_board*n_board*1+n_board*n_board+1,), dtype=np.int8)
		else:
			self.observation_space = Box(low=-1, high=1, shape=(n_board*n_board+n_board*n_board+1,), dtype=np.int8)
		self.action_space = Discrete(np.prod(self.maze.shape))
		self.nA=n_board*n_board

		self.prev_reward=0
		self.prev_action=np.zeros((self.nA,))

		self.valid_actions=[1 for _ in range(self.nA)]
	"""
	def null_sample(self):
		if self.gibbs_idx>=self.size_buffer:
			self.gibbs_buffer=batch_gibbs(S=4,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,4,4))
			self.gibbs_idx=0
		grid=self.gibbs_buffer[self.gibbs_idx]
		self.gibbs_idx=self.gibbs_idx+1 
		while np.sum(grid)<3:
			if self.gibbs_idx>=self.size_buffer:
				self.gibbs_buffer=batch_gibbs(S=4,numSweeps=20,batch_size=self.size_buffer,network=network).reshape((-1,4,4))
				self.gibbs_idx=0
			grid=self.gibbs_buffer[self.gibbs_idx]
			self.gibbs_idx=self.gibbs_idx+1 
		return grid 
	"""

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
		
	def step(self, action):
		#print(self.tot_solves)
		#print(action,self.valid_actions,self.valid_actions[action]) 
		prev_position=self.current_position
		self.current_position=self.action_converter[action]
		reward=0
		
		if self.board[self.current_position[0],self.current_position[1]]==-1:
			if self.maze[self.current_position[0],self.current_position[1]]==1:
				self.board[self.current_position[0],self.current_position[1]]=1
				self.num_hits+=1
				reward=1
			else:
				self.board[self.current_position[0],self.current_position[1]]=0
				reward=-1
		else:
			reward=-4
			if (self.current_position[0],self.current_position[1]) not in self.self_hits.keys():
				self.self_hits[(self.current_position[0],self.current_position[1])]=1
			else:
				self.self_hits[(self.current_position[0],self.current_position[1])]+=1
			
			
						
			
		if self._is_goal():
			reward=+5 
			done = True
			if self.hold_out==-1:
				self.maze_idx+=1
		else:
			done = False

		p_action=self.prev_action
		p_reward=self.prev_reward
		self.prev_action=np.zeros((self.nA,))
		self.prev_action[action]=1
		self.prev_reward=reward 

		if self.render_img:
			obs=self.get_image().flatten()
		else:
			obs=self.board.flatten()

		obs_array=np.concatenate((obs,p_action,[p_reward]))

		self.valid_actions=[0 for _ in range(self.nA)]
		for i in range(self.nA):
			pos=self.action_converter[i]
			if self.board[pos[0],pos[1]]==0:
				self.valid_actions[i]=1
		
		
		return obs_array, reward, done, {'valid_actions':self.valid_actions}
	
	def get_action_mask(self):
		return self.valid_actions
	def _is_goal(self):
		return np.sum(self.board==1)==np.sum(self.maze==1)
	
	def get_image(self):
		"""
		img=np.empty((28,28, 1), dtype=np.int8) 
		#fills=[[0,0,255],[255,0,0],[255,255,255]]
		#fills=[0,128,255]
		for r in range(self.board.shape[0]):
			for c in range(self.board.shape[1]):
				#fill=fills[self.board[r,c].astype('int')]
				img[4*r:4*r+4,4*c:4*c+4]=self.board[r,c]
		return img  
		"""
		return np.reshape(self.board,(4,4,1))
		
	
	def set_task(self,task):
		self.maze = task
		self.board=np.zeros(self.maze.shape)
		self.current_position=[np.random.choice(range(self.maze.shape[0])),np.random.choice(self.maze.shape[1])]

		self.num_hits=0
		self.self_hits={}
		return self.board.flatten()
	
	def reset(self):
		if self.hold_out==-1:
			self.maze=np.reshape(self.heldout[self.maze_idx%len(self.heldout)],(4,4))
			start=np.load('data/'+self.rules+'_sample_starts.npy')[self.maze_idx%len(self.heldout)] 
		else:
			
			if 'null' not in self.rules:
				r_idx=np.random.choice(np.arange(self.full_distribution.shape[0]),p=self.full_probs)
				grid=self.full_distribution[r_idx].reshape((4,4))
				reds=np.vstack(np.where(grid==1)).T 
				start=reds[np.random.choice(np.arange(reds.shape[0]))]
				gen=(grid,start) 
			else:
				raise Exception('No training on control distributions.')
				#grid=self.null_sample()
				#reds=np.vstack(np.where(grid==1)).T 
				#start=reds[np.random.choice(np.arange(reds.shape[0]))]
				#gen=(grid,start)
			while tuple(gen[0].flatten()) in self.heldout:
				if 'null' not in self.rules:
					r_idx=np.random.choice(np.arange(self.full_distribution.shape[0]),p=self.full_probs)
					grid=self.full_distribution[r_idx].reshape((4,4))
					reds=np.vstack(np.where(grid==1)).T 
					start=reds[np.random.choice(np.arange(reds.shape[0]))]
					gen=(grid,start) 
				else:
					raise Exception('No training on control distributions.')
					#grid=self.null_sample()
					#reds=np.vstack(np.where(grid==1)).T 
					#start=reds[np.random.choice(np.arange(reds.shape[0]))]
					#gen=(grid,start)


			self.maze=grid 

		self.board=np.ones(self.maze.shape)*-1
		self.current_position=start 
		self.board[self.current_position[0],self.current_position[1]]=1

		self.num_hits=0
		self.self_hits={}
		if self.render_img:
			obs=self.get_image().flatten()
		else:
			obs=self.board.flatten()


		obs_array=np.concatenate((obs,self.prev_action,[self.prev_reward]))
		self.valid_actions=[1 for _ in range(self.nA)] 
		return obs_array
	
	def render(self, mode='human', max_width=500): 
		img = self.get_image()
		img = np.asarray(img).astype(np.uint8)
		img_height, img_width = img.shape[:2]
		ratio = max_width/img_width
		img = PILImage.fromarray(img).resize([int(ratio*img_width), int(ratio*img_height)])
		img = np.asarray(img)
		if mode == 'rgb_array':
			return img
		elif mode == 'human':
			from gym.envs.classic_control.rendering import SimpleImageViewer
			if self.viewer is None:
				self.viewer = SimpleImageViewer()
			self.viewer.imshow(img)  
			
			return self.viewer.isopen
	def close(self): 
		if self.viewer is not None:
			self.viewer.close()
			self.viewer = None
	
def register_small_env(env_id,rules,n_board=4,hold_out=0,permute=0,max_episode_steps=25,pretrain=0,render_img=0):  
	gym.envs.register(id=env_id, entry_point=BattleshipEnv, max_episode_steps=max_episode_steps,kwargs={'rules':rules,'n_board':n_board,'hold_out':hold_out,'permute':permute,'pretrain':pretrain,'render_img':render_img})  