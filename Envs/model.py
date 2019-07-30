import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models
EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(256,256)
		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		self.fcs2 = nn.Linear(256,128)
		self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

		self.fca1 = nn.Linear(action_dim,128)
		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,1)
		self.fc3.weight.data.uniform_(-EPS,EPS)

		# self.cnn = CNN().cuda()
		self.cnn = nn.Sequential(
			models.resnet34(pretrained=True,progress=True).cuda (),
			nn.Linear (in_features=1000, out_features=512),
			nn.ReLU (),
			nn.Linear (in_features=512, out_features=256)
		).cuda()


	def forward(self, state, action):
		state = torch.tensor (state)
		try:
			state = self.cnn (state.transpose (1, 3).float ())
		except:
			state = state.squeeze ().unsqueeze (0)
			state = self.cnn (state.transpose (1, 3).float ())
		state = state.squeeze ()

		# print("critic state shape:",state.shape,"   action shape:",action.shape)
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		x = torch.cat((s2,a1),dim=1)

		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim, action_lim):
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = action_lim

		self.fc1 = nn.Linear(256,256)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,64)
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

		self.fc4 = nn.Linear(64,action_dim)
		self.fc4.weight.data.uniform_(-EPS,EPS)

		# self.cnn = CNN().cuda()
		self.cnn = nn.Sequential (
			models.resnet18 (pretrained=True, progress=True).cuda (),
			nn.Linear (in_features=1000, out_features=512),
			nn.ReLU (),
			nn.Linear (in_features=512, out_features=256)
		).cuda ()

	def forward(self, state):
		state = torch.tensor (state)
		try:
			state = self.cnn (state.transpose (1, 3).float ())
		except:
			state = state.squeeze ().unsqueeze (0)
			state = self.cnn (state.transpose (1, 3).float ())
		state = state.squeeze ()

		# print ("actor state shape/:", state.shape)
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		action = F.tanh(self.fc4(x))

		action = action * self.action_lim

		return action


class CNN(nn.Module):
	def __init__ (self):
		super (CNN, self).__init__ ()
		self.resnet = models.resnet18 (pretrained=True)
		self.fc = nn.Sequential (
			nn.Linear (in_features=1000, out_features=256),
			nn.ReLU (),
			nn.Linear (in_features=256, out_features=64),
			nn.ReLU (),
			nn.Linear (in_features=64, out_features=32)
		)

	def forward (self, input):
		x = self.resnet(input)
		x = self.fc (x)
		return x