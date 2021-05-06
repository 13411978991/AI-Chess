import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from torch import optim
from torch import nn
from torch.autograd import Variable
import numpy as np

"""创建神经网络"""
class Model(nn.Module):
    def __init__(self, board_width, board_height):
        super(Model, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act),dim=1)
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class Model_train():
    def __init__(self,w,h,model_file=None,cuda_enable=False):
        self.model = (Model(w,h).cuda() if cuda_enable else Model(w,h))
        self.cuda_enable = cuda_enable
        if model_file:
            self.model.load_state_dict(torch.load(model_file))
        self.data = []

    def pv_net(self,board):
        legal_act = board.empty_loc
        state = np.ascontiguousarray(board.get_state())
        if self.cuda_enable:
            act_probs_org, value = self.model(Variable(torch.from_numpy(state).unsqueeze(0)).cuda().float())
            act_probs_org = np.exp(act_probs_org.data.cpu().numpy().flatten())
        else:
            act_probs_org, value = self.model(Variable(torch.from_numpy(state).unsqueeze(0)).float())
            act_probs_org = np.exp(act_probs_org.data.numpy().flatten())
        act_probs_new = zip(legal_act, act_probs_org[legal_act])
        value = value.item()
        return act_probs_new, value

    def pv(self):
        state = self.data[2]
        if self.cuda_enable:
            act_probs_org, value = self.model(state)
            act_probs_org = np.exp(act_probs_org.data.cpu().numpy())
        else:
            act_probs_org, value = self.model(state)
            act_probs_org = np.exp(act_probs_org.data.numpy())
        value = value.data.numpy()
        return act_probs_org, value

    def set_optim(self,lr,weight_decay):
        self.optimer = optim.Adam(params=self.model.parameters(),lr=lr,weight_decay=weight_decay)

    def unzip_data(self,data_loder):
        winner_list,probs_list,state_list = [],[],[]
        for item in data_loder:
            winner_list.append(item[0])
            probs_list.append(item[1])
            state_list.append(item[2])
        if self.cuda_enable:
            winner_list,probs_list,state_list = Variable(torch.FloatTensor(winner_list).cuda()),
            Variable(torch.FloatTensor(probs_list).cuda()),Variable(torch.FloatTensor(state_list).cuda())
        else:
            winner_list,probs_list,state_list = Variable(torch.FloatTensor(winner_list)),\
            Variable(torch.FloatTensor(probs_list)),Variable(torch.FloatTensor(state_list))
        self.data = (winner_list,probs_list,state_list)

    def train_step(self):
        # print(data_loder)
        winner_list,probs_list,state_list = self.data
        self.optimer.zero_grad()
        act_probs,val_probs = self.model(state_list)
        val_loss = F.mse_loss(val_probs.view(-1),winner_list)
        act_loss = -torch.mean(torch.sum(probs_list*act_probs, 1))
        loss = val_loss + act_loss
        loss.backward()
        self.optimer.step()
        return loss
