import numpy as np
from torch.autograd import Variable
import torch
import copy
from operator import itemgetter
"""MCTS获取先验概率"""
def get_probs(board):
    act_prob = np.ones(len(board.empty_loc))/len(board.empty_loc)
    return zip(board.empty_loc,act_prob)

def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(board.empty_loc))
    return zip(board.empty_loc, action_probs)


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

"""蒙特卡洛-树"""
class MctsTree(object):
    def __init__(self, parent, prob):
        self.parent = parent
        self.n_ = 0
        self.q_ = 0
        self.c_ = 5
        self.uct_ = 0
        self.p_ = prob
        self.children = {}

    def get_uct(self):
        self.u_ = (self.c_ * self.p_ * np.sqrt(self.parent.n_) / (1 + self.n_))
        uct = self.u_ + self.q_
        return uct

    def select(self):
        return max(self.children.items(), key=lambda act: act[1].get_uct())

    def expend(self, empty_loc):
        for act, prob in empty_loc:
            if act not in self.children:
                self.children[act] = MctsTree(self, prob)

    def update(self, value):
        self.n_ += 1
        self.q_ += 1 * (value - self.q_) / self.n_

    def update_r(self, value):
        if self.parent:
            self.parent.update_r(-value)
        self.update(value)


"""蒙特卡洛树搜索，开始搜索，此处与传统蒙特卡洛树搜索略有不同，搜索到子节点不进行模拟，而是直接搜索到棋局结束代替模拟"""
class MctsPlay(object):
    def __init__(self, prob_init=1,n_play_times=100):
        self.prob_init = prob_init
        self.root = MctsTree(None, self.prob_init)
        self.cnt = 0
        self.n_play_times = n_play_times
######改良版本的mcts更强，起始100轮对弈基本AI玩不过
    # def start_play_to_win(self, board):
    #     node = self.root
    #     while not board.is_end()[0]:
    #         if node.children == {}:
    #             act_prob = get_probs(board)
    #             node.expend(act_prob)
    #         if not node.children == {}:
    #             act, node = node.select()
    #             board.move(act)
    #     node.update_r(1)

#####原始版本
    def start_play_to_win(self, board):
        node = self.root
        while(1):
            if node.children == {}:
                break
            # Greedily select next move.
            action, node = node.select()
            board.move(action)

        action_probs = get_probs(board)#随机选择一个节点
        # Check for end of game
        end, winner = board.is_end()
        if not end:
            node.expend(action_probs)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(board)
        # Update value and visit count of nodes in this traversal.
        node.update_r(-leaf_value)#感觉这里应该传入leaf_value


    def _evaluate_rollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.is_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1
#####################################



    def start_play(self, board):
        probs = np.zeros(board.width * board.height)

        for i in range(self.n_play_times):
            board_copy = copy.deepcopy(board)
            self.start_play_to_win(board_copy)

        act = max(self.root.children.items(), key=lambda act: act[1].n_)[0]

        ###########+ai与mcts对战数据
        act_visits = [(act, node.n_)  #################用该网络更新所有节点的下一步走子，
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / 1e-3 * np.log(np.array(visits) + 1e-10))
        probs[list(acts)] = act_probs
        #####################
        # return act
        return act,probs

    def reset_mcts(self, act):
        if act in self.root.children:
            self.root = self.root.children[act]
            self.root.parent = None
        else:
            self.root = MctsTree(None, self.prob_init)


"""零狗蒙特卡洛树搜索"""
class Ce_MctsPlay(object):
    def __init__(self, pv_net,prob_init=1, n_play_times=100,cuda_enable=False,return_prob=1, is_selfplay=1):
        self.prob_init = prob_init
        self.root = MctsTree(None, self.prob_init)
        self.cuda_enable = cuda_enable
        self.n_play_times = n_play_times
        self.pv_net = pv_net
        self.return_prob = return_prob
        self.is_selfplay = is_selfplay

    def start_play_to_win(self, board,pv_net):
        node = self.root
        while (1):
            if node.children == {}:
                break
            action, node = node.select()
            board.move(action)
        action_probs, leaf_value = pv_net(board)
        end, winner = board.is_end()
        if not end:
            node.expend(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == board.get_current_player() else -1.0
                )
        node.update_r(-leaf_value)

    def start_play(self, board):
        probs = np.zeros(board.width * board.height)
        for i in range(self.n_play_times):
            board_copy = copy.deepcopy(board)
            self.start_play_to_win(board_copy,self.pv_net)
        act_visits = [(act, node.n_)  #################用该网络更新所有节点的下一步走子，
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / 1e-3 * np.log(np.array(visits) + 1e-10))
        probs[list(acts)] = act_probs
        if self.is_selfplay:
            move = np.random.choice(acts, p=0.75 * act_probs +
            0.25 * np.random.dirichlet(0.3 * np.ones(len(act_probs))))  ################网络更新的策略+狄利克雷
            self.reset_mcts(move)
        else:
            move = np.random.choice(acts, p=act_probs)
            self.reset_mcts(-1)
        if self.return_prob:
            return move, probs
        else:
            return move

    def reset_mcts(self, act):
        if act in self.root.children:
            self.root = self.root.children[act]
            self.root.parent = None
        else:
            self.root = MctsTree(None, self.prob_init)


"""自奕网络"""
class Self_Mcts(object):
    def __init__(self,board):
        self.board = board
        self.board.init_board()

    def start_play_to_win(self,pv_net):
        while not self.board.is_end()[0]:
            act_prob = pv_net(self.board)[0]
            act,prob = zip(*act_prob)
            move = act[np.argmax(prob)]
            self.board.move(move)
        end,winner = self.board.is_end()
        self.board.init_board()
        return (True if winner==1 else False)








