from Model import Model_train
from Board import Board, Game
from Mctsplay import Ce_MctsPlay, MctsPlay, Self_Mcts
import numpy as np
import copy
import random
import torch
from tqdm import tqdm
from collections import deque


class train_net():
    def __init__(self, model_file=None):
        self.width = 3
        self.height = 3
        self.n = 3
        self.board = Board(self.width, self.height, self.n)
        self.board.init_board()
        self.max_mcts_playtimes = 5000
        self.init_mcts_playtimes = 100
        self.lr = 2e-4
        self.l2 = 1e-4
        self.cuda_enable = torch.cuda.is_available()
        self.playtimes = 10  ##与机器对战多少盘
        self.n_play_times = 100
        self.batch_size = 512
        self.epoch = 20000
        self.model = Model_train(self.width, self.height, model_file, self.cuda_enable)
        self.model.set_optim(self.lr, self.l2)
        self.mcts_player = MctsPlay()
        self.play_data = deque(maxlen=10000)
        self.best_ratio = 0
        self.dir_cnt = 0
        self.max_epo = 5  # 控制每次训练最大
        """添加kl散度的内容"""
        self.kl_targ = 0.04
        self.lr_multiplier = 1.0
        """添加自奕"""
        self.self_mcts = Self_Mcts(self.board)

    def get_equi_data(self, play_data):  ##########get等价棋谱
        extend_data = []

        for item in play_data:
            winner, mcts_porb, state = item
            # print('winner', winner)
            # print('probs_list', mcts_porb)
            # print('state_list', state)
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.width, self.height)), i)
                extend_data.append((winner, np.flipud(equi_mcts_prob).flatten(), equi_state
                                    ))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((winner, np.flipud(equi_mcts_prob).flatten(), equi_state
                                    ))
        return extend_data

    def collect_data(self,mcts_data=None):
        if mcts_data:
            extend_data = self.get_equi_data(mcts_data)
            self.play_data.extend(extend_data)
            print('自奕扩张了{}个数据'.format(len(extend_data)))

        else:
            extend_data = self.collect_selfplay_data(self.board)
            extend_data = self.get_equi_data(extend_data)
            self.play_data.extend(extend_data)
            while 1:
                if len(self.play_data) < self.batch_size:
                    extend_data = self.collect_selfplay_data(self.board)
                    extend_data = self.get_equi_data(extend_data)
                    self.play_data.extend(extend_data)
                else:
                    data = random.sample(self.play_data, self.batch_size)
                    print('playdata len:', len(self.play_data))
                    return data

    def collect_selfplay_data(self, board):
        player_list, probs_list, state_list = [], [], []
        board_copy = copy.deepcopy(board)
        self.ce_model = Ce_MctsPlay(pv_net=self.model.pv_net, n_play_times=self.n_play_times,
                                    cuda_enable=self.cuda_enable)
        while not board_copy.is_end()[0]:
            state = board_copy.get_state()
            state_list.append(state)
            move, probs = self.ce_model.start_play(board_copy)  #######修改
            board_copy.move(move)
            self.ce_model.reset_mcts(move)
            probs_list.append(probs)
            player_list.append(board_copy.get_current_player())
        winner_list = np.zeros(len(player_list))
        end, winner = board_copy.is_end()

        if winner != -1:
            winner_list[np.array(player_list) == winner] = 1
            winner_list[np.array(player_list) != winner] = -1
        return zip(winner_list, probs_list, state_list)

    def evaluate(self):
        board = Board(self.width, self.height, self.n)
        mcts_player = MctsPlay(n_play_times=self.init_mcts_playtimes)
        ce_mctsplayer = Ce_MctsPlay(pv_net=self.model.pv_net, cuda_enable=self.cuda_enable,
                                    return_prob=1, is_selfplay=0)
        win_dic = {}
        win_list = []
        self.game = Game(board, mcts_player, ce_mctsplayer, self.dir_cnt, is_show=0)
        for i in tqdm(range(self.playtimes)):
            # winner = self.game.start_play(i % 2)
            ##################+ai与mcts对弈结果
            winner,mcts_data = self.game.start_play(i % 2)
            self.collect_data(mcts_data)
            ###################
            win_dic[winner] = win_dic.get(winner, 0) + 1
            self.game.reset_game()
            win_list.append(winner)
        print(win_dic, win_list)
        return (win_dic.get(2, 0) + win_dic.get(-1, 0) * 0.5) / self.playtimes

    def train(self):
        for epoch in range(self.epoch):
            data = self.collect_data()
            self.model.unzip_data(data)
            self.model.model.train()
            loss_deque = deque(maxlen=self.max_epo)
            loss_deque.extend([float('inf')] * self.max_epo)
            loss = 0
            cnt = 1  # 辅助计数，打印loss
            old_p, old_v = self.model.pv()
            while loss < sum(loss_deque) / self.max_epo:  # 损失下降到一定程度，停止当批次的训练
                loss = self.model.train_step()
                loss_deque.append(loss)
                new_p, new_v = self.model.pv()
                entropy = - np.mean(np.sum(new_p * np.log(new_p), 1))
                kl = np.mean(np.sum(old_p * (np.log(old_p + 1e-10) - np.log(new_p + 1e-10)), axis=1))
                if cnt % 1 == 0:
                    print('epoch:{},cnt:{},loss:{},kl:{},entropy:{},lr_multiplier:{},playtimes:{}'
                          .format(epoch, cnt, loss, kl, entropy, self.lr_multiplier, self.init_mcts_playtimes))
                cnt += 1
                if cnt > self.max_epo or kl > 4 * self.kl_targ:
                    break
            if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
            elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5
            self.model.set_optim(lr=self.lr * self.lr_multiplier, weight_decay=self.l2)

            if (epoch + 1) % 10 == 0:
                win_ratio = self.evaluate()
                print('对局结束，AI胜率:{}'.format(win_ratio))
                if win_ratio > self.best_ratio:
                    self.best_ratio = win_ratio
                    torch.save(self.model.model.state_dict(),
                               '{}_{}_{}.pth'.format(self.init_mcts_playtimes, win_ratio, epoch))
                    if win_ratio == 1:
                        self.init_mcts_playtimes += 500
                        self.best_ratio = 0
            # elif (epoch + 1) % 10 == 0 and not self.self_mcts.start_play_to_win(self.model.pv_net):
            #     print('自奕输局，不进行mcts对弈')

            if self.init_mcts_playtimes > self.max_mcts_playtimes:
                print('训练完成')
                break


if __name__ == '__main__':
    train_model = train_net()
    train_model.train()
