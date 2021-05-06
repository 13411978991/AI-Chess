import numpy as np
import pickle
"""创建棋盘，记录棋的位置，落子位置，判断输赢"""
class Board(object):
    def __init__(self, width=8, height=8, n_=5):
        self.width = width
        self.height = height
        self.n_ = n_
        self.players = [1, 2]

    def get_current_player(self):
        return self.current_player

    def init_board(self, is_humen_first=1):
        self.states = {}
        self.empty_loc = [i for i in range(self.width * self.height)]
        self.current_player = (self.players[0] if is_humen_first else self.players[1])
        self.last_move = -1

    def move(self, loc):
        self.states[loc] = self.current_player
        self.empty_loc.remove(loc)
        self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
        self.last_move = loc

    def get_state(self):  #############获取当前局面、最后落子、落子玩家
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0
        return square_state[:, ::-1, :]


    def is_end(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_
        moved = list(set(range(width * height)) - set(self.empty_loc))
        if len(moved) < n * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]
            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player
            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player
        if not self.empty_loc:
            return True, -1
        return False, -1

"""使用蒙特卡洛树搜索与人对战"""
class Game(object):
    def __init__(self, board, player1, player2, path,is_show=0):
        self.board = board
        self.AI = player1
        self.AI1 = player2
        self.is_show = is_show
        self.path = path
        ###########添加ai与mcts对战的记录，增加数据，用于训练模型，为了解决胜率波动问题


    def start_play(self, is_humen_first=1):
        self.board.init_board(is_humen_first)
        player_list, probs_list, state_list = [], [], []
        cnt = 0
        while not self.board.is_end()[0]:
            player = self.board.get_current_player()
            if player == 2:
                # act = self.AI1.start_play(self.board)
                ###########+ai与mcts对战数据
                act, probs = self.AI1.start_play(self.board)
                probs_list.append(probs)
                player_list.append(player)
                state_list.append(self.board.get_state())
                ###################

                self.board.move(act)
                self.AI.reset_mcts(act)
                self.AI1.reset_mcts(act)
            else:
                # act = self.AI.start_play(self.board)
                act, probs = self.AI.start_play(self.board)
                ###########+ai与mcts对战数据
                probs_list.append(probs)
                player_list.append(player)
                state_list.append(self.board.get_state())
                ###################

                self.board.move(act)
                self.AI.reset_mcts(act)
                self.AI1.reset_mcts(act)

            if self.is_show:
                self.graph(self.board)

            cnt+=1
        end,winner = self.board.is_end()
        ###########+ai与mcts对战数据
        winner_list = np.zeros(len(player_list))
        if winner != -1:
            winner_list[np.array(player_list) == winner] = 1
            winner_list[np.array(player_list) != winner] = -1
        return winner,zip(winner_list, probs_list, state_list)
        ################
        # return winner


    def reset_game(self):
        self.board.init_board()
        self.AI.reset_mcts(-1)
        self.AI1.reset_mcts(-1)


    def graph(self, board):
        width = board.width
        height = board.height

        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == 1:
                    print('X'.center(8), end='')
                elif p == 2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

class Humen(object):
    def __init__(self,board):
        self.width = board.width
        self.height = board.height
        self.board = board
    def start_play(self):
        loc = input('请输入落子位置：')
        try:
            m,n = [int(i) for i in loc.split(',')]
            move = m*self.width+n
        except:
            move = -1
        if move == -1:
            move = self.start_play()
        return move

