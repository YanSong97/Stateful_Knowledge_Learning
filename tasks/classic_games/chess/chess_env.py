import numpy as np
import chess
import chess.pgn
import gymnasium
import random
import pandas as pd
import os
import io
from dataclasses import dataclass

import tasks.classic_games.chess.chess_utils as chess_utils
from tasks.classic_games.chess.tool import StockFish_Client


class ChessEnv_Wrapper:
    def __init__(self, env_config):
        # self.reset()
        self.ego_agent = "White"
        self.ego_agent_idx = 0

    def _get_all_legal_moves(self):
        legal_moves_dict = {}
        for i in self.game.legal_moves:
            legal_moves_dict[i.uci()] = i

        self.legal_moves_dict = legal_moves_dict
        # return legal_moves_dict

    def _current_player(self):
        if self.game.turn == chess.WHITE:
            return "White"
        elif self.game.turn == chess.BLACK:
            return "Black"
        else:
            raise NotImplementedError

    @property
    def legal_moves(self):
        # return list(self.legal_moves_dict.keys())
        return self.legal_moves_dict

    @property
    def legal_moves_list(self):
        return list(self.legal_moves_dict.keys())

    @property
    def current_player(self):
        return self._current_player()

    def sample_action(self, return_string):
        random_move = random.sample(list(self.game.legal_moves), 1)
        if return_string:
            return random_move[0].uci()
        else:
            return random_move[0]

    def reset(self, seed=None):

        self.game = chess.Board()
        self.board = str(self.game)
        assert self._current_player() == "White"

        self.agents = ["White", "Black"]  #[f"player_{i}" for i in range(2)]
        self.possible_agents = self.agents[:]
        self._get_all_legal_moves()

        self.last_reset_seed = seed
        if seed is not None:
            random.seed(seed)

        # return {
        #     self._current_player(): self.observe()
        # }
        return self.observe()       # return White as default

    def reset_state(self, key_stats_dict):
        fen_format = key_stats_dict['FEN']
        self.game = chess.Board(fen_format)
        self.board = str(self.game)
        assert self._current_player() == "White"

        self._get_all_legal_moves()

        # here we do not reset the seed, as the agent does not know the opponent's strategy in advanced
        # return {
        #     self._current_player(): self.observe()
        # }
        return self.observe()


    def get_key_stats(self):
        return {
            "FEN": self.game.fen()
        }


    def observe(self):
        # assert agent in self.possible_agents
        # current_index = self.possible_agents.index(agent)
        return self.board


    def step(self, action):
        assert self._current_player() == "White", print(f"Decision player is not White")

        if isinstance(action, str):
            assert action in self.legal_moves_dict, print(f"action {action} is not valid {list(self.legal_moves_dict.keys())}")
            input_action = self.legal_moves_dict[action]
        elif isinstance(action, chess.Move):
            input_action = action
        else:
            raise NotImplementedError(f"Type {type(action)} is not supported")

        self.game.push(input_action)
        game_over = self.check_termination()

        if not game_over:
            # if the game continue, get Black's move
            self.game.push(self.sample_action(return_string=False))
            game_over = self.check_termination()

        self.board = str(self.game)

        if game_over:
            result = self.game.result(claim_draw=True)
            result_val = chess_utils.result_to_int(result)
            if result_val == 1:
                _all_reward = [1, -1]
            elif result_val == -1:
                _all_reward = [-1, 1]
            elif result_val == 0:
                _all_reward = [0, 0]
            else:
                raise NotImplementedError
        else:
            _all_reward = [0, 0]

        # obs = {
        #     self._current_player(): self.observe()
        # }
        obs = self.observe()

        self.reward = _all_reward[self.ego_agent_idx]

        return obs, self.reward, game_over, ""

    def check_success(self):
        return (self.reward > 0)       # assume always play White

    def check_results(self):
        if (self.reward > 0):
            return "win"
        elif (self.reward < 0):
            return "lose"
        else:
            assert (self.reward == 0)
            return "tie"


    def check_termination(self):
        self._get_all_legal_moves()
        is_stale_or_checkmate = not any(self.legal_moves_dict)
        is_insufficient_material = self.game.is_insufficient_material()
        can_claim_draw = self.game.can_claim_draw()
        end = can_claim_draw or is_stale_or_checkmate or is_insufficient_material
        return end


    def render(self):
        split_board = self.board.split('\n')
        render_board = []
        render_board.append("  ---------------")
        for idx, row in enumerate(split_board):
            render_board.append(f"{8-idx}|" + row)

        render_board.append("  ---------------")
        render_board.append("  a b c d e f g h")
        render_board = "\n".join(render_board)

        # print('---------------')
        # print(render_board)
        # print('---------------\n')
        return render_board


class AsyncChessEnv_Wrapper:
    def __init__(self, env_config):
        # self.reset()
        self.ego_agent = "White"
        self.ego_agent_idx = 0

        self.opponent_mode = env_config.opponent_mode
        assert self.opponent_mode in ["random", "fixed", "stockfish"]
        # load stockfish if there the opponent mode is stockfish
        if env_config.use_stockfish or (self.opponent_mode == "stockfish"):
            if not env_config.stockfish_url:
                raise ValueError("stockfish_url is required when use_stockfish is true or opponent_mode is stockfish.")
            self.stockfish_client = StockFish_Client(base_url=env_config.stockfish_url)
        
        self.move_format = env_config.move_format       # ['fen', 'uci']
        self.board_format = env_config.board_format       # ['fen', 'plain']
        assert self.move_format in ['san', 'uci']
        assert self.board_format in ['fen', 'plain']

        self.san_move_leak_symbol = ["+", "#"]
        self.max_legal_action_n = env_config.max_legal_action_n
        # Maximum number of retries for stockfish analyse_position calls
        self.stockfish_max_retries = getattr(env_config, 'stockfish_max_retries', 10)
        self.full_game_n = env_config.full_game_n

    def _get_all_legal_moves(self):
        legal_moves_dict = {}
        for i in self.game.legal_moves:
            if self.move_format == 'san':
                original_san_move = self.game.san(i)
                clean_san_move = original_san_move
                for symbol in self.san_move_leak_symbol:
                    if symbol in original_san_move:
                        clean_san_move = original_san_move.replace(symbol, "")
                        break
                legal_moves_dict[clean_san_move] = i
            elif self.move_format == 'uci':
                legal_moves_dict[i.uci()] = i
            else:
                raise NotImplementedError(f"move format {self.move_format} is not supported.")

    def _current_player(self):
        if self.game.turn == chess.WHITE:
            return "White"
        elif self.game.turn == chess.BLACK:
            return "Black"
        else:
            raise NotImplementedError

    @property
    def legal_moves(self):
        return self.legal_moves_dict

    @property
    def legal_moves_list(self):
        all_legal_moves = list(self.legal_moves_dict.keys())
        return all_legal_moves

    @property
    def legal_moves_string(self):
        all_legal_moves = list(self.legal_moves_dict.keys())
        if len(all_legal_moves) > self.max_legal_action_n:
            output_string = all_legal_moves[:self.max_legal_action_n] + ["..."]
            return str(output_string)
        else:
            output_string = all_legal_moves
            return str(output_string)

    @property
    def current_player(self):
        return self._current_player()

    @property
    def num_game(self):
        return len(self.game_pool)

    def reset(self, seed=None, specified_game_idx=None):
        self.last_reset_seed = seed
        if seed is not None:
            random.seed(seed)
        

        self.game = chess.Board()
        self.board = str(self.game)
        assert self._current_player() == "White"

        self.agents = ["White", "Black"]  #[f"player_{i}" for i in range(2)]
        self.possible_agents = self.agents[:]
        self._get_all_legal_moves()

        return self.observe()       # return White as default

    def reset_state(self, key_stats_dict):
        fen_format = key_stats_dict['FEN']
        self.game = chess.Board(fen_format)
        self.board = str(self.game)
        assert self._current_player() == "White"

        self._get_all_legal_moves()
        return self.observe()

    def get_key_stats(self):
        return {
            "FEN": self.game.fen()
        }

    def observe(self):
        if self.board_format == 'fen':
            output_board = self.game.fen()
        elif self.board_format == 'plain':
            output_board = str(self.game)
        else:
            raise NotImplementedError

        return output_board

    async def sample_action(self, return_string):
        """Async version of sample_action that supports async Stockfish client."""
        if self.opponent_mode == "random":
            oppo_move = random.sample(list(self.game.legal_moves), 1)[0]
        elif self.opponent_mode == 'fixed':
            oppo_move = list(self.game.legal_moves)[0]
        elif self.opponent_mode == "stockfish":
            stockfish_result = await self.stockfish_client.play_move(self.game.fen())   # default move format is uci
            oppo_move = chess.Move.from_uci(stockfish_result['move'])
        else:
            raise NotImplementedError(f"opponent mode {self.opponent_mode} is not supported.")

        if return_string:
            return oppo_move.uci()
        else:
            return oppo_move

    async def step(self, action):
        assert self._current_player() == "White", print(f"Decision player is not White")
        meta_info = {}

        if isinstance(action, str):
            assert action in self.legal_moves_dict, print(f"action {action} is not valid {list(self.legal_moves_dict.keys())}")
            input_action = self.legal_moves_dict[action]
        elif isinstance(action, chess.Move):
            input_action = action
        else:
            raise NotImplementedError(f"Type {type(action)} is not supported")

        self.game.push(input_action)
        game_over = self.check_termination()

        if not game_over:
            # if the game continue, get Black's move
            oppo_move = await self.sample_action(return_string=False)
            if self.opponent_mode == "stockfish":
                if self.move_format == 'san':
                    oppo_move_san = self.game.san(oppo_move)
                    meta_info['oppo_move'] = oppo_move_san.replace("+", "").replace("#", "").lower()    # lower letter for better demonstration
                elif self.move_format == 'uci':
                    meta_info['oppo_move'] = str(oppo_move)
                else:
                    raise NotImplementedError(f"move format {self.move_format} is not supported.")
            else:
                meta_info['oppo_move'] = str(oppo_move)

            self.game.push(oppo_move)
            game_over = self.check_termination()

        self.board = str(self.game)

        if game_over:
            result = self.game.result(claim_draw=True)
            result_val = chess_utils.result_to_int(result)
            if result_val == 1:
                _all_reward = [1, -1]
            elif result_val == -1:
                _all_reward = [-1, 1]
            elif result_val == 0:
                _all_reward = [0, 0]
            else:
                raise NotImplementedError
        else:
            _all_reward = [0, 0]

        obs = self.observe()

        self.reward = _all_reward[self.ego_agent_idx]

        return obs, self.reward, game_over, meta_info

    def check_success(self):
        return (self.reward > 0)       # assume always play White

    def check_results(self):
        if (self.reward > 0):
            return "win"
        elif (self.reward < 0):
            return "lose"
        else:
            assert (self.reward == 0)
            return "tie"


    def check_termination(self):
        self._get_all_legal_moves()
        is_stale_or_checkmate = not any(self.legal_moves_dict)
        is_insufficient_material = self.game.is_insufficient_material()
        can_claim_draw = self.game.can_claim_draw()
        end = can_claim_draw or is_stale_or_checkmate or is_insufficient_material
        return end


    def render(self):
        split_board = self.board.split('\n')
        render_board = []
        render_board.append("  ---------------")
        for idx, row in enumerate(split_board):
            render_board.append(f"{8-idx}|" + row)

        render_board.append("  ---------------")
        render_board.append("  a b c d e f g h")
        render_board = "\n".join(render_board)

        # print('---------------')
        # print(render_board)
        # print('---------------\n')
        return render_board
    
    async def close(self):
        """Close async resources like the Stockfish client."""
        if hasattr(self, 'stockfish_client') and self.stockfish_client is not None:
            await self.stockfish_client.close()    # close the stockfish client


if __name__ == "__main__":

    @dataclass(kw_only=True)
    class EnvConfig:
        env_name: str = "Chess"
        batch_sample: bool = False
        batch_sample_size: int = 64
        game_mode: str = "full"
        puzzle_scope: str = "[10,20]"
        map_seed = None


    cfg = EnvConfig
    cfg.game_mode = "puzzle"

    game1 = ChessEnv_Wrapper(EnvConfig)
    state = game1.reset(seed=123)


    game1.render()
    done = False
    steps = 0

    move1 = game1.sample_action(return_string=True)
    game1.step(move1)

    game1.render()

    game1_state = game1.get_key_stats()

    game2 = ChessEnv_Wrapper(None)
    game2.reset_state(game1_state)

    game2.render()



    # while not done:
    #     move = game1.sample_action(True)
    #     print(f"player = {list(state.keys())[0]}, Move = {move}")
    #     next_state, reward, done, _ = game1.step(move)
    #     game1.render()
    #
    #     if done and (reward != [0, 0]):
    #         print('1')
    #     steps += 1
    #
    #     state = next_state
    #
    #
    # print(f"Total steps = {steps}, reward = {reward}")
