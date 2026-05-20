import numpy as np
import chess
import chess.pgn
import gymnasium
import random
import pandas as pd
import os
import io
import asyncio
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

import tasks.classic_games.chess.chess_utils as chess_utils
from tasks.classic_games.chess.tool import StockFish_Client, StockFish_http_client

ROOT_PATH=Path(__file__).parent.parent.parent.parent.resolve()


class ChessPuzzleEnv_Wrapper:
    def __init__(self, env_config):
        # self.reset()
        self.ego_agent = "White"
        self.ego_agent_idx = 0
        self.game_mode = env_config.game_mode    # puzzle or full

        self.opponent_mode = env_config.opponent_mode  #"fixed"        # ["random", "fixed"]
        assert self.opponent_mode in ["random", "fixed"]

        self.puzzle_path = os.path.join(ROOT_PATH, "tasks/classic_games/chess/data/puzzles.csv")

        if env_config.puzzle_scope is not None:
            self.puzzle_scope = env_config.puzzle_scope # get the task pool
        else:
            self.puzzle_scope = None        # if None, get all data

        self._load_puzzle(rating_range=env_config.rating_range)

        self.move_format = env_config.move_format       # ['fen', 'uci']
        self.board_format = env_config.board_format       # ['fen', 'plain']
        assert self.move_format in ['san', 'uci']
        assert self.board_format in ['fen', 'plain']

        self.san_move_leak_symbol = ["+", "#"]
        self.max_legal_action_n = env_config.max_legal_action_n


    def _load_puzzle(self, rating_range=None):
        self.puzzles_pool = pd.read_csv(self.puzzle_path)

        if rating_range is not None:
            # level 1: rating [0-1000]
            # level 2: rating [1000-1500]
            # level 3: rating [1500-2000]
            # level 4: rating [2000-...]
            self.puzzles_pool = self.puzzles_pool[
                self.puzzles_pool["Rating"].between(rating_range[0],rating_range[1])
            ].copy()

        if self.puzzle_scope is not None:
            self.puzzles_pool = self.puzzles_pool.iloc[range(self.puzzle_scope[0], self.puzzle_scope[1] + 1)]
            print(f"Number of Instances = {len(self.puzzles_pool)}")


    def _stats(self):
        total_length = []
        total_rating = []
        for i in self.puzzles_pool['Moves']:
            total_length.append(len(i.split(' ')))
        for i in self.puzzles_pool["Rating"]:
            total_rating.append(int(i))

        print(f"Expert Moves avg length = {np.mean(total_length)}")
        print(f"Avg Puzzle Rating = {np.mean(total_rating)}")

        show_rating_histogram = True

        if show_rating_histogram:
            # Calculate average rating
            avg_rating = self.puzzles_pool["Rating"].mean()

            ratings = self.puzzles_pool["Rating"]
            plt.figure(figsize=(12,7))

            # Create the histogram
            plt.hist(ratings, bins=50, edgecolor='black', alpha=0.7, color='skyblue')

            # Add titles and labels for clarity
            plt.title('Distribution of Puzzle Ratings', fontsize=16)
            plt.xlabel('Rating', fontsize=12)
            plt.ylabel('Number of Puzzles', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add a vertical line for the average rating to give context
            plt.axvline(avg_rating, color='r', linestyle='dashed', linewidth=2, label=f'Avg Rating: {avg_rating:.0f}')
            plt.legend()

            print("\nDisplaying puzzle rating distribution histogram...")
            plt.show()



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
        return len(self.puzzles_pool['PGN'])

    def sample_action(self, return_string):
        if self.opponent_mode == "random":
            oppo_move = random.sample(list(self.game.legal_moves), 1)[0]
        elif self.opponent_mode == 'fixed':
            oppo_move = list(self.game.legal_moves)[0]
        elif self.opponent_mode == "stockfish":
            raise NotImplementedError("Stockfish mode requires AsyncChessPuzzleEnv_Wrapper for async support")
        else:
            raise NotImplementedError(f"opponent mode {self.opponent_mode} is not supported.")

        if return_string:
            return oppo_move.uci()
        else:
            return oppo_move

    def reset(self, seed=None, specified_game_idx=None):

        self.last_reset_seed = seed
        if seed is not None:        # if new seed comes, reset
            random.seed(seed)

        if specified_game_idx is not None:      # specify game
            puzzle_idx = specified_game_idx
        else:
            # sample a puzzle
            if self.puzzle_scope is not None:
                puzzle_idx = random.randrange(0, self.puzzle_scope[1]-self.puzzle_scope[0])
            else:       # sample from all data
                puzzle_idx = random.randrange(0, len(self.puzzles_pool))

        pgn_game = chess.pgn.read_game(io.StringIO(self.puzzles_pool['PGN'].iloc[puzzle_idx]))
        expert_move_list = self.puzzles_pool['Moves'].iloc[puzzle_idx].split(" ")
        self.puzzle_rating = int(self.puzzles_pool['Rating'].iloc[puzzle_idx])

        self.game = pgn_game.end().board()

        # check the first move is as White
        if self.current_player != "White":
            self.game.push(chess.Move.from_uci(expert_move_list[0]))
            expert_move_list = expert_move_list[1:]
            assert self.current_player == "White", print(f"First move is not White, current player is {self.current_player}")
            assert not self.check_termination(), print(f"Game is already over, current player is {self.current_player}")

        self.board = str(self.game)
        assert self._current_player() == "White"

        #chess.Move.from_uci(expert_move_list[0])

        self.agents = ["White", "Black"]  # [f"player_{i}" for i in range(2)]
        self.possible_agents = self.agents[:]
        self._get_all_legal_moves()
        # return {
        #     self._current_player(): self.observe()
        # }
        return self.observe()

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
        if self.board_format == 'fen':
            output_board = self.game.fen()
        elif self.board_format == 'plain':
            output_board = str(self.game)
        else:
            raise NotImplementedError

        return output_board


    def step(self, action):
        assert self._current_player() == "White", print(f"Decision player is not White")
        meta_info = {}

        if isinstance(action, str):
            assert action in self.legal_moves_dict, print(f"\n### action {action} is not valid {list(self.legal_moves_dict.keys())}")
            input_action = self.legal_moves_dict[action]
        elif isinstance(action, chess.Move):
            input_action = action
        else:
            raise NotImplementedError(f"Type {type(action)} is not supported")

        self.game.push(input_action)
        game_over = self.check_termination()

        if not game_over:
            # if the game continue, get Black's move
            oppo_move = self.sample_action(return_string=False)
            self.game.push(oppo_move)
            meta_info['oppo_move'] = str(oppo_move)
            game_over = self.check_termination()

        self.board = str(self.game)

        if game_over:
            result = self.game.result(claim_draw=True)
            result_val = chess_utils.result_to_int(result)
            if result_val == 1:
                _all_reward = [1., -1.]
            elif result_val == -1:
                _all_reward = [-1., 1.]
            elif result_val == 0:
                _all_reward = [0., 0.]
            else:
                raise NotImplementedError
        else:
            _all_reward = [0., 0.]

        # obs = {
        #     self._current_player(): self.observe()
        # }
        obs = self.observe()

        self.reward = _all_reward[self.ego_agent_idx]

        return obs, self.reward, game_over, meta_info

    def check_success(self):
        return (self.reward > 0)       # assume always play White

    def check_lose(self):
        return (self.reward < 0)

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


    def render(self, print_board=False):
        split_board = str(self.game).split('\n')
        render_board = []
        render_board.append("  ---------------")
        for idx, row in enumerate(split_board):
            render_board.append(f"{8-idx}|" + row)

        render_board.append("  ---------------")
        render_board.append("  a b c d e f g h")
        render_board = "\n".join(render_board)

        # print('---------------')
        if print_board:
            print(render_board)
        # print('---------------\n')
        return render_board


class AsyncChessPuzzleEnv_Wrapper:
    """Async version of ChessPuzzleEnv_Wrapper that supports async operations like Stockfish client."""
    
    def __init__(self, env_config):
        # self.reset()
        self.ego_agent = "White"
        self.ego_agent_idx = 0
        self.game_mode = env_config.game_mode    # puzzle or full

        self.opponent_mode = env_config.opponent_mode  #"fixed"        # ["random", "fixed"]
        assert self.opponent_mode in ["random", "fixed", "stockfish"]
        # load stockfish if there the opponent mode is stockfish
        if env_config.use_stockfish or (self.opponent_mode == "stockfish"):
            if not env_config.stockfish_url:
                raise ValueError("stockfish_url is required when use_stockfish is true or opponent_mode is stockfish.")
            # self.stockfish_client = StockFish_Client(base_url=env_config.stockfish_url)
            self.stockfish_client = StockFish_http_client(search_url=f"{env_config.stockfish_url}/analyse")

        self.puzzle_path = os.path.join(ROOT_PATH, "tasks/classic_games/chess/data/puzzles.csv")

        if env_config.puzzle_scope is not None:
            self.puzzle_scope = env_config.puzzle_scope # get the task pool
        else:
            self.puzzle_scope = None        # if None, get all data

        self._load_puzzle(rating_range=env_config.rating_range)

        self.move_format = env_config.move_format       # ['fen', 'uci']
        self.board_format = env_config.board_format       # ['fen', 'plain']
        assert self.move_format in ['san', 'uci']
        assert self.board_format in ['fen', 'plain']

        self.san_move_leak_symbol = ["+", "#"]
        self.max_legal_action_n = env_config.max_legal_action_n
        # Maximum number of retries for stockfish analyse_position calls
        self.stockfish_max_retries = getattr(env_config, 'stockfish_max_retries', 10)

    async def connect(
        cls,
        env_config,
        ):

        env = cls(env_config)
        return env

    def _load_puzzle(self, rating_range=None):
        self.puzzles_pool = pd.read_csv(self.puzzle_path)

        if rating_range is not None:
            # level 1: rating [0-1000]
            # level 2: rating [1000-1500]
            # level 3: rating [1500-2000]
            # level 4: rating [2000-...]
            self.puzzles_pool = self.puzzles_pool[
                self.puzzles_pool["Rating"].between(rating_range[0],rating_range[1])
            ].copy()

        if self.puzzle_scope is not None:
            self.puzzles_pool = self.puzzles_pool.iloc[range(self.puzzle_scope[0], self.puzzle_scope[1] + 1)]
            print(f"Number of Instances = {len(self.puzzles_pool)}")

    def _stats(self):
        total_length = []
        total_rating = []
        for i in self.puzzles_pool['Moves']:
            total_length.append(len(i.split(' ')))
        for i in self.puzzles_pool["Rating"]:
            total_rating.append(int(i))

        print(f"Expert Moves avg length = {np.mean(total_length)}")
        print(f"Avg Puzzle Rating = {np.mean(total_rating)}")

        show_rating_histogram = True

        if show_rating_histogram:
            # Calculate average rating
            avg_rating = self.puzzles_pool["Rating"].mean()

            ratings = self.puzzles_pool["Rating"]
            plt.figure(figsize=(12,7))

            # Create the histogram
            plt.hist(ratings, bins=50, edgecolor='black', alpha=0.7, color='skyblue')

            # Add titles and labels for clarity
            plt.title('Distribution of Puzzle Ratings', fontsize=16)
            plt.xlabel('Rating', fontsize=12)
            plt.ylabel('Number of Puzzles', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add a vertical line for the average rating to give context
            plt.axvline(avg_rating, color='r', linestyle='dashed', linewidth=2, label=f'Avg Rating: {avg_rating:.0f}')
            plt.legend()

            print("\nDisplaying puzzle rating distribution histogram...")
            plt.show()

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

        self.legal_moves_dict = legal_moves_dict

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
        return len(self.puzzles_pool['PGN'])

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

    async def analyse_position(self, num_moves=1):
        """
        the stockfish engine takes input of:
        1. game fen
        2. multipv: number of principal variations to return, =1 means the optimal move
        the results will be like:
        {'engine_id': 14, 'results': [
            {'score': {'type': 'mate', 'moves': 1}, 'pv': ['f1f8'], 'depth': 16, 'seldepth': 2}, 
            {'score': {'type': 'cp', 'value': -605}, 'pv': ['d3d4', 'c6f6', 'f1f6', 'g7f6', 'd4e5', 'f6e5', 'a1d1', 'd7e6', 'g1h1', 'h8g8'], 'depth': 16, 'seldepth': 22}, 
            {'score': {'type': 'cp', 'value': -601}, 'pv': ['c3c4', 'g5e3', 'g1h1', 'c6f6', 'f1f6', 'g7f6', 'a1e1', 'e3c5', 'e1f1', 'c5e7'], 'depth': 15, 'seldepth': 19}
            ]}
        
        the return is a list of optimal first move in the rank of their optimality
        """
        max_retries = self.stockfish_max_retries
        
        # for attempt in range(max_retries):
        #     try:

        analysis_result = await self.stockfish_client.analyse_position(self.game.fen(), multipv=num_moves)
        
        proposed_move_list = []
        for result in analysis_result['results']:
            first_move = result['pv'][0]
            uci_move = chess.Move.from_uci(first_move)
            if self.move_format == 'san':
                san_move = self.game.san(uci_move)
                for symbol in self.san_move_leak_symbol:
                    if symbol in san_move:
                        san_move = san_move.replace(symbol, "")
                        break
                proposed_move_list.append(san_move)
            elif self.move_format == 'uci':
                proposed_move_list.append(first_move)
            else:
                raise NotImplementedError(f"move format {self.move_format} is not supported.")
        
        return proposed_move_list
            # except Exception as e:
            #     if attempt < max_retries - 1:
            #         # Wait a bit before retrying (exponential backoff)
            #         await asyncio.sleep(0.1 * (2 ** attempt))
            #     continue
        
        # If all retries failed, return empty list
        # return []


    async def reset(self, seed=None, specified_game_idx=None):
        self.last_reset_seed = seed
        if seed is not None:        # if new seed comes, reset
            random.seed(seed)

        if specified_game_idx is not None:      # specify game
            puzzle_idx = specified_game_idx
        else:
            # sample a puzzle
            if self.puzzle_scope is not None:
                puzzle_idx = random.randrange(0, self.puzzle_scope[1]-self.puzzle_scope[0])
            else:       # sample from all data
                puzzle_idx = random.randrange(0, len(self.puzzles_pool))

        pgn_game = chess.pgn.read_game(io.StringIO(self.puzzles_pool['PGN'].iloc[puzzle_idx]))
        expert_move_list = self.puzzles_pool['Moves'].iloc[puzzle_idx].split(" ")
        self.puzzle_rating = int(self.puzzles_pool['Rating'].iloc[puzzle_idx])
        self.puzzle_id = self.puzzles_pool['PuzzleId'].iloc[puzzle_idx]

        self.game = pgn_game.end().board()

        # check the first move is as White
        if self.current_player != "White":
            self.game.push(chess.Move.from_uci(expert_move_list[0]))
            expert_move_list = expert_move_list[1:]
            assert self.current_player == "White", print(f"First move is not White, current player is {self.current_player}")
            assert not self.check_termination(), print(f"Game is already over, current player is {self.current_player}")

        self.board = str(self.game)
        assert self._current_player() == "White"

        self.agents = ["White", "Black"]
        self.possible_agents = self.agents[:]
        self._get_all_legal_moves()
        return self.observe()

    async def reset_state(self, key_stats_dict):
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

    async def step(self, action):
        """Async version of step that supports async opponent moves."""
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
            # if the game continue, get Black's move (async)
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
                _all_reward = [1., -1.]
            elif result_val == -1:
                _all_reward = [-1., 1.]
            elif result_val == 0:
                _all_reward = [0., 0.]
            else:
                raise NotImplementedError
        else:
            _all_reward = [0., 0.]

        obs = self.observe()

        self.reward = _all_reward[self.ego_agent_idx]

        return obs, self.reward, game_over, meta_info

    def check_success(self):
        return (self.reward > 0)       # assume always play White

    def check_lose(self):
        return (self.reward < 0)

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

    def render(self, print_board=False):
        split_board = str(self.game).split('\n')
        render_board = []
        render_board.append("  ---------------")
        for idx, row in enumerate(split_board):
            render_board.append(f"{8-idx}|" + row)

        render_board.append("  ---------------")
        render_board.append("  a b c d e f g h")
        render_board = "\n".join(render_board)

        if print_board:
            print(render_board)
        return render_board

    async def close(self):
        """Close async resources like the Stockfish client."""
        if hasattr(self, 'stockfish_client') and self.stockfish_client is not None:
            await self.stockfish_client.close()


if __name__ == "__main__":

    @dataclass(kw_only=True)
    class EnvConfig:
        env_name: str = "Chess"
        batch_sample: bool = False
        batch_sample_size: int = 64
        game_mode: str = "full"
        puzzle_scope = [0,9999]
        move_format = "san"
        board_format = "fen"
        rating_range = [0,500]
        opponent_mode = "stockfish"
        max_legal_action_n: int = 30


    cfg = EnvConfig()
    cfg.game_mode = "puzzle"
    cfg.puzzle_scope = None

    # Use async version for stockfish mode, sync version for others
    if cfg.opponent_mode == "stockfish":
        async def test_async_env():
            game1 = AsyncChessPuzzleEnv_Wrapper(cfg)
            state = game1.reset()

            game1._stats()

            game1.render(True)
            done = False
            steps = 0
            print(f"legal_moves_list = {game1.legal_moves_list}")

            analysis_result = await game1.analyse_position(num_moves=3)

            move1 = random.choice(game1.legal_moves_list)   #await game1.sample_action(return_string=True)
            print(f"move1 = {move1}")
            obs, reward, done, meta = await game1.step(move1)
            print(f"oppo_move = {meta['oppo_move']}")
            game1.render(True)

            game1_state = game1.get_key_stats()
            await game1.close()
            return obs, reward, done, meta

        # Run async test
        asyncio.run(test_async_env())
    else:
        # Use sync version for non-stockfish modes
        game1 = ChessPuzzleEnv_Wrapper(cfg)
        state = game1.reset()

        game1._stats()

        game1.render(True)
        done = False
        steps = 0
        print(game1.legal_moves_list)
        print(game1.game.legal_moves)

        move1 = random.choice(game1.legal_moves_list)   #game1.sample_action(return_string=True)
        print(f"move1 = {move1}")
        game1.step(move1)

        game1.render()

        game1_state = game1.get_key_stats()




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
