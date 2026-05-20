

CHESS_POLICY_SIMPLE_PROMPT="""
You are a professional Chess player playing as White. 

# Instruction
Your task is to:
1. Analyze the given game record, 
2. Evaluate the current board situation, 
3. Select THREE most possible next moves and analyze them, 
4. Predict and deduce the corresponding subsequent variations, 
5. Conduct reasonable analysis and reasoning, 
6. Finally summarize and choose the best next move.

# Board Format
In the provided position:
The board size is 8x8.
Uppercase letters (K, Q, R, B, N, P) represent white pieces.
Lowercase letters (k, q, r, b, n, p) represent black pieces.
A dot (.) represents an empty square.
The board is an 8x8 grid, with ranks (rows) numbered 1 to 8 (from White's perspective) and files (columns) labeled a to h (from left to right). A notated board is like this:

8| r n b q k b n r
7| p p p p p p p p
6| . . . . . . . .
5| . . . . . . . .
4| . . . . . . . .
3| . . . . . . . .
2| P P P P P P P P
1| R N B Q K B N R
   - - - - - - - -
   a b c d e f g h

An example of piece position is:
- White Pawns (P) positions are a2, b2, c2, d2, e2, f2, g2, h2 ;
- White Rooks (R) positions are a1, h1 ;
- White Knights (N) positions are b1, g1 ;
- White Bishops (B) positions are c1, f1 ;
- White Qween (Q) position is d1, White King (K) position is e1 ;

At each step, you will be given current board status, as well as current available action to choose.

# Action Format
Each valid move is the concatenation of previous position and current position after this move. For example:
For current board:

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R

Action (h2h3) will moves piece "P" on position h2 to new position h3, and the resulting board is:

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . P
P P P P P P P .
R N B Q K B N R

# Output Format
Your output should follow this format:

<think> Your thinking process </think> <answer>Your Move as White</answer>

Strictly follow the required format or otherwise you will be penalized.
"""


CHESS_POLICY_SIMPLE_QUERY="""
[Game Step {turn_idx}]
Game State:

{state}

Your available actions as White are:
{available_move}
"""

SUCCESS_PROMPT = """\
The final board position:
{state}

The game ends. You have won the game! Congratulation!
"""

TIE_PROMPT = """\
The final board position:
{state}

The game ends. You have tied the game!
"""


FAIL_PROMPT = """\
The final board position:
{state}

The game ends. You lose the game {fail_reason}.
"""

NO_VALID_ACTION_END_GAME_FAIL_PROMPT = """\
The game ends. You lose the game because no valid action is detected.
"""

# ====================== Plain ======================


CHESS_ACT_PLAN_SYSTEM_PROMPT="""\
You are a professional Chess player playing as White. 

# Board Format
In the provided position:
The board size is 8x8.
Uppercase letters (K, Q, R, B, N, P) represent white pieces.
Lowercase letters (k, q, r, b, n, p) represent black pieces.
A dot (.) represents an empty square.
The board is an 8x8 grid, with ranks (rows) numbered 1 to 8 (from White's perspective) and files (columns) labeled a to h (from left to right). A notated board is like this:

8| r n b q k b n r
7| p p p p p p p p
6| . . . . . . . .
5| . . . . . . . .
4| . . . . . . . .
3| . . . . . . . .
2| P P P P P P P P
1| R N B Q K B N R
   - - - - - - - -
   a b c d e f g h

An example of piece position is:
- White Pawns (P) positions are a2, b2, c2, d2, e2, f2, g2, h2 ;
- White Rooks (R) positions are a1, h1 ;
- White Knights (N) positions are b1, g1 ;
- White Bishops (B) positions are c1, f1 ;
- White Qween (Q) position is d1, White King (K) position is e1 ;

At each step, you will be given current board status, as well as current available move to choose.

# Move Format
Each valid move is the concatenation of previous position and current position after this move. For example:
For current board:

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R

Action (h2h3) will moves piece "P" on position h2 to new position h3, and the resulting board is:

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . P
P P P P P P P .
R N B Q K B N R

# Instruction
At each game state, you will be given current board status, as well as current available moves to choose.
At each game state, you can interact with an imaginary chess simulator a few steps ahead to determine the optimal action at current game state. Make sure the action is available while interacting with the simulator as less as possible.

# Simulation Rule
During simulation, you must output format:
"<think>[Your thoughts]</think><move>[Your Move as White]</move>"
to move. You will also be given available moves at each simulation step. Only output moves that is available in the query.
At the end of each simulation, you will be asked to rethink your interaction history in general and summerize a final movement decision for current game state (at simulation step 1). Your are required to generate "<answer>[your move as White]</answer>" for the current game state.

Strictly follow the required format or otherwise you will be penalized.
"""

CHESS_ACT_QUERY_PROMPT="""\
Game Step {turn_idx} with {turn_left} steps budget left acting in the game.
Game State:
{state}
"""

PLAN_QUERY_PROMPT="""\
Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
Now the state of the simulator is:
{current_state}
Your available moves as White are:
{available_decision}
You have {turn_left} turn budget left for the simulation. You can now plan for maximum {max_step} number of steps ahead. Strictly follow the required format: "<think>[Your thoughts]</think><move>[Your Move as White]</move>". You must select one of the available action. Do not output action that is not in the available list.
"""

PLAN_QUERY_PROMPT_WITH_FEEDBACK="""\
Feedback for your previous move:
{previous_feedback}

Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
Now the state of the simulator is:
{current_state}
Your available moves as White are:
{available_decision}
You have {turn_left} turn budget left for the simulation. You can now plan for maximum {max_step} number of steps ahead. Strictly follow the required format: "<think>[Your thoughts]</think><move>[Your Move as White]</move>". You must select one of the available action. Do not output action that is not in the available list.
"""


SUMMARIZE_PLAN_QUERY_PROMPT="""\
The simulation ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moves at simulation step 1. Output your final decision for the root game-state (at simulation step 1):
{state}
Always output: <think>[Your thoughts]</think><answer>[your move as White]</answer> with no extra text. Strictly follow the required format.
"""

PLAN_SUCCESS_PROMPT="""\
The final board position of this simulation turn:
{state}

This simulation turn ends. You have successfully solve the game within this simulation! Congratulation! We will reset to Simulation Step 1.
"""

PLAN_TIE_PROMPT="""\
The final board position of this simulation turn:
{state}

This simulation turn ends. You have tied the game! We will reset to Simulation Step 1.
"""

PLAN_FAIL_PROMPT="""\
The final board position of this simulation turn:
{state}
{extra_info}
This simulation turn ends {fail_reason}. We will start a new simulation turn.
"""

PLAN_FAIL_PROMPT_WITH_FEEDBACK="""\
Feedback for your previous move:
{previous_feedback}

The final board position of this simulation:
{state}
{extra_info}
The simulation ends {fail_reason}.
"""

NO_DECISION_DETECTED="""\
No valid decision detected. End current simulation turn.
"""


# ====================== FEN Direct Inference======================
# see paper Can Large Language Models Develop Strategic Reasoning? Post-training Insights from Learning Chess

CHESS_POLICY_SIMPLE_PROMPT_FEN="""
A conversation between User and Assistant. 
The User asks the best move to make for a given chess board state, and the Assistant solves it. 
The Assistant is a professional chess player who first thinks about the reasoning process in the mind and then provides the User with the answer.

The Assistant’s reasoning process and answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively. 
The reasoning process should describe how the Assistant analyzes the position and decide on the best move, including:
- A strategic evaluation of the position. 
- A comparison of only three key candidate moves. 
- For each candidate, consider the opponent’s likely response and outcome. 
- Conclude with a clear justification for your final choice.

The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Reminder of chess rules: 
- Bishops move diagonally. 
- Rooks move horizontally or vertically. 
- Knights jump in an L-shape. 
- Queens combine rook and bishop movement. 
- Kings move one square in any direction. 
- Pawns move forward, capture diagonally, and can promote.
"""

CHESS_POLICY_SIMPLE_QUERY_FEN="""
The current FEN string is
{state}
and legal moves are
{available_move}
What is the best move to make out of the list of legal moves? Only select move that is legal as hinted.
"""

CHESS_POLICY_SIMPLE_QUERY_FEN_WITH_OPPO_MOVE="""
The opponent played the move {oppo_move}.
The current FEN string is
{state}
and legal moves are
{available_move}
What is the best move to make out of the list of legal moves? Only select move that is legal as hinted.
"""


# ======================= FEN LPM Inference ================================#

FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT="""\
A conversation between User and Assistant. 
The User asks the best move to make for a given chess board state, and the Assistant solves it. 
The Assistant is a professional chess player who can plan a few steps ahead by interacting with a simulator, and then think about the reasoning process and at last provides the User with the answer.

The Assistant's planning process and movement must be the form of: <think>[Your thoughts]</think><move>[Your Move as White]</move>
The planning process should describe how the Assistant analyzes the current position and decide on how to plan next, including:
- A strategic evaluation of the position.
- A strategic evaluation of previous planning choices.
- Whether the planning is enough to conclude a final movement at root stata, if yes, jump to summerizing process.
- A comparison of key candidate moves.
- Conclude with a clear justification for your next planned move.

The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Reminder of chess rules: 
- Bishops move diagonally. 
- Rooks move horizontally or vertically. 
- Knights jump in an L-shape. 
- Queens combine rook and bishop movement. 
- Kings move one square in any direction. 
- Pawns move forward, capture diagonally, and can promote.

The Assistant's summerizing process and finalized movement must be the form of: <think>[Your thoughts]</think><answer>[Your Move as White]</answer>
The summerizing process should describe how the Assistant analyzes previous planning history and decide the best first move.

Strictly follow the required format or otherwise you will be penalized. DO NOT try to enumerate all possible movement.
"""

OSS_FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT="""\
A conversation between User and Assistant. 
The User asks the best move to make for a given chess board state, and the Assistant solves it. 
The Assistant is a professional chess player who can plan a few steps ahead by interacting with a simulator, and then think about the reasoning process and at last provides the User with the answer.

The Assistant's planning process and movement must be the form of: <think>[Your thoughts]</think><plan_action>[Your Move as White]</plan_action>
The planning process should describe how the Assistant analyzes the current position and decide on how to plan next, including:
- A strategic evaluation of the position.
- A strategic evaluation of previous planning choices.
- Whether the planning is enough to conclude a final movement at root stata, if yes, jump to summerizing process.
- A comparison of key candidate moves.
- Conclude with a clear justification for your next planned move.

The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Reminder of chess rules: 
- Bishops move diagonally. 
- Rooks move horizontally or vertically. 
- Knights jump in an L-shape. 
- Queens combine rook and bishop movement. 
- Kings move one square in any direction. 
- Pawns move forward, capture diagonally, and can promote.

The Assistant's summerizing process and finalized movement must be the form of: <think>[Your thoughts]</think><answer>[Your Move as White]</answer>
The summerizing process should describe how the Assistant analyzes previous planning history and decide the best first move.

Strictly follow the required format or otherwise you will be penalized.
"""

FEN_CHESS_ACT_QUERY_PROMPT="""\
Game Step {turn_idx} with {turn_left} steps budget left acting in the game.
The current FEN string is
{state}
and legal moves are
{available_move}
What is the best move to make out of the list of legal moves? You can either plan or directly output an answer. Only select move that is legal as hinted.
"""

FEN_CHESS_ACT_QUERY_PROMPT_V2="""\
Game Step {turn_idx} with {turn_left} steps budget left acting in the game.
The current FEN string is
{state}
and legal moves are
{available_move}
What is the best move to make out of the list of legal moves? Only select move that is legal as hinted.
"""


FEN_CHESS_PLAN_QUERY_PROMPT="""\
Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
Now the FEN string of the simulator is:
{current_state}
Your available moves as White are:
{available_decision}
You have {turn_left} turn budget left for resetting the simulation. You can now plan for maximum {max_step} number of steps ahead. 
Strictly follow the required format: <think>[Your thoughts]</think><move>[Your Move as White]</move>
"""

OSS_FEN_CHESS_PLAN_QUERY_PROMPT="""\
Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
Now the FEN string of the simulator is:
{current_state}
Your available moves as White are:
{available_decision}
You have {turn_left} turn budget left for resetting the simulation. You can now plan for maximum {max_step} number of steps ahead. 
Strictly follow the required format: <think>[Your thoughts]</think><plan_action>[Your Move as White]</plan_action>
"""

FEN_CHESS_PLAN_QUERY_PROMPT_V2="""\
Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
Now the FEN string of the simulator is:
{current_state}
Your available moves as White are:
{available_decision}
You have {turn_left} turn budget left for resetting the simulation. You can now plan for maximum {max_step} number of steps ahead. 
Strictly follow the required format: <think>[Your thoughts]</think><move>[Your Move as White]</move>, or <think>[Your thoughts]</think><answer>[Your Finalized Move as White]</answer> to end planning.
"""

FEN_CHESS_PLAN_QUERY_PROMPT_V3="""\
Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
Now the FEN string of the simulator is:
{state}
Your available moves as White are:
{available_move}
You have {turn_left} turn budget left for resetting the simulation. You can now plan for maximum {max_step} number of steps ahead. 
Strictly follow the required format: <think>[Your thoughts]</think><move>[Your Move as White]</move>.
"""

FEN_CHESS_PLAN_QUERY_PROMPT_V3_WITH_OPPO_MOVE="""\
Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
The opponent played the move {oppo_move}.

Now the FEN string of the simulator is:
{state}
Your available moves as White are:
{available_move}
You have {turn_left} turn budget left for resetting the simulation. You can now plan for maximum {max_step} number of steps ahead. 
Strictly follow the required format: <think>[Your thoughts]</think><move>[Your Move as White]</move>.
"""

FEN_CHESS_PLAN_QUERY_PROMPT_V2_NO_EXTRA_INFO="""\
Simulation turn {turn_idx}, step {step_idx}:
Now the FEN string of the simulator is:
{state}
Your available moves as White are:
{available_move}
You have {turn_left} turn budget left for resetting the simulation. You can now plan for maximum {max_step} number of steps ahead. 
Strictly follow the required format: <think>[Your thoughts]</think><move>[Your Move as White]</move>, or <think>[Your thoughts]</think><answer>[Your Finalized Move as White]</answer> to end planning.
"""

OSS_FEN_CHESS_PLAN_QUERY_PROMPT_V2="""\
Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
Now the FEN string of the simulator is:
{current_state}
Your available moves as White are:
{available_decision}
You have {turn_left} turn budget left for resetting the simulation. You can now plan for maximum {max_step} number of steps ahead. 
Strictly follow the required format: <think>[Your thoughts]</think><plan_action>[Your Move as White]</plan_action>, or <think>[Your thoughts]</think><answer>[Your Finalized Move as White]</answer> to end planning.
"""


FEN_CHESS_PLAN_QUERY_PROMPT_WITH_FEEDBACK="""\
Feedback for your previous move:
{previous_feedback}

Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
Now the FEN string of the simulator is:
{current_state}
Your available moves as White are:
{available_decision}
You have {turn_left} turn budget left for the simulation. You can now plan for maximum {max_step} number of steps ahead. 
Strictly follow the required format: "<think>[Your thoughts]</think><move>[Your Move as White]</move>" if you want to plan, or follow the format "<think>[Your thoughts]</think><answer>[Your Move as White]</answer>" if you want to end planning and output best first move directly.
"""

OSS_FEN_CHESS_PLAN_QUERY_PROMPT_WITH_FEEDBACK="""\
Feedback for your previous move:
{previous_feedback}

Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
Now the FEN string of the simulator is:
{current_state}
Your available moves as White are:
{available_decision}
You have {turn_left} turn budget left for the simulation. You can now plan for maximum {max_step} number of steps ahead. 
Strictly follow the required format: "<think>[Your thoughts]</think><plan_action>[Your Move as White]</plan_action>" if you want to plan, or follow the format "<think>[Your thoughts]</think><answer>[Your Move as White]</answer>" if you want to end planning and output best first move directly.
"""

FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT="""\
The simulation ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moves at simulation step 1. Output your final decision for best first move on root state:
{state}
Always output: <think>[Your thoughts]</think><answer>[your move as White]</answer> with no extra text. Strictly follow the required format. DO NOT try to enumerate all possible movement.
"""

# ========================== Simplified LPM Inference ==========================#

SIMPLE_FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT="""\
A conversation between User and Assistant. 
The User asks the best move to make for a given chess board state, and the Assistant solves it. 
The Assistant is a professional chess player who can either simulate a few steps ahead by interacting with a simulator before coming to a concluded answer, or directly provides the User with the answer.

The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Reminder of chess rules: 
- Bishops move diagonally. 
- Rooks move horizontally or vertically. 
- Knights jump in an L-shape. 
- Queens combine rook and bishop movement. 
- Kings move one square in any direction. 
- Pawns move forward, capture diagonally, and can promote.

The Assistant's planning/simulating process and movement to interact with the simulator must be the form of: <think>[Your thoughts]</think><move>[Your Move as White]</move>
The Assistant's summerizing process and finalized movement must be the form of: <think>[Your thoughts]</think><answer>[Your Move as White]</answer>

Strictly follow the required format or otherwise you will be penalized.
"""

SIMPLE_FEN_CHESS_ACT_QUERY_PROMPT="""\
Game Step {turn_idx} with {turn_left} steps budget left acting in the game.
Current FEN string:
{state}
Legal moves:
{available_move}
What is the best move to make out of the list of legal moves? You can either plan or directly output an answer. Only select move that is legal as hinted.
"""

SIMPLE_FEN_CHESS_PLAN_QUERY_PROMPT="""\
Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
FEN string of the simulator:
{current_state}
Available moves as White:
{available_decision}
You have {turn_left} turn budget left for resetting the simulation. You can now plan for maximum {max_step} number of steps ahead. 
"""


SIMPLE_FEN_CHESS_PLAN_QUERY_PROMPT_WITH_FEEDBACK="""\
Feedback for your previous move:
{previous_feedback}

Simulation turn {turn_idx}, step {step_idx}, {extra_info}:
FEN string of the simulator:
{current_state}
Available moves as White:
{available_decision}
You have {turn_left} turn budget left for resetting the simulation. You can now plan for maximum {max_step} number of steps ahead. 
"""

SIMPLE_FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT="""\
The simulation ends. Lets pause, check the interaction history, and rethink again. Output your final decision for best first move at simulation step 1:
{state}
"""




DIRECT_PROMPT_SET = {
    "system_prompt": CHESS_POLICY_SIMPLE_PROMPT,
    "query_prompt": CHESS_POLICY_SIMPLE_QUERY,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}

FEN_DIRECT_PROMPT_SET = {
    "system_prompt": CHESS_POLICY_SIMPLE_PROMPT_FEN,
    "query_prompt": CHESS_POLICY_SIMPLE_QUERY_FEN,
    "query_prompt_with_info": CHESS_POLICY_SIMPLE_QUERY_FEN_WITH_OPPO_MOVE,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}


PLAN_ACT_PROMPT_SET = {
    "system_prompt": CHESS_ACT_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": CHESS_ACT_QUERY_PROMPT,
    "plan_query_prompt": PLAN_QUERY_PROMPT,
    "plan_query_prompt_with_feedback": PLAN_QUERY_PROMPT_WITH_FEEDBACK,
    "plan_action_query_prompt": SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,
    "no_valid_decision_feedback": NO_DECISION_DETECTED
}

FEN_PLAN_ACT_PROMPT_SET={
    "system_prompt": FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": FEN_CHESS_ACT_QUERY_PROMPT,
    "plan_query_prompt": FEN_CHESS_PLAN_QUERY_PROMPT,
    "plan_query_prompt_with_feedback": FEN_CHESS_PLAN_QUERY_PROMPT_WITH_FEEDBACK,
    "plan_action_query_prompt": FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,
    "no_valid_decision_feedback": NO_DECISION_DETECTED
}

FEN_PLAN_ACT_PROMPT_SET_V2={
    "system_prompt": FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": FEN_CHESS_ACT_QUERY_PROMPT,
    "plan_query_prompt": FEN_CHESS_PLAN_QUERY_PROMPT_V2_NO_EXTRA_INFO,
    "plan_query_prompt_with_feedback": FEN_CHESS_PLAN_QUERY_PROMPT_WITH_FEEDBACK,
    "plan_action_query_prompt": FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,
    "no_valid_decision_feedback": NO_DECISION_DETECTED
}

SIMPLE_FEN_PLAN_ACT_PROMPT_SET={
    "system_prompt": SIMPLE_FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": SIMPLE_FEN_CHESS_ACT_QUERY_PROMPT,
    "plan_query_prompt": SIMPLE_FEN_CHESS_PLAN_QUERY_PROMPT,
    "plan_query_prompt_with_feedback": SIMPLE_FEN_CHESS_PLAN_QUERY_PROMPT_WITH_FEEDBACK,
    "plan_action_query_prompt": SIMPLE_FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,
    "no_valid_decision_feedback": NO_DECISION_DETECTED
}



#=========================== OSS =============================================++#



OSS_CHESS_POLICY_SIMPLE_QUERY="""
[Game Step {turn_idx}]
Game State:

{state}

Your available actions as White are:
{available_move}

DO NOT USE ANY TOOL!
"""


OSS_CHESS_POLICY_SIMPLEST_PROMPT="""
You are a professional Chess player playing as White. 

# Instruction
Your task is to analyze the given game record and choose the best next move as White.

# Board Format
In the provided position:
The board size is 8x8.
Uppercase letters (K, Q, R, B, N, P) represent white pieces.
Lowercase letters (k, q, r, b, n, p) represent black pieces.
A dot (.) represents an empty square.
The board is an 8x8 grid, with ranks (rows) numbered 1 to 8 (from White's perspective) and files (columns) labeled a to h (from left to right). A notated board is like this:

8| r n b q k b n r
7| p p p p p p p p
6| . . . . . . . .
5| . . . . . . . .
4| . . . . . . . .
3| . . . . . . . .
2| P P P P P P P P
1| R N B Q K B N R
   - - - - - - - -
   a b c d e f g h

An example of piece position is:
- White Pawns (P) positions are a2, b2, c2, d2, e2, f2, g2, h2 ;
- White Rooks (R) positions are a1, h1 ;
- White Knights (N) positions are b1, g1 ;
- White Bishops (B) positions are c1, f1 ;
- White Qween (Q) position is d1, White King (K) position is e1 ;

At each step, you will be given current board status, as well as current available action to choose.

# Action Format
Each valid move is the concatenation of previous position and current position after this move. For example:
For current board:

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R

Action (h2h3) will moves piece "P" on position h2 to new position h3, and the resulting board is:

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . P
P P P P P P P .
R N B Q K B N R

# Output Format
Your output should follow this format:

<think> Your thinking process </think> <answer>Your Move as White</answer>

Strictly follow the required format or otherwise you will be penalized.
"""



OSS_CHESS_PLAN_SIMPLEST_PROMPT="""
You are a professional Chess player playing as White. 

# Instruction
Your task is to analyze the given game record and choose the best next move as White.

# Board Format
In the provided position:
The board size is 8x8.
Uppercase letters (K, Q, R, B, N, P) represent white pieces.
Lowercase letters (k, q, r, b, n, p) represent black pieces.
A dot (.) represents an empty square.
The board is an 8x8 grid, with ranks (rows) numbered 1 to 8 (from White's perspective) and files (columns) labeled a to h (from left to right). A notated board is like this:

8| r n b q k b n r
7| p p p p p p p p
6| . . . . . . . .
5| . . . . . . . .
4| . . . . . . . .
3| . . . . . . . .
2| P P P P P P P P
1| R N B Q K B N R
   - - - - - - - -
   a b c d e f g h

An example of piece position is:
- White Pawns (P) positions are a2, b2, c2, d2, e2, f2, g2, h2 ;
- White Rooks (R) positions are a1, h1 ;
- White Knights (N) positions are b1, g1 ;
- White Bishops (B) positions are c1, f1 ;
- White Qween (Q) position is d1, White King (K) position is e1 ;

At each step, you will be given current board status, as well as current available action to choose.

# Action Format
Each valid move is the concatenation of previous position and current position after this move. For example:
For current board:

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R

Action (h2h3) will moves piece "P" on position h2 to new position h3, and the resulting board is:

r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . P
P P P P P P P .
R N B Q K B N R

During Simulation stage, you have three choice of actions:
1. Keep simulating at current state, in this case, output format: <think> Your thinking process </think> <move>Your Move as White</move>
2. Reset to the root state (at simulation step 1) to re-start simulation from the start, in this case, output format: <think> Your thinking process </think> <reset>1</reset>
3. End the simulation if you think you already know the best action at root state (at simulation step 1), in this case, output format: <think> Your thinking process </think> <end>1</end>
"""

OSS_CHESS_PLAN_SIMPLEST_PROMPT_V2="""
You are a professional Chess player playing as White. 

# Instruction
Your task is to analyze the given game record and choose the best next move as White.

# Board Format
In the provided position:
The board size is 8x8.
Uppercase letters (K, Q, R, B, N, P) represent white pieces.
Lowercase letters (k, q, r, b, n, p) represent black pieces.
A dot (.) represents an empty square.
The board is an 8x8 grid, with ranks (rows) numbered 1 to 8 (from White's perspective) and files (columns) labeled a to h (from left to right). A notated board is like this:

8| r n b q k b n r
7| p p p p p p p p
6| . . . . . . . .
5| . . . . . . . .
4| . . . . . . . .
3| . . . . . . . .
2| P P P P P P P P
1| R N B Q K B N R
   - - - - - - - -
   a b c d e f g h

An example of piece position is:
- White Pawns (P) positions are a2, b2, c2, d2, e2, f2, g2, h2 ;
- White Rooks (R) positions are a1, h1 ;
- White Knights (N) positions are b1, g1 ;
- White Bishops (B) positions are c1, f1 ;
- White Qween (Q) position is d1, White King (K) position is e1 ;

At each step, you will be given current board status, as well as current available action to choose.

# Action Format
Each valid move is the concatenation of previous position and current position after this move. For example, Action (h2h3) will moves piece "P" on position h2 to new position h3.

During Simulation stage, you have three choice of actions:
1. Keep simulating at current state, in this case, output format: <think> Your thinking process </think> <move>Your Move as White</move>
2. Reset to the root state (at simulation step 1) to re-start simulation from the start, in this case, output format: <think> Your thinking process </think> <reset>1</reset>
3. End the simulation if you think you already know the best action at root state (at simulation step 1), in this case, output format: <think> Your thinking process </think> <end>1</end>
"""

OSS_FEN_CHESS_ACT_QUERY_PROMPT="""\
[Game Step {turn_idx}]
Game State:

{state}

Your available actions as White are:
{available_move}
"""

OSS_FEN_CHESS_ACT_QUERY_PROMPT_CLEAN="""\
[Game Step {turn_idx}]
Game State:

{state}
"""

OSS_FEN_CHESS_ACT_QUERY_PROMPT_CLEAN_WITH_OPPO_MOVE="""\
[Game Step {turn_idx}]
The opponent played the move {oppo_move}.

Game State:

{state}
"""

OSS_FEN_CHESS_PLAN_QUERY_PROMPT_V3="""\
[Simulation turn {turn_idx}, Simulation step {step_idx}, Planning stage]

Game State:

{state}

Your available actions as White are:
{available_move}

Simulation budgets left:
{turn_left} turn budget left for resetting the simulation. {max_step} steps for planning ahead.
"""


OSS_FEN_CHESS_PLAN_QUERY_PROMPT_V3_WITH_FEEDBACK="""\
[Simulation turn {turn_idx}, Simulation step {step_idx}, Planning stage] ({extra_info}):

Game State:

{state}

Your available actions as White are:
{available_move}

Simulation budgets left:
{turn_left} turn budget left for resetting the simulation. {max_step} steps for planning ahead.
"""

OSS_FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT="""\
[Summarizing stage]

The simulation ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moves at simulation step 1. 

## State

{state}

## Output format
<think> Your thinking process </think> <answer>Your Move as White</answer>
"""

OSS_FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT_V2="""\
[Summarizing stage]

The simulation ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moves at simulation step 1. 
Refer to previous simulation history for possible outcome of the the possible move.

## State

{state}

## Output format
<think> Your thinking process </think> <answer>Your Move as White</answer>
"""

# ========================== Qwen Simplified LPM Inference ==========================#

QWEN_CHESS_PLAN_SIMPLEST_PROMPT="""
You are a professional Chess player playing as White. 

# Instruction
Your task is to analyze the given game record and choose the best next move as White.

# Board Format
In the provided position:
The board size is 8x8.
Uppercase letters (K, Q, R, B, N, P) represent white pieces.
Lowercase letters (k, q, r, b, n, p) represent black pieces.
A dot (.) represents an empty square.
The board is an 8x8 grid, with ranks (rows) numbered 1 to 8 (from White's perspective) and files (columns) labeled a to h (from left to right). A notated board is like this:

8| r n b q k b n r
7| p p p p p p p p
6| . . . . . . . .
5| . . . . . . . .
4| . . . . . . . .
3| . . . . . . . .
2| P P P P P P P P
1| R N B Q K B N R
   - - - - - - - -
   a b c d e f g h

An example of piece position is:
- White Pawns (P) positions are a2, b2, c2, d2, e2, f2, g2, h2 ;
- White Rooks (R) positions are a1, h1 ;
- White Knights (N) positions are b1, g1 ;
- White Bishops (B) positions are c1, f1 ;
- White Qween (Q) position is d1, White King (K) position is e1 ;

At each step, you will be given current board status, as well as current available action to choose.

# Action Format
Each valid move is the concatenation of previous position and current position after this move. For example, Action (h2h3) will moves piece "P" on position h2 to new position h3.

During Simulation stage, you need to output format: <think> Your thinking process </think> <move>Your Move as White</move> to keep simulating at current state.
During Summarizing stage, you need to output format: <think> Your thinking process </think> <answer>Your Move as White</answer> to summarize the best move at root state.
"""


QWEN_FEN_CHESS_PLAN_QUERY_PROMPT="""\
[Simulation turn {turn_idx}, Simulation step {step_idx}, Planning stage]

Game State:

{state}

Your available actions as White are:
{available_move}

Simulation budgets left:
{turn_left} turn budget left for resetting the simulation. {max_step} steps for planning ahead.

Output format: <think> Your thinking process </think> <move>Your Move as White</move>
"""

QWEN_FEN_CHESS_PLAN_QUERY_PROMPT_WITH_OPPO_MOVE="""\
[Simulation turn {turn_idx}, Simulation step {step_idx}, Planning stage]
The opponent played the move {oppo_move}.

Game State:

{state}

Your available actions as White are:
{available_move}

Simulation budgets left:
{turn_left} turn budget left for resetting the simulation. {max_step} steps for planning ahead.

Output format: <think> Your thinking process </think> <move>Your Move as White</move>
"""




QWEN_FEN_CHESS_PLAN_QUERY_PROMPT_WITH_FEEDBACK="""\
[Simulation turn {turn_idx}, Simulation step {step_idx}, Planning stage] ({extra_info}):

Game State:

{state}

Your available actions as White are:
{available_move}

Simulation budgets left:
{turn_left} turn budget left for resetting the simulation. {max_step} steps for planning ahead.

Strictly follow the action format given in the system prompt.
"""

INDEPENDENT_SUMMARY_INSTRUCTION="""\
You are a professional Chess player playing as White. 

# Instruction
Your task is to analysize the simulation history and summarize the best move at root state.
"""

INDEPENDENT_SUMMARY_PROMPT="""\
[Summarizing stage]

The simulation ends. Lets pause, check the simulation history, and summarize from the simulation history to get the best move at root state.

# Root State

{state}

# Simulation History

{history}

## Output format
<think> Your summary </think> <answer>Your Move as White at root state</answer>
"""

INDEPENDENT_SUMMARY_PROMPT_DETAILED="""\
[Summarizing stage]

The simulation ends. Your task is to act as a **Strategic Analyst** to critically evaluate the provided simulation history and determine the **optimal move for White at the Root State**.

### Analysis Guidelines:

1.  **Data Structure:** The simulation history contains multiple independent trajectories. Each trajectory is a sequence of turns represented by tuples:
    (state, action, opponent_action, environment_reward, next_state, game_terminated)

2.  **Sparse Reward Handling:** The environment_reward is **sparse** (often 0 for intermediate steps). **Do not** judge an action solely by an immediate zero reward. Instead, you must:
    * **Infer Value:** For intermediate steps, evaluate the **quality of next_state** relative to the goal. A high-quality action is one that leads to a winning or strongly advantageous position, even if the immediate reward is 0.

3.  **Root State Focus:** Your final decision **must be** a move playable *only* from the provided Root State.


### Root State

{state}

### Simulation History

{history}

### Output format
<think> Your summary </think> <answer>Your Move as White at root state</answer>
"""


VALUE_ITERATION_SUMMARY_PROMPT="""\
[Summarizing stage]

You are an evaluation agent analyzing a set of simulation trajectories that originate from the same root chess state.
Your goal is to conceptually emulate value-iteration-like reasoning:

### Analysis Guidelines

1. The simulation history contains multiple independent trajectories. Each trajectory is a sequence of turns represented by tuples:
    (state, action, opponent_action, environment_reward, next_state, game_terminated)
2. Use the simulation rollouts as noisy signals of the action quality.
3. Examine the successor states reached by each action.
4. Combine the rollout evidence with your chess understanding to estimate the long-term value of each root action.
5. You are not required to compute numerical values—only conceptual judgments.
6. The simulations may not contain wins or terminal rewards, so you must infer potential future value from position quality.

### Root State

{state}

### Simulation History

{history}

### Output format

<think>
(Your internal reasoning from analyzing the simulation history.
Evaluate each action in terms of successor position quality, strategic factors,
and inferred long-term value. Do NOT make calculations; use qualitative chess evaluation.
Do not repeat the summary or the final answer here.)
</think>

<summary>
(High-level synthesis of which root action is best.
Explain the main conceptual reasons why one action has the best long-term potential
based on the rollouts and chess knowledge.)
</summary>

<answer>Your Move as White at root state</answer>
"""


VALUE_ITERATION_SUMMARY_PROMPT_DETAILED="""\
[Summarizing stage]

You are an analysis agent. Your job is to evaluate actions at a root state in a chess position using a set of simulation trajectories.

These trajectories represent partial rollouts: they may not extend to the end of the game, so rewards may be missing or zero.

You must conceptually approximate the idea of value iteration: combining (1) simulation outcomes, (2) the quality of resulting states, and (3) your strategic understanding of chess to decide which root action is best.

### What you must do:

1. Read the root state.

2. Read each trajectory. For each action taken at the root:

 - Look at the next state(s) reached by simulations using that action.

 - Even if reward is 0, examine whether the resulting position is strategically better, worse, or unclear.

 - Evaluate qualitatively the “value” of each action by judging the likelihood of favorable long-term outcomes (winning chances, material, king safety, positional advantages).

3. Conceptually combine simulation evidence and chess evaluation.

 - Think like value iteration: the value of an action ≈ the potential value of the future states it leads to.

 - Use simulation outcomes as noisy signals of outcome quality, not final truth.

 - Favor actions whose simulated successor states appear strategically advantageous or promising.

4. Recommend the best action at the root, with a short explanation of why its future value appears highest.

### Important:

- Do not do numerical calculations or probabilities.

- Instead, produce a qualitative, strategic ranking.

- Use expert-level chess reasoning to assess the successor states.

### Root State

{state}

### Simulation History

{history}

### Output format

<think>
(Your internal reasoning from analyzing the simulation history.
Evaluate each action in terms of successor position quality, strategic factors,
and inferred long-term value. Do NOT make calculations; use qualitative chess evaluation.
Do not repeat the summary or the final answer here.)
</think>

<summary>
(High-level synthesis of which root action is best.
Explain the main conceptual reasons why one action has the best long-term potential
based on the rollouts and chess knowledge.)
</summary>

<answer>
BEST_ACTION_HERE
</answer>
"""






OSS_NO_TOOL_DIRECT_PROMPT_SET = {
    "system_prompt": CHESS_POLICY_SIMPLE_PROMPT,
    "query_prompt": OSS_CHESS_POLICY_SIMPLE_QUERY,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}

OSS_SIMPLEST_DIRECT_PROMPT_SET = {
    "system_prompt": OSS_CHESS_POLICY_SIMPLEST_PROMPT,
    "query_prompt": OSS_CHESS_POLICY_SIMPLE_QUERY,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
}


OSS_FEN_PLAN_ACT_PROMPT_SET={
    "system_prompt": OSS_FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": FEN_CHESS_ACT_QUERY_PROMPT,
    "plan_query_prompt": OSS_FEN_CHESS_PLAN_QUERY_PROMPT,   # extra
    "plan_query_prompt_with_feedback": OSS_FEN_CHESS_PLAN_QUERY_PROMPT_WITH_FEEDBACK,   # extra
    "plan_action_query_prompt": FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,       # extra
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,   # extra
    "no_valid_decision_feedback": NO_DECISION_DETECTED
}

OSS_SIMPLEST_PLAN_PROMPT_SET={
    "system_prompt": OSS_CHESS_PLAN_SIMPLEST_PROMPT,
    "act_query_prompt": OSS_FEN_CHESS_ACT_QUERY_PROMPT,
    "plan_query_prompt": OSS_FEN_CHESS_PLAN_QUERY_PROMPT_V3,
    "plan_query_prompt_with_feedback": OSS_FEN_CHESS_PLAN_QUERY_PROMPT_V3_WITH_FEEDBACK,
    "plan_action_query_prompt":OSS_FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,  # extra
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,  # extra
    "no_valid_decision_feedback": NO_DECISION_DETECTED,
}


OSS_SIMPLEST_PLAN_PROMPT_SET_V2={
    "system_prompt": OSS_CHESS_PLAN_SIMPLEST_PROMPT_V2,
    "act_query_prompt": OSS_FEN_CHESS_ACT_QUERY_PROMPT,
    "plan_query_prompt": OSS_FEN_CHESS_PLAN_QUERY_PROMPT_V3,
    "plan_query_prompt_with_feedback": OSS_FEN_CHESS_PLAN_QUERY_PROMPT_V3_WITH_FEEDBACK,
    "plan_action_query_prompt":OSS_FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,  # extra
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,  # extra
    "no_valid_decision_feedback": NO_DECISION_DETECTED,
}

QWEN_SIMPLEST_PLAN_PROMPT_SET={
    "system_prompt": QWEN_CHESS_PLAN_SIMPLEST_PROMPT,
    "act_query_prompt": OSS_FEN_CHESS_ACT_QUERY_PROMPT_CLEAN,
    "act_query_prompt_with_info": OSS_FEN_CHESS_ACT_QUERY_PROMPT_CLEAN_WITH_OPPO_MOVE,
    "plan_query_prompt": QWEN_FEN_CHESS_PLAN_QUERY_PROMPT,
    "plan_query_prompt_with_info": QWEN_FEN_CHESS_PLAN_QUERY_PROMPT_WITH_OPPO_MOVE,
    "plan_query_prompt_with_feedback": QWEN_FEN_CHESS_PLAN_QUERY_PROMPT_WITH_FEEDBACK,
    "plan_action_query_prompt":OSS_FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT_V2,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,  # extra
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,  # extra
    "no_valid_decision_feedback": NO_DECISION_DETECTED,
}

QWEN_PREVIOUS_PLAN_PROMPT_SET={
    "system_prompt": FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": OSS_FEN_CHESS_ACT_QUERY_PROMPT_CLEAN,
    "act_query_prompt_with_info": OSS_FEN_CHESS_ACT_QUERY_PROMPT_CLEAN_WITH_OPPO_MOVE,
    "plan_query_prompt": QWEN_FEN_CHESS_PLAN_QUERY_PROMPT,
    "plan_query_prompt_with_info": QWEN_FEN_CHESS_PLAN_QUERY_PROMPT_WITH_OPPO_MOVE,
    "plan_query_prompt_with_feedback": QWEN_FEN_CHESS_PLAN_QUERY_PROMPT_WITH_FEEDBACK,
    "plan_action_query_prompt":FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,  # extra
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,  # extra
    "no_valid_decision_feedback": NO_DECISION_DETECTED,
}


QWEN_SHORT_PLAN_PROMPT_SET={
    "system_prompt": FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": FEN_CHESS_ACT_QUERY_PROMPT_V2,
    "plan_query_prompt": FEN_CHESS_PLAN_QUERY_PROMPT_V3,
    "plan_query_prompt_with_info": FEN_CHESS_PLAN_QUERY_PROMPT_V3_WITH_OPPO_MOVE,
    "plan_action_query_prompt": FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "tie_prompt": TIE_PROMPT,
    "plan_tie_prompt": PLAN_TIE_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,
    "no_valid_decision_feedback": NO_DECISION_DETECTED,
    "independent_summary_instruction": INDEPENDENT_SUMMARY_INSTRUCTION,
    "independent_summary_prompt": INDEPENDENT_SUMMARY_PROMPT,
}

QWEN_SHORT_DETAILED_SUMMARY_PLAN_PROMPT_SET={
    "system_prompt": FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": FEN_CHESS_ACT_QUERY_PROMPT_V2,
    "plan_query_prompt": FEN_CHESS_PLAN_QUERY_PROMPT_V3,
    "plan_query_prompt_with_info": FEN_CHESS_PLAN_QUERY_PROMPT_V3_WITH_OPPO_MOVE,
    "plan_action_query_prompt": FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "no_valid_action_end_game_fail_prompt": NO_VALID_ACTION_END_GAME_FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "tie_prompt": TIE_PROMPT,
    "plan_tie_prompt": PLAN_TIE_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,
    "no_valid_decision_feedback": NO_DECISION_DETECTED,
    "independent_summary_instruction": INDEPENDENT_SUMMARY_INSTRUCTION,
    "independent_summary_prompt": INDEPENDENT_SUMMARY_PROMPT_DETAILED,
}


QWEN_VALUE_SUMMARY_PLAN_PROMPT_SET={
    "system_prompt": FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": FEN_CHESS_ACT_QUERY_PROMPT_V2,
    "plan_query_prompt": FEN_CHESS_PLAN_QUERY_PROMPT_V3,
    "plan_query_prompt_with_info": FEN_CHESS_PLAN_QUERY_PROMPT_V3_WITH_OPPO_MOVE,
    "plan_action_query_prompt": FEN_CHESS_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "no_valid_action_end_game_fail_prompt": NO_VALID_ACTION_END_GAME_FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "tie_prompt": TIE_PROMPT,
    "plan_tie_prompt": PLAN_TIE_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,
    "no_valid_decision_feedback": NO_DECISION_DETECTED,
    "independent_summary_instruction": INDEPENDENT_SUMMARY_INSTRUCTION,
    "independent_summary_prompt": VALUE_ITERATION_SUMMARY_PROMPT,
}


# ========================== Qwen Two-Stage Value Inference ==========================#

TWO_STAGE_VALUE_SYSTEM_PROMPT="""\
A conversation between User and Assistant. 
The User asks the best move to make for a given chess board state, and the Assistant solves it. 
The Assistant is a professional chess player who can plan a few steps ahead by interacting with a simulator, and then think about the reasoning process and at last provides the User with the answer.

The Assistant's thinking process must include an understanding of the game state, based on which the Assistant can later decide on the next move.
The understanding should describe how the Assistant analyzes the current position, including:
- A strategic evaluation of the position.
- A comparison of key candidate moves (not all of them) in terms of their possible subsequent outcomes.
- Do not try to enumerate all possible moves and outcomes, only consider the most likely ones.

The decision on next move must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).

Reminder of chess rules: 
- Bishops move diagonally. 
- Rooks move horizontally or vertically. 
- Knights jump in an L-shape. 
- Queens combine rook and bishop movement. 
- Kings move one square in any direction. 
- Pawns move forward, capture diagonally, and can promote.
"""

TWO_STAGE_VALUE_SYSTEM_PROMPT_V2="""\
A conversation between User and Assistant. 
The User asks the best move to make for a given chess board state, and the Assistant solves it. 
The Assistant is a professional chess player who think about the reasoning process and at last provides the User with the answer.

The Assistant's thinking process must include an understanding of the game state, based on which the Assistant can later decide on the next move.
The understanding should describe how the Assistant analyzes the current position, including:
- A strategic evaluation of the position.
- A comparison of key candidate moves (not all of them) in terms of their possible subsequent outcomes.
- Do not try to enumerate all possible moves and outcomes, only consider the most likely ones.

The decision on next move must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).

Reminder of chess rules: 
- Bishops move diagonally. 
- Rooks move horizontally or vertically. 
- Knights jump in an L-shape. 
- Queens combine rook and bishop movement. 
- Kings move one square in any direction. 
- Pawns move forward, capture diagonally, and can promote.
"""


TWO_STAGE_VALUE_ACT_QUERY_PROMPT="""\
Current Game State:
{state}
Legal moves:
{available_move}
"""

TWO_STAGE_VALUE_ACT_QUERY_PROMPT_WITH_OPPO_MOVE="""
The opponent played the move {oppo_move}.

Current Game State:
{state}
Legal moves:
{available_move}
"""

TWO_STAGE_VALUE_PLAN_QUERY_PROMPT="""\
Simulation turn {turn_idx}, step {step_idx}:

Game State of the simulator:
{state}
Available moves as White are:
{available_move}

You have {turn_left} turn budget left for resetting the simulation. You can now plan for maximum {max_step} number of steps ahead. 
"""

TWO_STAGE_VALUE_PLAN_QUERY_PROMPT_WITH_OPPO_MOVE="""\
Simulation turn {turn_idx}, step {step_idx}:
The opponent played the move {oppo_move}.

Game State of the simulator:
{state}
Available moves as White are:
{available_move}

You have {turn_left} turn budget left for resetting the simulation. You can now plan for maximum {max_step} number of steps ahead. 
"""

TWO_STAGE_VALUE_SUMMARIZE_PLAN_QUERY_PROMPT="""\
The simulation ends. Lets pause, check the interaction history, and rethink again both the benefits and harm of each possible moves at simulation step 1. 
"""


STATE_VALUE_QUERY_PROMPT="""\
What is your understanding of the game state:

{state}
"""

SIMULATION_STATE_VALUE_QUERY_PROMPT="""\
Based on the simulation history, what is your updated understanding of the game state:

{state}
"""


STATE_MOVE_QUERY_PROMPT="""\
You are now in simulation, your current goal is to move in order to better understand potential future outcomes of the game, so you can either explore or exploit.

Based on your previous understanding of the game state and your current goal, what is the decided move?

Output your move in a form of <move>[Your Move as White]</move>
"""

STATE_ANSWER_QUERY_PROMPT="""\
The simulation ends, your current goal is to output the optimal move for the root state.

Based on previous simulation, your understanding of the game state and your current goal, what is the best move to make?

Output your move in a form of <answer>[Your Move as White]</answer>
"""

STATE_ANSWER_DIRECT_PROMPT="""
Based on your understanding of the game state, what is the decided move?

Output your move in a form of <answer>[Your Move as White]</answer>
"""  # for direct inference


TWO_STAGE_VALUE_SUMMARY_PROMPT="""\
[Summarizing stage]

You are an evaluation agent analyzing a set of simulation trajectories that originate from the same root chess state.
Your goal is to conceptually emulate value-iteration-like reasoning:

### Analysis Guidelines

1. The simulation history contains multiple independent trajectories. Each trajectory is a sequence of turns represented by tuples:
    (state, action, opponent_action, environment_reward, next_state, game_terminated)
2. Use the simulation rollouts as noisy signals of the action quality.
3. Examine the successor states reached by each action.
4. Combine the rollout evidence with your chess understanding to estimate the long-term value of each root action.
5. You are not required to compute numerical values—only conceptual judgments.
6. The simulations may not contain wins or terminal rewards, so you must infer potential future value from position quality.

### Simulation History

{history}
"""


QWEN_TWO_STAGE_VALUE_PROMPT_SET={
    "system_prompt": TWO_STAGE_VALUE_SYSTEM_PROMPT,
    "act_query_prompt": TWO_STAGE_VALUE_ACT_QUERY_PROMPT,
    "plan_query_prompt": TWO_STAGE_VALUE_PLAN_QUERY_PROMPT,
    "plan_query_prompt_with_info": TWO_STAGE_VALUE_PLAN_QUERY_PROMPT_WITH_OPPO_MOVE,
    "plan_action_query_prompt": TWO_STAGE_VALUE_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "no_valid_action_end_game_fail_prompt": NO_VALID_ACTION_END_GAME_FAIL_PROMPT,
    "plan_success_prompt": PLAN_SUCCESS_PROMPT,
    "tie_prompt": TIE_PROMPT,
    "plan_tie_prompt": PLAN_TIE_PROMPT,
    "plan_fail_prompt": PLAN_FAIL_PROMPT,
    "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,
    "no_valid_decision_feedback": NO_DECISION_DETECTED,
    # "independent_summary_instruction": INDEPENDENT_SUMMARY_INSTRUCTION,
    "independent_summary_prompt": TWO_STAGE_VALUE_SUMMARY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "simulation_state_value_query_prompt": SIMULATION_STATE_VALUE_QUERY_PROMPT,
    "state_move_query_prompt": STATE_MOVE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_QUERY_PROMPT,
}

QWEN_TWO_STAGE_VALUE_DIRECT_PROMPT_SET = {
    "system_prompt": TWO_STAGE_VALUE_SYSTEM_PROMPT,
    "query_prompt": TWO_STAGE_VALUE_ACT_QUERY_PROMPT,
    "query_prompt_with_info": TWO_STAGE_VALUE_ACT_QUERY_PROMPT_WITH_OPPO_MOVE,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_DIRECT_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}

QWEN_TWO_STAGE_VALUE_DIRECT_PROMPT_SET_V2 = {
    "system_prompt": TWO_STAGE_VALUE_SYSTEM_PROMPT_V2,
    "query_prompt": TWO_STAGE_VALUE_ACT_QUERY_PROMPT,
    "query_prompt_with_info": TWO_STAGE_VALUE_ACT_QUERY_PROMPT_WITH_OPPO_MOVE,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_DIRECT_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}


VALUE_BASED_CHESS_POLICY_SIMPLE_PROMPT_FEN="""
A conversation between User and Assistant. 
The User asks the best move to make for a given chess board state, and the Assistant solves it. 
The Assistant is a professional chess player who think about the reasoning process and at last provides the User with the answer.


The Assistant’s reasoning process and answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively. 
The reasoning process should describe the understanding of the state and decide on the best move, including:
- A strategic evaluation of the position. 
- A comparison of only three key candidate moves. 
- For each candidate, consider the opponent’s likely response and outcome. 
- Conclude with a clear justification for your final choice.

The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Reminder of chess rules: 
- Bishops move diagonally. 
- Rooks move horizontally or vertically. 
- Knights jump in an L-shape. 
- Queens combine rook and bishop movement. 
- Kings move one square in any direction. 
- Pawns move forward, capture diagonally, and can promote.
"""

VALUE_BASED_CHESS_POLICY_SIMPLE_QUERY_FEN="""
Current Game State:
{state}
Legal moves:
{available_move}
What is your understanding of the current state and What is the best move to make out of the list of legal moves?
"""

VALUE_BASED_CHESS_POLICY_SIMPLE_QUERY_FEN_WITH_OPPO_MOVE="""
The opponent played the move {oppo_move}.
Current Game State:
{state}
Legal moves:
{available_move}
What is your understanding of the current state and What is the best move to make out of the list of legal moves?
"""

VALUE_BASED_FEN_DIRECT_PROMPT_SET = {
    "system_prompt": VALUE_BASED_CHESS_POLICY_SIMPLE_PROMPT_FEN,
    "query_prompt": VALUE_BASED_CHESS_POLICY_SIMPLE_QUERY_FEN,
    "query_prompt_with_info": VALUE_BASED_CHESS_POLICY_SIMPLE_QUERY_FEN_WITH_OPPO_MOVE,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}


# ========================== Qwen Two-Stage Value Inference with Parallel Planning ==========================#

VALUE_CENTRIC_PARALLEL_PLAN_SYSTEM_PROMPT="""\
A conversation between User and Assistant. 
The User asks the best move to make for a given chess board state, and the Assistant solves it. 
The Assistant is a professional chess player who can plan a few steps ahead by interacting with a simulator, and then think about the reasoning process and at last provides the User with the answer.

The Assistant's thinking process must include an understanding of the game state, based on which the Assistant can later decide on the next move.
The understanding should describe how the Assistant analyzes the current position, including:
- A strategic evaluation of the position.
- A comparison of key candidate moves (not all of them) in terms of their possible subsequent outcomes.
- Do not try to enumerate all possible moves and outcomes, only consider the most likely ones.

The decision on next move must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).

Reminder of chess rules: 
- Bishops move diagonally. 
- Rooks move horizontally or vertically. 
- Knights jump in an L-shape. 
- Queens combine rook and bishop movement. 
- Kings move one square in any direction. 
- Pawns move forward, capture diagonally, and can promote.
"""

VALUE_CENTRIC_PARALLEL_ACT_QUERY_PROMPT="""\
Current Game State:
{state}
Legal moves:
{available_move}
"""


VALUE_CENTRIC_PARALLEL_PLAN_QUERY_PROMPT="""\
Current Simulation State:
{state}
Legal moves:
{available_move}
"""

VALUE_CENTRIC_PARALLEL_PLAN_QUERY_PROMPT_WITH_OPPO_MOVE="""
The opponent played the move {oppo_move}.

Current Simulation State:
{state}
Legal moves:
{available_move}
"""

VALUE_CENTRIC_PARALLEL_PLAN_PLAN_QUERY_PROMPT="""\
Simulation turn {turn_idx}, step {step_idx}:

Game State of the simulator:
{state}
Available moves as White are:
{available_move}

You can now plan for maximum {max_step} number of steps ahead. You can generate {num_actions} independent parallel actions for each state.
"""

PARALLEL_PLAN_STATE_MOVE_QUERY_PROMPT="""\
You are now in simulation, your current goal is to move in order to better understand potential future outcomes of the game, so you can either explore or exploit. 
You can generate at most {num_actions} candidate moves to be tested with the simulator.
Do not generate duplicate moves.
Choose the move from available move list:
{available_move}

Based on your previous understanding of the game state and your current goal, generate moves in a form of:

<move>[Your First Candidate Move as White]</move>
<move>[Your Second Candidate Move as White]</move>
...
"""

PLAIN_PLAN_SUCCESS_PROMPT="""\
Results: You have successfully solve the game within this simulation!
"""
# for success move, no need to output the next state

PLAIN_PLAN_FAIL_PROMPT="""\
The next state of current simulation:
{state}
Results: You lose the game within this simulation.
"""

PLAIN_PLAN_TIE_PROMPT="""\
The next state of current simulation:
{state}
Results: You tie the game within this simulation.
"""


QWEN_TWO_STAGE_VALUE_PARALLEL_PLAN_PROMPT_SET={
    "system_prompt": VALUE_CENTRIC_PARALLEL_PLAN_SYSTEM_PROMPT,
    "act_query_prompt": VALUE_CENTRIC_PARALLEL_ACT_QUERY_PROMPT,
    "plan_query_prompt": VALUE_CENTRIC_PARALLEL_PLAN_QUERY_PROMPT,
    "plan_query_prompt_with_info": VALUE_CENTRIC_PARALLEL_PLAN_QUERY_PROMPT_WITH_OPPO_MOVE,
    "plan_action_query_prompt": TWO_STAGE_VALUE_SUMMARIZE_PLAN_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT,
    "plan_success_prompt": PLAIN_PLAN_SUCCESS_PROMPT,
    "tie_prompt": TIE_PROMPT,
    "plan_tie_prompt": PLAIN_PLAN_TIE_PROMPT,
    "plan_fail_prompt": PLAIN_PLAN_FAIL_PROMPT,
    # "plan_fail_prompt_with_feedback": PLAN_FAIL_PROMPT_WITH_FEEDBACK,
    # "independent_summary_instruction": INDEPENDENT_SUMMARY_INSTRUCTION,
    "independent_summary_prompt": TWO_STAGE_VALUE_SUMMARY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "simulation_state_value_query_prompt": SIMULATION_STATE_VALUE_QUERY_PROMPT,
    "state_move_query_prompt": PARALLEL_PLAN_STATE_MOVE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_QUERY_PROMPT,
}


# ========================== Qwen SFT Loss ==========================#

SFT_SYSTEM_PROMPT="""\
A conversation between User and Assistant. 
The User asks the best move to make for a given chess board state, and the Assistant solves it. 
The Assistant is a professional chess player who can imagine the possible moves and outcomes of the game, and then provide the User with the best move.

The Assistant's thinking process and movement must be the form of: <think>[Your thoughts]</think><answer>[Your Move as White]</answer>

The answer must be in SAN notation, strictly using the moving piece and the destination square (e.g., Nf3, Rxf2, c5).
Reminder of chess rules: 
- Bishops move diagonally. 
- Rooks move horizontally or vertically. 
- Knights jump in an L-shape. 
- Queens combine rook and bishop movement. 
- Kings move one square in any direction. 
- Pawns move forward, capture diagonally, and can promote.

Strictly follow the required format or otherwise you will be penalized.
"""

SFT_QUERY_PROMPT="""
Game State:
{state}

Your available actions as White are:
{available_move}
"""


QWEN_CHESS_SFT_PROMPT_SET={
    "system_prompt": SFT_SYSTEM_PROMPT,
    "query_prompt": SFT_QUERY_PROMPT,
}

QWEN_ALIGNED_CHESS_SFT_PROMPT_SET={
    "system_prompt": FEN_CHESS_ACT_PLAN_SYSTEM_PROMPT,
    "query_prompt": SFT_QUERY_PROMPT,
    }



# ====================== Qwen Auxiliary inference ==========================#


QWEN_TWO_STAGE_VALUE_AUXILIARY_DIRECT_PROMPT_SET = {
    "system_prompt": TWO_STAGE_VALUE_SYSTEM_PROMPT,
    "query_prompt": TWO_STAGE_VALUE_ACT_QUERY_PROMPT,
    "query_prompt_with_info": TWO_STAGE_VALUE_ACT_QUERY_PROMPT,
    "state_value_query_prompt": STATE_VALUE_QUERY_PROMPT,
    "state_answer_query_prompt": STATE_ANSWER_QUERY_PROMPT,
    "success_prompt": SUCCESS_PROMPT,
    "fail_prompt": FAIL_PROMPT
}
