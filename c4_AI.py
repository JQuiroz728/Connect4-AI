import pygame
import numpy as np
import sys
import math
import random

# Constants
# Pygame colors
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
YELLOW = (255,255,0)
BLUE = (0,0,255)
SILVER = (204,229,255)
# Board dimensions
ROWS = 6
COLUMNS = 7
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)
width = COLUMNS * SQUARESIZE
height = (ROWS + 1) * SQUARESIZE
size = (width, height)

PLAYER = 0
AI = 1

PLAYER_PIECE = 1
AI_PIECE = 2


# Game functions
def generate_board():
	board = np.zeros((ROWS, COLUMNS))
	return board

def place_piece(board, row, col, piece):
	board[row][col] = piece

def is_valid_placement(board, col):
	return board[ROWS - 1][col] == 0

def get_next_open_row(board, col):
	for row in range(ROWS):
		if board[row][col] == 0:
			return row

def display_board(board):
	print(np.flip(board, 0))


# Determine if a player has won by verifying if a winning position has been achieved (4 consecutive pieces)
def winning_move(board, piece):
	# Horizontal win
	for col in range(COLUMNS - 3):
		for row in range(ROWS):
			if board[row][col] == piece and board[row][col + 1] == piece and board[row][col + 2] == piece and board[row][col + 3] == piece:
				return True

	# Vertical win
	for col in range(COLUMNS):
		for row in range(ROWS - 3):
			if board[row][col] == piece and board[row + 1][col] == piece and board[row + 2][col] == piece and board[row + 3][col] == piece:
				return True

	# Positively sloped diagonal
	for col in range(COLUMNS - 3):
		for row in range(ROWS - 3):
			if board[row][col] == piece and board[row + 1][col + 1] == piece and board[row + 2][col + 2] == piece and board[row + 3][col + 3] == piece:
				return True

	# Negatively sloped diagonal
	for col in range(COLUMNS - 3):
		for row in range(3, ROWS):
			if board[row][col] == piece and board[row - 1][col + 1] == piece and board[row - 2][col + 2] == piece and board[row - 3][col + 3] == piece:
				return True

# Updates GUI to display current gamestate
def draw_board(board):
	for col in range(COLUMNS):
		for row in range(ROWS):
			pygame.draw.rect(screen, BLUE, (col * SQUARESIZE, row * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
			pygame.draw.circle(screen, SILVER, (int(col * SQUARESIZE + SQUARESIZE / 2), int(row * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)
	
	for col in range(COLUMNS):
		for row in range(ROWS):		
			if board[row][col] == PLAYER_PIECE:
				pygame.draw.circle(screen, RED, (int(col * SQUARESIZE + SQUARESIZE / 2), height - int(row * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
			elif board[row][col] == AI_PIECE: 
				pygame.draw.circle(screen, YELLOW, (int(col * SQUARESIZE + SQUARESIZE / 2), height - int(row * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
	pygame.display.update()


# AI Scoring Heuristic to determine current gamestate and best possible move
# Score values are assigned here and can be tweaked to optimize AI's performance
def score_window(window, piece):
	score = 0
	opponent = PLAYER_PIECE
	if piece == PLAYER_PIECE:
		opponent = AI_PIECE 

	if window.count(piece) == 4:
		score += 100
	elif window.count(piece) == 3 and window.count(0) == 1:
		score += 5
	elif window.count(piece) == 2 and window.count(0) == 2:
		score += 2

	if window.count(opponent) == 3 and window.count(0) == 1:
		score -= 4

	return score


def score_gamestate(board, piece):
	# Prioritize ways to win

	score = 0
	# Prioritize Center Column Placements
	center_array = [int(i) for i in list(board[:, COLUMNS // 2])]
	center_count = center_array.count(piece)
	score += center_count * 3

	# Score Horizontal Positions
	for row in range(ROWS):
		row_array = [int(i) for i in list(board[row, :])]
		for col in range(COLUMNS - 3):
			window = row_array[col : col + 4]
			score += score_window(window, piece)

	## Score Vertical Positions
	for col in range(COLUMNS):
		col_array = [int(i) for i in list(board[:, col])]
		for row in range(ROWS - 3):
			window = col_array[row : row + 4]
			score += score_window(window, piece)

	## Score Positive Sloped Diagonals
	for row in range(ROWS - 3):
		for col in range(COLUMNS - 3):
			window = [board[row + i][col + i] for i in range(4)]
			score += score_window(window, piece)

	## Score Negative Sloped Diagonals
	for row in range(ROWS - 3):
		for col in range(COLUMNS - 3):
			window = [board[row + 3 - i][col + i] for i in range(4)]
			score += score_window(window, piece)

	return score

# Minimax algorithm with alpha-beta pruning implementation
def is_terminal_node(board):
	return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizingPlayer):
	locations = valid_locations(board)
	is_terminal = is_terminal_node(board)
	if depth == 0 or is_terminal:
		if is_terminal:
			if winning_move(board, AI_PIECE):
				return (None, 10000000000)
			elif winning_move(board, PLAYER_PIECE):
				return (None, -10000000000)
			else:
				return (None, 0) # Game over, no more possible moves
		else: # Depth is 0
			return (None, score_gamestate(board, AI_PIECE))

	if maximizingPlayer:
		value = -math.inf
		column = random.choice(locations)
		for col in locations:
			row = get_next_open_row(board, col)
			temp_board = board.copy()
			place_piece(temp_board, row, col, AI_PIECE)
			new_score = minimax(temp_board, depth - 1, alpha, beta, False)[1]
			if new_score > value:
				value = new_score
				column = col
			alpha = max(alpha, value)
			if alpha >= beta:
				break

		return column, value

	else: # Minimizing Player
		value = math.inf
		column = random.choice(locations)
		for col in locations:
			row = get_next_open_row(board, col)
			temp_board = board.copy()
			place_piece(temp_board, row, col, PLAYER_PIECE)
			new_score = minimax(temp_board, depth - 1, alpha, beta, True)[1]
			if new_score < value:
				value = new_score
				column = col
			beta = min(beta, value)
			if alpha >= beta:
				break

		return column, value


def valid_locations(board):
	locations = []
	for col in range(COLUMNS):
		if is_valid_placement(board, col):
			locations.append(col)
	return locations
		
# Finds best possible move by assesing score values
def best_move(board, piece):
	best_score = -10000
	possible_locations = valid_locations(board)
	best_col = random.choice(possible_locations)
	for col in possible_locations:
		row = get_next_open_row(board, col)
		temp_board = board.copy()
		place_piece(temp_board, row, col, piece)
		score = score_gamestate(temp_board, piece)
		if score > best_score:
			best_score = score
			best_col = col
	return best_col


# Game Setup
board = generate_board()
game_over = False

pygame.init()

pygame.display.set_caption('Connect 4')
screen = pygame.display.set_mode(size)
draw_board(board)
pygame.display.update()

game_font = pygame.font.SysFont("monospace", 75)

turn = random.randint(PLAYER, AI)


# Game loop
while not game_over:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()

		if event.type == pygame.MOUSEMOTION:
			pygame.draw.rect(screen, SILVER, (0, 0, width, SQUARESIZE))
			pos_x = event.pos[0]
			if turn == PLAYER:
				pygame.draw.circle(screen, RED, (pos_x, int(SQUARESIZE / 2)), RADIUS)
		pygame.display.update()

		if event.type == pygame.MOUSEBUTTONDOWN:
			pygame.draw.rect(screen, SILVER, (0, 0, width, SQUARESIZE))
			# Player's turn
			if turn == PLAYER:
				pos_x = event.pos[0]
				col = int(math.floor(pos_x / SQUARESIZE))

				if is_valid_placement(board, col):
					row = get_next_open_row(board, col)
					place_piece(board, row, col, PLAYER_PIECE)

					if winning_move(board, PLAYER_PIECE):
						label = game_font.render("Red Wins!", 1, RED)
						screen.blit(label, (150,10))
						game_over = True

					turn += 1
					turn = turn % 2

					draw_board(board)

	# AI's turn
	if turn == AI and not game_over:				
		# AI Intelligence Levels
		# col = random.randint(0, COLUMNS - 1)    -- Random placement : Beginner
		# col = best_move(board, AI_PIECE)		  -- Selects best move based on analyzing current possible moves and corresponding score values : Intermediate
		col, minimax_score = minimax(board, 5, -math.inf, math.inf, True) # -- Utilizing the minimax algorithm with alpha-beta pruning : Expert

		if is_valid_placement(board, col):
			row = get_next_open_row(board, col)
			place_piece(board, row, col, AI_PIECE)

			if winning_move(board, AI_PIECE):
				label = game_font.render("Yellow Wins!", 1, YELLOW)
				screen.blit(label, (100,10))
				game_over = True

			draw_board(board)

			turn += 1
			turn = turn % 2

	if game_over:
		pygame.time.wait(5000)