import STcpClient
import numpy as np
import random
import copy
import math


def get_center_position(board):
    rows = len(board)
    cols = len(board[0])
    return rows // 2, cols // 2


def choose_starting_position(board):
    obstacle_positions = []
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == -1:
                obstacle_positions.append((i, j))

    max_openness = 0
    best_starting_point = None

    for obstacle_position in obstacle_positions:
        i, j = obstacle_position
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_i = i + di
            new_j = j + dj
            if 0 <= new_i < len(board) and len(board[0]) > new_j >= 0 == board[new_i][new_j]:
                openness = sum(1 for di in [-1, 0, 1] for dj in [-1, 0, 1] if
                               0 <= new_i + di < len(board) and len(board[0]) > new_j + dj >= 0 == board[new_i + di][new_j + dj])
                if openness > max_openness:
                    max_openness = openness
                    best_starting_point = (new_i, new_j)

    return best_starting_point


def calculate_score(board, player_id):
    visited = set()
    score = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == player_id and (i, j) not in visited:
                region_size = dfs(board, player_id, i, j, visited)
                score += math.pow(region_size, 1.25)
    return round(score)


def dfs(board, player_id, i, j, visited):
    if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != player_id or (i, j) in visited:
        return 0
    visited.add((i, j))
    size = 1
    size += dfs(board, player_id, i + 1, j, visited)
    size += dfs(board, player_id, i - 1, j, visited)
    size += dfs(board, player_id, i, j + 1, visited)
    size += dfs(board, player_id, i, j - 1, visited)
    return size


def calculate_score_(board, player_id):
    score = 0
    for row in board:
        for cell in row:
            if cell == player_id:
                score += 1
    return score


def possible_moves(board, sheep, player_id):
    moves = []
    dirs = [(j, i) for i in range(-1, 2) for j in range(-1, 2)]
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == player_id and sheep[i][j] > 1:
                for idx, direction in enumerate(dirs):
                    if direction == (0, 0):
                        continue
                    new_i, new_j = i, j
                    steps = 0
                    while 0 <= new_i + direction[0] < len(board) and 0 <= new_j + direction[1] < len(board[0]) and (
                            board[new_i + direction[0]][new_j + direction[1]] == 0):
                        new_i += direction[0]
                        new_j += direction[1]
                        steps += 1
                    if steps > 0:
                        moves.append(((i, j), (new_i, new_j), idx + 1))

    return moves


def make_move(board, sheep, move):
    start_pos, end_pos, _ = move
    num_sheep = sheep[start_pos[0]][start_pos[1]] // 2
    board[end_pos[0]][end_pos[1]] = board[start_pos[0]][start_pos[1]]
    sheep[start_pos[0]][start_pos[1]] -= num_sheep
    sheep[end_pos[0]][end_pos[1]] += num_sheep
    return board, sheep


def random_simulation(board, sheep, depth, player_id, target_player_id, num_players):
    if depth == 0:
        return calculate_score(board, player_id)

    moves = possible_moves(board, sheep, player_id)
    if not moves:
        return calculate_score(board, player_id) *  (-1 if player_id != target_player_id else 1)

    move = random.choice(moves)
    next_player_id = (player_id % num_players) + 1

    new_board, new_sheep = make_move(copy.deepcopy(board), copy.deepcopy(sheep), move)
    value = random_simulation(new_board, new_sheep, depth - (1 if next_player_id == target_player_id else 0), next_player_id, target_player_id, num_players)

    return value


def monte_carlo_tree_search_ucb(board, sheep, player_id, num_iterations, exploration_param=math.sqrt(2), num_players=4):
    root_node = {
        'board': copy.deepcopy(board),
        'sheep': copy.deepcopy(sheep),
        'player_id': player_id,
        'num_iterations': num_iterations,
        'parent': None,
        'moves': None,
        'children': dict()
    }

    root_node['moves'] = possible_moves(root_node['board'], root_node['sheep'], root_node['player_id'])

    for _ in range(num_iterations):
        node = root_node
        while moves := node.get('moves'):
            if not moves:
                node['moves'] = possible_moves(node['board'], node['sheep'], node['player_id'])
                moves = node['moves']
            if not node['children']:
                for move in moves:
                    new_board, new_sheep = make_move(copy.deepcopy(node['board']), copy.deepcopy(node['sheep']), move)
                    child_node = {
                        'board': new_board,
                        'sheep': new_sheep,
                        'player_id': (node['player_id'] % num_players) + 1,
                        'num_iterations': 0,
                        'parent': node,
                        'move': move,
                        'moves': None,
                        'visits': 0,
                        'cumulative_score': 0,
                        'children': dict()
                    }

                    node['children'][move] = child_node

            if all(child['num_iterations'] > 0 for child in node['children'].values()):
                move = max(node['children'].values(), key=lambda child: child['cumulative_score'] / child[
                    'num_iterations'] + exploration_param * math.sqrt(
                    math.log(node['num_iterations']) / child['num_iterations']))['move']
            else:
                move = random.choice(moves)

            child_node = node['children'][move]
            node = child_node

        score = random_simulation(node['board'], node['sheep'], 8, node['player_id'], player_id, num_players)

        while node != root_node:
            node['visits'] += 1
            node['cumulative_score'] += score
            node = node['parent']

    best_move = max(root_node['children'].values(),
                    key=lambda child: child['cumulative_score'] / child['visits'] if child['visits'] > 0 else 0)['move']
    return best_move


def InitPos(mapStat):
    init_pos = choose_starting_position(mapStat)
    return init_pos


def GetStep(playerID, mapStat, sheepStat):
    mapStat = np.array(mapStat, dtype=int)
    sheepStat = np.array(sheepStat, dtype=int)
    move = monte_carlo_tree_search_ucb(mapStat, sheepStat, playerID, num_iterations=2500)
    start_pos, _, direction = move
    step = [start_pos, sheepStat[start_pos[0]][start_pos[1]] // 2, direction]
    return step


(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)
    STcpClient.SendStep(id_package, Step)
