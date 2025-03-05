# Team name: 是何人演奏我的春日影
# Team ID: 3
# Team member: 109550074 吳秉澍, 109550076 林睿騰

import STcpClient

from collections import deque
import numpy as np
import random
import copy
import math
import time


def choose_starting_position(board):
    available_positions = []
    for i in range(1, board.shape[0]-1):
        for j in range(1, board.shape[1]-1):
            if board[i][j] == 0:
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    if board[i + di][j + dj] == -1:
                        available_positions.append((i, j))
                        break

    def get_openness_block(x, y):
        openness_cnt, block_cnt = 0, 0
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            openness_cnt += (board[x+dx][y+dy] == 0)
            block_cnt += (board[x+dx][y+dy] > 0)
        return openness_cnt, block_cnt

    openness_dict, block_dict = dict(), dict()
    for i in range(1, board.shape[0]-1):
        for j in range(1, board.shape[1]-1):
            if board[i][j] == 0:
                openness, block = get_openness_block(i, j)
                openness_dict[(i, j)] = openness
                block_dict[(i, j)] = block

    best_score = 0
    best_starting_point = None
    for available_position in available_positions:
        i, j = available_position

        score = 0
        for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
            new_i, new_j = i + di, j + dj
            if board[new_i][new_j] == 0:
                score += openness_dict[(new_i, new_j)] * (1 + (di * dj == 0))

        if score > best_score:
            best_score = score
            best_starting_point = (i, j)

    return best_starting_point


def calculate_score(board, sheep, player_id):
    visited = set()
    score = 0
    for i in range(1, board.shape[0]-1):
        for j in range(1, board.shape[1]-1):
            if board[i][j] == player_id and (i, j) not in visited:
                region_size = 0
                queue = deque()
                queue.append((i, j))
                visited.add((i, j))
                while queue:
                    x, y = queue.popleft()

                    openness = 0
                    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                        openness += (board[x + dx][y + dy] == 0) * (1 + (dx * dy == 0))
                    if sheep[x][y] != 0:
                        region_size += openness / 12 * sheep[x][y]
                    else:
                        region_size += 1

                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        next_position = (x + dx, y + dy)
                        if board[x + dx][y + dy] == player_id and next_position not in visited:
                            queue.append(next_position)
                            visited.add(next_position)

                score += math.pow(region_size, 1.25)
    return score


def possible_moves(board, sheep, player_id, equal_sheep=False):
    def get_openness(x, y):
        openness = 0
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            openness += (board[x+dx][y+dy] == 0) * (1 + (dx * dy == 0))
        return openness

    moves = []
    directions = [(j, i) for i in [-1, 0, 1] for j in [-1, 0, 1]]
    for i in range(1, board.shape[0]-1):
        for j in range(1, board.shape[1]-1):
            if board[i][j] == player_id and sheep[i][j] > 1:
                for idx, direction in enumerate(directions):
                    if direction == (0, 0):
                        continue
                    new_i, new_j, di, dj, steps = i, j, direction[0], direction[1], 0
                    while board[new_i + di][new_j + dj] == 0:
                        new_i += di
                        new_j += dj
                        steps += 1
                    if steps > 0:
                        if equal_sheep:
                            moves.append(((i, j), (new_i, new_j), sheep[i][j]//2, idx + 1))
                        else:
                            start_score, end_score = get_openness(i, j), get_openness(new_i, new_j)
                            num_sheep = max(1, int(math.floor(end_score / (start_score + end_score) * sheep[i][j])))
                            moves.append(((i, j), (new_i, new_j), num_sheep, idx + 1))

    return moves


def blind_possible_moves(board, sheep, player_id):
    moves = []
    directions = [(j, i) for i in [-1, 0, 1] for j in [-1, 0, 1]]
    for i in range(1, board.shape[0]-1):
        for j in range(1, board.shape[1]-1):
            if board[i][j] == player_id:
                for idx, direction in enumerate(directions):
                    if direction == (0, 0):
                        continue
                    new_i, new_j, di, dj, steps = i, j, direction[0], direction[1], 0
                    while board[new_i + di][new_j + dj] == 0:
                        new_i += di
                        new_j += dj
                        steps += 1
                    if steps > 0:
                        moves.append(((i, j), (new_i, new_j), 1, idx + 1))

    return moves


def blind_make_move(board, sheep, move):
    start_pos, end_pos, _, _ = move
    board[end_pos[0]][end_pos[1]] = board[start_pos[0]][start_pos[1]]
    sheep[end_pos[0]][end_pos[1]] = 1
    return board, sheep


def make_move(board, sheep, move):
    start_pos, end_pos, num_sheep, _ = move
    board[end_pos[0]][end_pos[1]] = board[start_pos[0]][start_pos[1]]
    sheep[start_pos[0]][start_pos[1]] -= num_sheep
    sheep[end_pos[0]][end_pos[1]] += num_sheep
    return board, sheep


def monte_carlo_tree_search_ucb(board, sheep, player_id, iteration_time_limit, exploration_param=math.sqrt(2), num_players=4):
    start_time = time.time()

    root_node = {
        'board': copy.deepcopy(board),
        'sheep': copy.deepcopy(sheep),
        'player_id': player_id,
        'parent': None,
        'moves': None,
        'visits': 0,
        'children': dict()
    }

    root_node['moves'] = possible_moves(root_node['board'], root_node['sheep'], root_node['player_id'])

    while True:
        node = root_node
        root_node['visits'] += 1
        while moves := node.get('moves'):
            if not moves:
                if node['player_id'] == player_id:
                    node['moves'] = possible_moves(node['board'], node['sheep'], node['player_id'])
                else:
                    node['moves'] = blind_possible_moves(node['board'], node['sheep'], node['player_id'])
                moves = node['moves']
            if not node['children']:
                for move in moves:
                    if node['player_id'] == player_id:
                        new_board, new_sheep = make_move(copy.deepcopy(node['board']), copy.deepcopy(node['sheep']), move)
                    else:
                        new_board, new_sheep = blind_make_move(copy.deepcopy(node['board']), copy.deepcopy(node['sheep']), move)
                    child_node = {
                        'board': new_board,
                        'sheep': new_sheep,
                        'player_id': (node['player_id'] % num_players) + 1,
                        'parent': node,
                        'move': move,
                        'moves': None,
                        'visits': 0,
                        'cumulative_score': 0,
                        'children': dict()
                    }
                    node['children'][move] = child_node

            if all(child['visits'] > 0 for child in node['children'].values()):
                move = max(node['children'].values(), key=lambda child: child['cumulative_score'] / child[
                    'visits'] + exploration_param * math.sqrt(
                    math.log(node['visits']) / child['visits']))['move']
            else:
                move = random.choice(moves)

            child_node = node['children'][move]
            node = child_node

        score = calculate_score(board, sheep, node['player_id']) * (3 if node['player_id'] == player_id else -1)

        while node != root_node:
            node['visits'] += 1
            node['cumulative_score'] += score
            node = node['parent']

        end_time = time.time()
        if end_time - start_time > iteration_time_limit:
            break

    best_move = max(root_node['children'].values(),
                    key=lambda child: child['cumulative_score'] / child['visits'] if child['visits'] > 0 else 0)['move']
    return best_move


def add_edge(matrix, value):
    new_matrix = np.zeros((matrix.shape[0]+2, matrix.shape[1]+2), dtype=np.int8) + value
    for i in range(1, new_matrix.shape[0]-1):
        for j in range(1, new_matrix.shape[1]-1):
            new_matrix[i][j] = matrix[i-1][j-1]
    return new_matrix


def InitPos(mapStat):
    mapStat = add_edge(np.array(mapStat, dtype=np.int8), -1)
    position = choose_starting_position(mapStat)
    original_pos = (position[0]-1, position[1]-1)
    return original_pos


def GetStep(playerID, mapStat, sheepStat):
    mapStat = add_edge(np.array(mapStat, dtype=np.int8), -1)
    sheepStat = add_edge(np.array(sheepStat, dtype=np.int8), 0)
    move = monte_carlo_tree_search_ucb(mapStat, sheepStat, playerID, iteration_time_limit=2.75)
    start_pos, _, num_sheep, direction = move
    position = (start_pos[0]-1, start_pos[1]-1)
    step = [position, num_sheep, direction]
    return step


# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    sheepStat = np.where(mapStat == playerID, sheepStat, 0) # hide other player's sheep number
    # hide other player's sheep number
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
# DON'T MODIFY ANYTHING IN THIS WHILE LOOP OR YOU WILL GET 0 POINT IN THIS QUESTION