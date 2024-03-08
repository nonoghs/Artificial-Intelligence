import heapq


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    Calculate the sum of Manhattan distances from the current state to the goal state.

    INPUT:
        from_state: The current state of the puzzle (list of length 9).
        to_state: The goal state of the puzzle (defaults to [1, 2, 3, 4, 5, 6, 7, 0, 0]).

    RETURNS:
        The sum of Manhattan distances for all tiles.
    """
    distance = 0

    # print(f"Calculating Manhattan Distance for State: {from_state}")

    for index, value in enumerate(from_state):
        if value != 0:
            goal_index = to_state.index(value)
            current_row, current_col = index // 3, index % 3
            goal_row, goal_col = goal_index // 3, goal_index % 3
            tile_distance = abs(current_row - goal_row) + abs(current_col - goal_col)
            distance += tile_distance
            # print(
            # f"Tile: {value}, Current: ({current_row}, {current_col}), Goal: ({goal_row}, {goal_col}), Tile Distance: {tile_distance}")

    # print(f"Total Manhattan Distance: {distance}")
    return distance


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT:
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle.
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


# Some code is from https://www.geeksforgeeks.org/a-search-algorithm/
def get_succ(state):

    succ_states = []
    index_of_zero = [i for i, x in enumerate(state) if x == 0]  # Find the indices of empty spaces

    # Possible moves for each position in the grid
    moves = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1, 5],
        3: [0, 4, 6],
        4: [1, 3, 5, 7],
        5: [2, 4, 8],
        6: [3, 7],
        7: [4, 6, 8],
        8: [5, 7]
    }

    # print(f"Original State: {state}")  # Print the original state

    for zero in index_of_zero:
        for move in moves[zero]:
            if state[move] != 0:  # Ensure we're moving a tile, not an empty space
                new_state = state.copy()
                new_state[zero], new_state[move] = new_state[move], new_state[zero]
                if new_state not in succ_states:
                    succ_states.append(new_state)

                    # print(f"Move: {zero}->{move}, New State: {new_state}")  # Debug print

    return sorted(succ_states)


# Some code is from https://www.geeksforgeeks.org/a-search-algorithm/
def solve(initial_state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    def reconstruct_path(came_from, current):
        path = [current]  # Start with the goal state
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get the path from initial to goal

    open_set = []
    heapq.heappush(open_set, (get_manhattan_distance(initial_state), tuple(initial_state)))
    came_from = {}
    g_score = {tuple(initial_state): 0}
    max_queue_length = 1  # Initialize the max queue length

    while open_set:
        _, current = heapq.heappop(open_set)
        current = tuple(current)

        # Update the maximum queue length
        max_queue_length = max(max_queue_length, len(open_set) + 1)

        if current == tuple(goal_state):
            path = reconstruct_path(came_from, current)
            for state in path:
                state_list = list(state)
                h = get_manhattan_distance(state_list)
                moves = g_score[state]
                print(f"{state_list} h={h} moves: {moves}")
            print("Max queue length: {}".format(max_queue_length))
            return

        for next_state in get_succ(list(current)):
            next_state_tuple = tuple(next_state)
            new_cost = g_score[current] + 1
            if next_state_tuple not in g_score or new_cost < g_score[next_state_tuple]:
                came_from[next_state_tuple] = current
                g_score[next_state_tuple] = new_cost
                f_score = new_cost + get_manhattan_distance(next_state)
                heapq.heappush(open_set, (f_score, next_state_tuple))

    print("No solution found")
    print("Max queue length: {}".format(max_queue_length))

if __name__ == "__main__":

    test_state = [2, 5, 1, 4, 0, 6, 7, 0, 3]
    goal_state = [1, 2, 3, 4, 5, 6, 7, 0, 0]  # Optional to specify this

    print("Testing print_succ function:")
    print_succ(test_state)
    print()

    print("Testing get_manhattan_distance function:")
    distance = get_manhattan_distance(test_state, goal_state)
    print(f"Manhattan distance from test state to goal state: {distance}")
    print()

    print("Testing solve function:")
    solve(test_state)
    print()




    """
    test_cases = [
        ([0, 4, 6, 3, 0, 1, 7, 2, 5], "print_succ_1"),
        ([0, 6, 0, 3, 5, 1, 7, 2, 4], "print_succ_2")
    ]

    for test_state, case_name in test_cases:
        print(f"Testing {case_name}:")
        print_succ(test_state)
        print()
    """


    """
    # Define the test state
    test_state = [0, 4, 7, 1, 3, 0, 6, 2, 5]

    # Call the solve function with the test state
    solve(test_state)
    """

    """
    test_state = [6, 0, 0, 3, 5, 1, 7, 2, 4]
    goal_state = [1, 2, 3, 4, 5, 6, 7, 0, 0]  # This is the goal state

    print("Testing solve function with test case:")
    solve(test_state)
    print()
    """
