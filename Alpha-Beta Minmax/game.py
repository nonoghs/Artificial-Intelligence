import random

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def heuristic_evaluation(self, state):
        """ Evaluate the heuristic value of a game state.
        Args:
            state (list of lists): the current state of the game board
        Returns:
            int: The heuristic value of the state, higher values are better for 'self.my_piece'.
        """
        score = 0

        # Check rows and columns for potential winning conditions
        for i in range(5):
            for j in range(5):
                if state[i][j] == self.my_piece:
                    # Horizontal
                    if j < 5-3 and all(state[i][j+k] == self.my_piece for k in range(4)):
                        score += 10
                    # Vertical
                    if i < 5-3 and all(state[i+k][j] == self.my_piece for k in range(4)):
                        score += 10

                elif state[i][j] == self.opp:
                    # Horizontal
                    if j < 5-3 and all(state[i][j+k] == self.opp for k in range(4)):
                        score -= 10
                    # Vertical
                    if i < 5-3 and all(state[i+k][j] == self.opp for k in range(4)):
                        score -= 10

        # Check for 2x2 squares of 'self.my_piece'
        for i in range(5-1):
            for j in range(5-1):
                if all(state[i+k][j+l] == self.my_piece for k in range(2) for l in range(2)):
                    score += 30
                elif all(state[i+k][j+l] == self.opp for k in range(2) for l in range(2)):
                    score -= 30

        return score

    def max_value(self, state, alpha, beta, depth):
        if self.terminal_test(state) or depth == 0:
            return self.heuristic_evaluation(state)

        v = float('-inf')

        for successor in self.generate_successors(state, self.my_piece):
            v = max(v, self.min_value(successor, alpha, beta, depth - 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)

        return v

    def min_value(self, state, alpha, beta, depth):
        if self.terminal_test(state) or depth == 0:
            return self.heuristic_evaluation(state)

        v = float('inf')

        for successor in self.generate_successors(state, self.opp):
            v = min(v, self.max_value(successor, alpha, beta, depth - 1))
            if v <= alpha:
                return v
            beta = min(beta, v)

        return v

    def minimax_decision(self, state, depth):
        best_score = float('-inf')
        beta = float('inf')
        best_move = None

        for move in self.generate_successors(state, self.my_piece):
            value = self.min_value(move, best_score, beta, depth - 1)
            if value > best_score:
                best_score = value
                best_move = move

        return best_move

    def generate_successors(self, state, piece):
        """ Generate all possible successor states from the current state for the given piece.

        Args:
            state (list of lists): the current state of the game board
            piece (str): 'b' or 'r', the piece for which to generate successors

        Returns:
            list: A list of successor states
        """
        successors = []
        drop_phase = sum(s.count(piece) for s in state) < 4

        if drop_phase:
            # Drop phase: add piece to empty spots
            for i in range(5):
                for j in range(5):
                    if state[i][j] == ' ':
                        new_state = [row[:] for row in state]  # Deep copy the state
                        new_state[i][j] = piece
                        successors.append(new_state)
        else:
            # Move phase: move pieces to adjacent spots
            for i in range(5):
                for j in range(5):
                    if state[i][j] == piece:
                        # Check all adjacent spots (up, down, left, right, and diagonals)
                        for di in range(-1, 2):
                            for dj in range(-1, 2):
                                if 0 <= i + di < 5 and 0 <= j + dj < 5 and state[i + di][j + dj] == ' ':
                                    new_state = [row[:] for row in state]  # Deep copy the state
                                    new_state[i][j] = ' '  # Remove piece from current position
                                    new_state[i + di][j + dj] = piece  # Place piece in new position
                                    successors.append(new_state)

        return successors

    def terminal_test(self, state):
        """ Return True if the game is over for the given state and False otherwise """
        return self.game_value(state) != 0

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        # Determine if it's the drop phase or move phase
        drop_phase = sum(row.count(self.my_piece) for row in state) < 4

        # Use Minimax algorithm to decide the best move
        best_move_state = self.minimax_decision(state, depth=3)  # depth can be adjusted

        # Extract the actual move from the best move state
        move = []
        if drop_phase:
            # Find the newly added piece in the best_move_state
            for i in range(5):
                for j in range(5):
                    if state[i][j] != best_move_state[i][j]:
                        move.append((i, j))
        else:
            # Find the moved piece in the best_move_state
            for i in range(5):
                for j in range(5):
                    if state[i][j] == self.my_piece and best_move_state[i][j] != self.my_piece:
                        source = (i, j)
                    elif state[i][j] != self.my_piece and best_move_state[i][j] == self.my_piece:
                        dest = (i, j)
            move = [dest, source]

        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        for i in range(2):
            for j in range(2):
                if state[i][j] != ' ' and state[i][j] == state[i + 1][j + 1] == state[i + 2][j + 2] == state[i + 3][j + 3]:
                    return 1 if state[i][j] == self.my_piece else -1

            # check / diagonal wins
        for i in range(2):
            for j in range(3, 5):
                if state[i][j] != ' ' and state[i][j] == state[i + 1][j - 1] == state[i + 2][j - 2] == state[i + 3][j - 3]:
                    return 1 if state[i][j] == self.my_piece else -1

            # check box wins
        for i in range(4):
            for j in range(4):
                if state[i][j] != ' ' and state[i][j] == state[i + 1][j] == state[i][j + 1] == state[i + 1][j + 1]:
                    return 1 if state[i][j] == self.my_piece else -1



        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
