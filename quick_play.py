import os
import sys
import time
import pygame
import random
import argparse
import numpy as np
import pandas as pd
from enum import Enum
from customised_sparse_matrix import MSA


''' 
The sparse Q table is used in this script.

  Grid world

  --------------------------
0 |    |    | P2 |    |    | <-- TE RL Agent - Random objective
1 |    |    |    |    |    |
2 |    |    |    |    |    |
3 |    |    |    |    |    |
4 |    |    |    |    |    |
5 |    |    |    |    |    | <-- Ending row
6 |    |    |    |    |    |
7 |    |    |    |    |    |
8 |    |    |    |    |    |
9 |    |    |    |    |    |
10|    |    | P1 |    |    | <-- Human - Random objective
  --------------------------
    0    1    2    3    4

Horizontal -- x
Vertical -- y

Corridor shape -- (11, 5)

'''


class Action(Enum):
    R = 0
    S = 1
    L = 2


class Objective(Enum):
    Pass = 0
    Meet = 1


class Player:
    def __init__(self, id, h_length):
        self.id = id
        
        # Set initial location
        self.x = np.random.choice([0,1,2,3,4])
        # self.x = 1

        if self.id == 1:
            self.y = 10
            self.dir = -1
        else:
            self.y = 0
            self.dir = 1

        # Set history recording
        self.h_length = h_length
        self.history = [self.x, self.y] * self.h_length

    def set_objective(self, obj):
        self.objective = obj
    
    def get_loc(self):
        return np.array([self.x, self.y])
    
    def move(self, action):
        # Record history movement
        self.history = self.history + [self.x, self.y]
        self.history = self.history[2:]
        # Move according to the action
        self.y += self.dir
        if action == Action.L:
            if 0 <= self.x - self.dir <= 4:
                self.x -= self.dir
        elif action == Action.R:
            if 0 <= self.x + self.dir <= 4:
                self.x += self.dir
        else:
            pass
    
    def get_state(self):
        return self.history + [self.x, self.y]

    def reset_state(self):
        self.x = np.random.choice([0,1,2,3,4])
        # self.x = 1
        if self.id == 1:
            self.y = 10
            self.dir = -1
        else:
            self.y = 0
            self.dir = 1

        self.history = [self.x, self.y] * self.h_length


class Q_RL:
    def __init__(self, save_dir):
        self.corridor = np.zeros((11,5))
        self.log_p1 = []
        self.log_p2 = []
        self.save_dir = save_dir
        self.test = 0
        self.test_log_p1_pass = []
        self.test_log_p2_pass = []
        self.test_log_p1_meet = []
        self.test_log_p2_meet = []
        self.test_log_p1 = []
        self.test_log_p2 = []

        # Initialize Pygame
        pygame.init()

        # Define constants
        self.GRID_WIDTH = 5
        self.GRID_HEIGHT = 11
        self.EXTRA_WIDTH = 8
        self.CELL_SIZE = 50
        self.FPS = 30

        # Define colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (0, 128, 255)
        self.RED = (255, 0, 0)
        self.ORANGE = (255, 165, 0)
        self.GREEN = (0, 200, 0)
        self.GOLD = (255, 215, 0)
        self.CYAN = (0, 255, 255)
        self.DARK_GREY = (80, 80, 80)
        self.LIGHT_GREY = (200, 200, 200)
        self.robot_colour_list = [
            (204, 153, 255),  # Lighter Purple
            (255, 204, 255),  # Lighter Magenta
            (205, 133, 63),   # Lighter Brown
            (255, 255, 153),  # Lighter Yellow
            (153, 255, 255),  # Lighter Aqua
            (255, 186, 102)   # Lighter Dark Orange
        ]


        self.robot_colour_counter = 0

        # Initialize the game window
        self.screen_width, self.screen_height = (self.GRID_WIDTH + self.EXTRA_WIDTH) * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Corridor Dilemma")

        # Set up the game clock and font
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 38)
        self.small_font = pygame.font.Font(None, 30)
        self.large_font = pygame.font.Font(None, 70)
        self.running = True

        # Set element locations ===================
        # The agent block
        self.the_agent_loc =            (self.GRID_WIDTH * self.CELL_SIZE + 10, 15)
        self.face_position =            (self.GRID_WIDTH * self.CELL_SIZE + 10, self.CELL_SIZE)
        self.agent_message_loc =        (self.GRID_WIDTH * self.CELL_SIZE + 10, self.CELL_SIZE * 3 + 10)

        # Objective block
        self.objective_loc =            (self.GRID_WIDTH * self.CELL_SIZE + 10, (self.GRID_HEIGHT - 6) * self.CELL_SIZE)
        self.objective_indicator_loc =  (self.GRID_WIDTH * self.CELL_SIZE + 10, (self.GRID_HEIGHT - 5) * self.CELL_SIZE)

        # Info block
        self.outcome_loc =              (self.GRID_WIDTH * self.CELL_SIZE + 10, (self.GRID_HEIGHT - 4) * self.CELL_SIZE + 10)
        self.round_loc =                (self.GRID_WIDTH * self.CELL_SIZE + 10, (self.GRID_HEIGHT - 3) * self.CELL_SIZE + 10)
        self.agent_score_loc =          (self.GRID_WIDTH * self.CELL_SIZE + 10, (self.GRID_HEIGHT - 2) * self.CELL_SIZE + 10)
        self.player_score_loc =         (self.GRID_WIDTH * self.CELL_SIZE + 10, (self.GRID_HEIGHT - 1) * self.CELL_SIZE + 10)
        self.agent_score_plus_loc =     ((self.GRID_WIDTH + 5) * self.CELL_SIZE + 10, (self.GRID_HEIGHT - 2) * self.CELL_SIZE + 10)
        self.player_score_plus_loc =    ((self.GRID_WIDTH + 5) * self.CELL_SIZE + 10, (self.GRID_HEIGHT - 1) * self.CELL_SIZE + 10)
        # =========================================
        
        # Load and set images
        self.happy_face = pygame.transform.scale(pygame.image.load("./images/happy.jpg"), (100, 100))
        self.happy_face_rect = self.happy_face.get_rect()
        self.happy_face_rect.topleft = self.face_position

        self.sad_face = pygame.transform.scale(pygame.image.load("./images/sad.jpg"), (100, 100))
        self.sad_face_rect = self.sad_face.get_rect()
        self.sad_face_rect.topleft = self.face_position

        self.neutral_face = pygame.transform.scale(pygame.image.load("./images/neutral.jpg"), (100, 100))
        self.neutral_face_rect = self.neutral_face.get_rect()
        self.neutral_face_rect.topleft = self.face_position


    def setup_env(self, history_len):
        self.history_len = history_len
        self.p1 = Player(1, h_length=self.history_len)
        self.p2 = Player(2, h_length=self.history_len)
        self.episode_ends = 0
        self.iteration = 1

        # States for P1
        x1_t = np.arange(5)
        y1_t = np.arange(5, 11)

        # States for P2
        x2_t = np.arange(5)
        y2_t = np.arange(0, 6)

        # Actions for P1 and P2
        a1_t = np.arange(3)
        a2_t = np.arange(3)

        # States & Actions
        self.state_action_space_p1 = [x1_t, y1_t] * (self.history_len+1) + [x2_t, y2_t] * (self.history_len+1) + [a1_t]
        self.state_action_space_p2 = [x1_t, y1_t] * (self.history_len+1) + [x2_t, y2_t] * (self.history_len+1) + [a2_t]
        # Q_dim_p1 = [len(x1_t), len(y1_t)] * (self.history_len+1) + [len(x2_t), len(y2_t)] * (self.history_len+1) + [len(a1_t)]
        self.Q_dim_p2 = tuple([len(x1_t), len(y1_t)] * (self.history_len+1) + [len(x2_t), len(y2_t)] * (self.history_len+1) + [len(a2_t)])
        # self.Q_table_p1_pass = np.zeros(tuple(Q_dim_p1))
        # self.Q_table_p1_meet = np.zeros(tuple(Q_dim_p1))
        self.Q_table_p2_pass = MSA(self.Q_dim_p2)
        self.Q_table_p2_meet = MSA(self.Q_dim_p2)

        # Random initial objective for p1 and p2
        self.p1_objective = np.random.choice(list(Objective))
        self.p2_objective = np.random.choice(list(Objective))


    def run_episode(self, show):
        while not self.episode_ends:
            self.test_step()
            self.show_corridor()


    def test_(self, dt, model_type, max_iteration):
        # Set agent colour
        self.ROBOT_COLOUR = self.robot_colour_list[self.robot_colour_counter]
        self.robot_colour_counter += 1

        # Set max. iteration
        self.max_iteration = max_iteration

        # Turn on test mode
        self.test = 1

        # Read in meta information
        meta_info = self.read_info(self.save_dir + dt + "/info.txt")
        history_len = int(meta_info['History(t-n)'])

        # Set up enviornment and load Q tables
        self.setup_env(history_len=history_len)
        self.Q_table_p2_pass.load_sparse(np.load(self.save_dir + dt + "/Q_table_p2_pass.npy"))
        self.Q_table_p2_meet.load_sparse(np.load(self.save_dir + dt + "/Q_table_p2_meet.npy"))

        # Setup objective pool
        objective_lots = [[Objective.Pass, Objective.Pass],
                          [Objective.Meet, Objective.Meet],
                          [Objective.Pass, Objective.Meet],
                          [Objective.Meet, Objective.Pass]]
        
        # Initialise player performance trakcing list
        psr = []
        msr = []
        asr = []

        # Start testing
        while self.iteration <= max_iteration:
            # Choose p1 and p2 objective
            obj_lot = np.random.choice(np.arange(len(objective_lots)))
            obj_lot = objective_lots[obj_lot]
            objective_lots.remove(obj_lot)
            self.p1_objective = obj_lot[0]
            self.p2_objective = obj_lot[1]
            if len(objective_lots) == 0:
                objective_lots = [[Objective.Pass, Objective.Pass],
                                [Objective.Meet, Objective.Meet],
                                [Objective.Pass, Objective.Meet],
                                [Objective.Meet, Objective.Pass]]
            self.show_corridor()
            # self.wait_for_start("to Start")
            self.run_episode(show=True)
            self.iteration += 1
            self.p1.reset_state()
            self.p2.reset_state()
            self.corridor = np.zeros((11,5))
            self.episode_ends = 0
            # Performance recording
            if len(self.test_log_p1_pass) == 0: psr.append(0)
            else: psr.append(np.sum(self.test_log_p1_pass)/len(self.test_log_p1_pass))
            if len(self.test_log_p1_meet) == 0: msr.append(0)
            else: msr.append(np.sum(self.test_log_p1_meet)/len(self.test_log_p1_meet))
            asr.append(np.sum(self.test_log_p1)/len(self.test_log_p1))

        # Save performance recording
        df = pd.DataFrame({'psr': psr, 'msr': msr, 'asr': asr})
        # Save DataFrame to CSV file
        df.to_csv(f'log/{model_type}.csv', index=False)
        print("Done saving")

        # Reseting
        self.log_p1 = []
        self.log_p2 = []
        self.test_log_p1_pass = []
        self.test_log_p2_pass = []
        self.test_log_p1_meet = []
        self.test_log_p2_meet = []
        self.test_log_p1 = []
        self.test_log_p2 = []
            
        # quit()


    def test_step(self):
        # Epsilon decay
        epsilon = 0
        # Select and take action for P2
        p1_action = self.wait_key_input()
        p2_action, _, _ = self.select_action(2, self.get_state(), epsilon)
        self.p1.move(p1_action)
        self.p2.move(p2_action)
        # Determine if the episode is finished
        self.log_test_score(1)
        self.log_test_score(2)


    def wait_key_input(self):
        # Handle events
        start_countdown = time.time()
        key_pressed = False
        time_limit = 15 # Seconds
        wait_time = 2 # Seconds
        # Create a mask for the countdown text
        old_element_rect = pygame.Rect(
                self.GRID_WIDTH * self.CELL_SIZE + 10,
                (self.GRID_HEIGHT - 4) * self.CELL_SIZE + 10,
                self.CELL_SIZE * 7,  # Replace with the actual width of your text
                self.CELL_SIZE  # Replace with the actual height of your text
            )
        while not key_pressed:
            # Clear only the area of the old element
            self.screen.fill(self.BLACK, old_element_rect)
            # Render countdown text
            line_text = self.font.render("Countdown: {:d}".format(int(time_limit - (time.time() - start_countdown) + 1)), True, self.WHITE)
            self.screen.blit(line_text, self.outcome_loc)
            # Update the display
            pygame.display.update(old_element_rect)

            if time.time() - start_countdown < time_limit:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            # Handle move forward action
                            self.screen.fill(self.BLACK, old_element_rect)
                            line_text = self.font.render("Waiting for opponent...", True, self.DARK_GREY)
                            self.screen.blit(line_text, self.outcome_loc)
                            pygame.display.update(old_element_rect)
                            # time.sleep(random.choice([0, random.triangular(0, 0.3, wait_time)]))  ## Enable the waiting time later
                            return Action.S
                        elif event.key == pygame.K_LEFT:
                            # Handle turn left action
                            self.screen.fill(self.BLACK, old_element_rect)
                            line_text = self.font.render("Waiting for opponent...", True, self.DARK_GREY)
                            self.screen.blit(line_text, self.outcome_loc)
                            pygame.display.update(old_element_rect)
                            # time.sleep(random.choice([0, random.triangular(0, 0.3, wait_time)]))  ## Enable the waiting time later
                            return Action.R
                        elif event.key == pygame.K_RIGHT:
                            # Handle turn right action
                            self.screen.fill(self.BLACK, old_element_rect)
                            line_text = self.font.render("Waiting for opponent...", True, self.DARK_GREY)
                            self.screen.blit(line_text, self.outcome_loc)
                            pygame.display.update(old_element_rect)
                            # time.sleep(random.choice([0, random.triangular(0, 0.3, wait_time)]))  ## Enable the waiting time later
                            return Action.L
            else:
                # If the time is up, just move straight
                return Action.S


    def wait_for_start(self, message):
        key_pressed = False
        start_counting = time.time()
        # Create a mask for the countdown text
        old_element_rect = pygame.Rect(
                self.GRID_WIDTH * self.CELL_SIZE + 10,
                (self.GRID_HEIGHT - 4) * self.CELL_SIZE + 10,
                self.CELL_SIZE * 7,  # Replace with the actual width of your text
                self.CELL_SIZE - 20  # Replace with the actual height of your text
            )
        while not key_pressed:
            if ((time.time() - start_counting)//0.5)%2 == 0:
                # Clear only the area of the old element
                self.screen.fill(self.BLACK, old_element_rect)
                # Render countdown text
                line_text = self.font.render("Press Enter {}".format(message), True, self.WHITE)
            else:
                self.screen.fill(self.LIGHT_GREY, old_element_rect)
                # Render countdown text
                line_text = self.font.render("Press Enter {}".format(message), True, self.BLACK)
            self.screen.blit(line_text, self.outcome_loc)
            # Update the display
            pygame.display.update(old_element_rect)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.screen.fill(self.BLACK, old_element_rect)
                        line_text = self.font.render("Waiting for opponent...", True, self.DARK_GREY)
                        self.screen.blit(line_text, self.outcome_loc)
                        pygame.display.update(old_element_rect)
                        time.sleep(random.choice([0, random.triangular(0, 0.2, 1)]))
                        return


    def find_sa_idx(self, pid, state_action):
        sa_idx = []
        for i, sa in enumerate(state_action):
            if pid == 1:
                sa_idx.append(np.where(self.state_action_space_p1[i]==sa)[0][0])
            else:
                sa_idx.append(np.where(self.state_action_space_p2[i]==sa)[0][0])
        return tuple(sa_idx)


    def get_state(self):
        return self.p1.get_state() + self.p2.get_state()


    def select_action(self, pid, state, epsilon):
        prob = np.random.uniform()
        if prob <= epsilon:
            # Select random action - Explore
            random_action_idx = np.random.choice([0,1,2])
            state_action = state + [random_action_idx]
            if pid == 1:
                if self.p1_objective == Objective.Pass:
                    Q_value = self.Q_table_p1_pass.get(self.find_sa_idx(pid, state_action))
                else:
                    Q_value = self.Q_table_p1_meet.get(self.find_sa_idx(pid, state_action))
            else:
                if self.p2_objective == Objective.Pass:
                    Q_value = self.Q_table_p2_pass.get(self.find_sa_idx(pid, state_action))
                else:
                    Q_value = self.Q_table_p2_meet.get(self.find_sa_idx(pid, state_action))
            return Action(random_action_idx), Q_value, state_action
        else:
            # Select the best action - Exploit
            Q_value_list = []
            for a in range(3):
                state_action = state + [a]
                if pid == 1:
                    if self.p1_objective == Objective.Pass:
                        Q_value = self.Q_table_p1_pass.get(self.find_sa_idx(pid, state_action))
                    else:
                        Q_value = self.Q_table_p1_meet.get(self.find_sa_idx(pid, state_action))
                else:
                    if self.p2_objective == Objective.Pass:
                        Q_value = self.Q_table_p2_pass.get(self.find_sa_idx(pid, state_action))
                    else:
                        Q_value = self.Q_table_p2_meet.get(self.find_sa_idx(pid, state_action))
                Q_value_list.append(Q_value)
            # Find the best action index
            # best_action_idx = np.argmax(Q_value_list)
            best_action_idx = np.random.choice([0,1,2], p=softmax(np.array(Q_value_list)))
            # Find the best Q value
            if pid == 1:
                if self.p1_objective == Objective.Pass:
                    best_Q_value = self.Q_table_p1_pass.get(self.find_sa_idx(pid, state_action))
                else:
                    best_Q_value = self.Q_table_p1_meet.get(self.find_sa_idx(pid, state_action))
            else:
                if self.p2_objective == Objective.Pass:
                    best_Q_value = self.Q_table_p2_pass.get(self.find_sa_idx(pid, state_action))
                else:
                    best_Q_value = self.Q_table_p2_meet.get(self.find_sa_idx(pid, state_action))
            return Action(best_action_idx), best_Q_value, state_action
    

    def log_test_score(self, pid):
        if pid == 1:
            objective = self.p1_objective
            test_log_pass = self.test_log_p1_pass
            test_log_meet = self.test_log_p1_meet
            test_log = self.test_log_p1
        else:
            objective = self.p2_objective
            test_log_pass = self.test_log_p2_pass
            test_log_meet = self.test_log_p2_meet
            test_log = self.test_log_p2

        if self.p1.get_loc()[1] == self.corridor.shape[0]//2 and self.p2.get_loc()[1] == self.corridor.shape[0]//2:
            self.episode_ends = 1
            if objective == Objective.Pass:
                if self.p1.get_loc()[0] == self.p2.get_loc()[0]:
                    # print('Meet')
                    test_log_pass.append(0)
                    test_log.append(0)
                else:
                    # print('Pass')
                    test_log_pass.append(1)
                    test_log.append(1)
            else:
                if self.p1.get_loc()[0] == self.p2.get_loc()[0]:
                    # print('Meet')
                    test_log_meet.append(1)
                    test_log.append(1)
                else:
                    # print('Pass')
                    test_log_meet.append(0)
                    test_log.append(0)

            if pid == 1:
                self.test_log_p1_pass = test_log_pass
                self.test_log_p1_meet = test_log_meet
                self.test_log_p1 = test_log
            else:
                self.test_log_p2_pass = test_log_pass
                self.test_log_p2_meet = test_log_meet
                self.test_log_p2 = test_log

    
    def show_corridor(self):
        # Render the grid and player
        self.screen.fill(self.BLACK)
        # Render the row
        for col in range(self.GRID_WIDTH):
            pygame.draw.rect(self.screen, self.DARK_GREY, (col * self.CELL_SIZE, 5 * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
        
        # Render the grid
        for row in range(self.GRID_HEIGHT):
            for col in range(self.GRID_WIDTH):
                pygame.draw.rect(self.screen, self.WHITE, (col * self.CELL_SIZE, row * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE), 1)

        # Draw players
        pygame.draw.rect(self.screen, self.BLUE, (self.p1.get_loc()[0] * self.CELL_SIZE, self.p1.get_loc()[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
        pygame.draw.rect(self.screen, self.ROBOT_COLOUR, (self.p2.get_loc()[0] * self.CELL_SIZE, self.p2.get_loc()[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Render objectives
        line_text = self.large_font.render("Your Objective:", True, self.WHITE)
        self.screen.blit(line_text, self.objective_loc)

        if self.p1_objective == Objective.Pass:
            line_text = self.large_font.render("Pass", True, self.GOLD)
            self.screen.blit(line_text, self.objective_indicator_loc)
        else:
            line_text = self.large_font.render("Meet", True, self.CYAN)
            self.screen.blit(line_text, self.objective_indicator_loc)

        # # Render instructions
        # instructions_lines = [
        #     "Instructions:",
        #     "Press UP to move forward,",
        #     "LEFT to move forward left,",
        #     "RIGHT to move forward right."]
        # for i, line in enumerate(instructions_lines):
        #     line_text = self.font.render(line, True, (150, 150, 150))
        #     self.screen.blit(line_text, (self.GRID_WIDTH * self.CELL_SIZE + 10, (self.GRID_HEIGHT - 8) * self.CELL_SIZE + 10 + i * 30))

        # Render round
        line_text = self.font.render("Round: {}/{}".format(self.iteration, self.max_iteration), True, self.WHITE)
        self.screen.blit(line_text, self.round_loc)

        # Render scores
        line_text = self.font.render("          Your Score: {}".format(sum(self.test_log_p1)), True, self.BLUE)
        self.screen.blit(line_text, self.player_score_loc)
        line_text = self.font.render("Opponent Score: {}".format(sum(self.test_log_p2)), True, self.ROBOT_COLOUR)
        self.screen.blit(line_text, self.agent_score_loc)

        # Show robot face
        line_text = self.font.render("Your opponent:", True, self.ROBOT_COLOUR)
        self.screen.blit(line_text, self.the_agent_loc)
        self.screen.blit(self.neutral_face, self.neutral_face_rect)

        # Render result
        if self.p2.get_loc()[1] == self.p1.get_loc()[1]:
            # If meet
            if self.p1.get_loc()[0] == self.p2.get_loc()[0]:
                # For player
                if self.p1_objective == Objective.Meet:
                    line_text = self.font.render("You Success!", True, self.GREEN)
                    self.screen.blit(line_text, self.outcome_loc)
                    line_text = self.font.render("+1", True, self.BLUE)
                    self.screen.blit(line_text, self.player_score_plus_loc)
                else:
                    line_text = self.font.render("You Fail...", True, self.RED)
                    self.screen.blit(line_text, self.outcome_loc)
                    line_text = self.font.render("+0", True, self.BLUE)
                    self.screen.blit(line_text, self.player_score_plus_loc)
                # For agent
                if self.p2_objective == Objective.Meet:
                    # Agent reacts
                    self.screen.blit(self.happy_face, self.happy_face_rect)
                    self.display_typewriter_text("Opponent successfully met you!", self.agent_message_loc, self.GREEN)
                    # Update score
                    line_text = self.font.render("+1", True, self.ROBOT_COLOUR)
                    self.screen.blit(line_text, self.agent_score_plus_loc)
                else:
                    # Agent reacts
                    self.screen.blit(self.sad_face, self.sad_face_rect)
                    self.display_typewriter_text("Opponent failed to bypass you.", self.agent_message_loc, self.RED)
                    # Update score
                    line_text = self.font.render("+0", True, self.ROBOT_COLOUR)
                    self.screen.blit(line_text, self.agent_score_plus_loc)
            # If pass
            else:
                # For player
                if self.p1_objective == Objective.Pass:
                    line_text = self.font.render("You Success!", True, self.GREEN)
                    self.screen.blit(line_text, self.outcome_loc)
                    line_text = self.font.render("+1", True, self.BLUE)
                    self.screen.blit(line_text, self.player_score_plus_loc)
                else:
                    line_text = self.font.render("You Fail...", True, self.RED)
                    self.screen.blit(line_text, self.outcome_loc)
                    line_text = self.font.render("+0", True, self.BLUE)
                    self.screen.blit(line_text, self.player_score_plus_loc)
                # For agent
                if self.p2_objective == Objective.Pass:
                    # Agent reacts
                    self.screen.blit(self.happy_face, self.happy_face_rect)
                    self.display_typewriter_text("Opponent successfully bypassed you!", self.agent_message_loc, self.GREEN)
                    # update score
                    line_text = self.font.render("+1", True, self.ROBOT_COLOUR)
                    self.screen.blit(line_text, self.agent_score_plus_loc)
                else:
                    # Agent reacts
                    self.screen.blit(self.sad_face, self.sad_face_rect)
                    self.display_typewriter_text("Opponent failed to meet you.", self.agent_message_loc, self.RED)
                    # Update score
                    line_text = self.font.render("+0", True, self.ROBOT_COLOUR)
                    self.screen.blit(line_text, self.agent_score_plus_loc)
                

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        self.clock.tick(self.FPS)
    

    def show_survey(self):
        question_bank = [["It was clear to me whether this opponent", "was trying to meet or pass me.", "(Please click one of the following.)"],
                         ["It was easy to anticipate where this opponent", "would move next.", "(Please click one of the following.)"],
                         ["I understood the strategies this opponent", "was using during the game.", "(Please click one of the following.)"],
                         ["I enjoyed playing with this opponent.", "(Please click one of the following.)"],
                         ["I think this opponent was a human.", "(Please click one of the following.)"]]
        
        options_bank = [
            ["Yes, very clear", "Yes, clear", "Neutral", "No, not clear", "No, not clear at all"],
            ["Yes, very easy", "Yes, easy", "Neutral", "No, difficult", "No, very difficult"],
            ["Yes, completely", "Yes, to some extent", "Neutral", "No, not really", "No, not at all"],
            ["Yes, very much", "Yes, sort of", "Neutral", "No, not really", "No, not at all"],
            ["Yes, very likely a human", "Yes, might be a human", "Neutral", "No, unlikely to be a human", "No, very unlikely to be a human"]
        ]


        answer_bank = []
        answer_idx = []

        for qid, question_texts in enumerate(question_bank):

            # Response options
            options = options_bank[qid]

            # Display loop
            answered = False
            while not answered:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # Left mouse button
                            mouse_x, mouse_y = event.pos
                            for i, option in enumerate(options):
                                option_rect = self.font.render(option, True, self.WHITE).get_rect(center=(self.screen_width // 2, (self.screen_height // 2) + i * 50))
                                if option_rect.collidepoint(mouse_x, mouse_y):
                                    print(f"User clicked: {option}")
                                    answer_bank.append(f"{option}")
                                    answer_idx.append(i)
                                    answered = True


                # Clear the screen
                self.screen.fill(self.BLACK)

                # Render the question ID 
                question_surface = self.font.render("Survey question: {}".format(qid+1), True, self.LIGHT_GREY)
                question_rect = question_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 8))
                self.screen.blit(question_surface, question_rect)

                # Render the question
                for j, line in enumerate(question_texts):
                    question_surface = self.font.render(line, True, self.LIGHT_GREY)
                    question_rect = question_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 4 + j * 30))
                    self.screen.blit(question_surface, question_rect)

                # Render response options with highlighting on mouse over
                option_y = self.screen_height // 2
                for j, option in enumerate(options):
                    option_surface = self.font.render(option, True, self.LIGHT_GREY)
                    option_rect = option_surface.get_rect(center=(self.screen_width // 2, option_y))

                    # Highlight the option if the mouse is over it
                    if option_rect.collidepoint(pygame.mouse.get_pos()):
                        pygame.draw.rect(self.screen, self.DARK_GREY, option_rect)

                    self.screen.blit(option_surface, option_rect)
                    option_y += 50  # Adjust spacing

                # Update the display
                pygame.display.flip()
        
        return answer_bank, answer_idx


    def show_result(self, close_after):
        # Compute result
        player_result = np.sum(self.test_log_p1)/len(self.test_log_p1)
        agent_result = np.sum(self.test_log_p2)/len(self.test_log_p2)

        # Clear the screen
        self.screen.fill(self.BLACK)

        # Render the results
        result_surface = self.font.render('You final score: {}'.format(np.sum(self.test_log_p1)), True, self.WHITE)
        result_rect = result_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 4))
        self.screen.blit(result_surface, result_rect)
        result_surface = self.font.render('Opponent final score: {}'.format(np.sum(self.test_log_p2)), True, self.WHITE)
        result_rect = result_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 4 + 30))
        self.screen.blit(result_surface, result_rect)

        # # Render the agent and message
        # if player_result == agent_result:
        #     self.neutral_face_rect.center = (self.screen_width // 2, self.screen_height // 2)
        #     self.screen.blit(self.neutral_face, self.neutral_face_rect)
        #     result_surface = self.font.render("It appears that you have experienced a tie.", True, self.WHITE)
        #     result_rect = result_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 80))
        #     self.screen.blit(result_surface, result_rect)
        # elif player_result > agent_result:
        #     self.sad_face_rect.center = (self.screen_width // 2, self.screen_height // 2)
        #     self.screen.blit(self.sad_face, self.sad_face_rect)
        #     result_surface = self.font.render("Your opponent has been defeated.", True, self.WHITE)
        #     result_rect = result_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 80))
        #     self.screen.blit(result_surface, result_rect)
        # else:
        #     self.happy_face_rect.center = (self.screen_width // 2, self.screen_height // 2)
        #     self.screen.blit(self.happy_face, self.happy_face_rect)
        #     result_surface = self.font.render("Your opponent has achieved success.", True, self.WHITE)
        #     result_rect = result_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 80))
        #     self.screen.blit(result_surface, result_rect)

        # # Reset face locations
        # self.happy_face_rect.topleft = self.face_position
        # self.sad_face_rect.topleft = self.face_position
        # self.neutral_face_rect.topleft = self.face_position

        if not close_after:
            # Next agent message
            result_surface = self.font.render("Next, you will engage with a different opponent.", True, self.WHITE)
            result_rect = result_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 60))
            self.screen.blit(result_surface, result_rect)

        # Update the display
        pygame.display.flip()

        # Wait for key to continue =====================
        key_pressed = False
        start_counting = time.time()
        # Create a mask for the countdown text
        old_element_rect = pygame.Rect(
                self.GRID_WIDTH * self.CELL_SIZE + 10,
                (self.GRID_HEIGHT - 4) * self.CELL_SIZE + 10,
                self.CELL_SIZE * 6 + 5,  # Replace with the actual width of your text
                self.CELL_SIZE - 20  # Replace with the actual height of your text
            )
        old_element_rect.center = (self.screen_width // 2, (self.GRID_HEIGHT - 4) * self.CELL_SIZE + 100)
        while not key_pressed:
            if ((time.time() - start_counting)//0.5)%2 == 0:
                # Clear only the area of the old element
                self.screen.fill(self.BLACK, old_element_rect)
                # Render countdown text
                line_text = self.font.render("Press Enter {}".format("to Continue"), True, self.WHITE)
            else:
                self.screen.fill(self.LIGHT_GREY, old_element_rect)
                # Render countdown text
                line_text = self.font.render("Press Enter {}".format("to Continue"), True, self.BLACK)
            line_text_rect = line_text.get_rect(center=(self.screen_width // 2, (self.GRID_HEIGHT - 4) * self.CELL_SIZE + 100))
            self.screen.blit(line_text, line_text_rect)
            # Update the display
            pygame.display.update(old_element_rect)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        # Clear the screen
                        self.screen.fill(self.BLACK)
                        # Update the display
                        pygame.display.flip()
                        return


    def display_typewriter_text(self, text, location, colour, delay=0.01):
        x, y = location
        background_color = self.BLACK

        for char in text:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Render the text up to the current character
            current_text = self.small_font.render(text[:text.index(char) + 2], True, colour)

            # Get the rect object for the text
            text_rect = current_text.get_rect(topleft=(x, y))

            # Clear only the region where the text is displayed
            self.screen.fill(background_color, text_rect)

            # Blit the current text onto the cleared region
            self.screen.blit(current_text, (x, y))

            # Update the display
            pygame.display.flip()

            # Introduce a delay to control the typing speed
            time.sleep(delay)


    def show_thank_you_page(self):
        # Clear the screen
        self.screen.fill(self.BLACK)

        # Render the results
        result_surface = self.font.render('This is the end of this experiment.', True, self.WHITE)
        result_rect = result_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 4))
        self.screen.blit(result_surface, result_rect)
        result_surface = self.font.render('Thank you very much for your participation!', True, self.WHITE)
        result_rect = result_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 4 + 30))
        self.screen.blit(result_surface, result_rect)

        self.happy_face_rect.center = (self.screen_width // 2, self.screen_height // 2)
        self.screen.blit(self.happy_face, self.happy_face_rect)
        result_surface = self.font.render("Thank you very much!", True, self.WHITE)
        result_rect = result_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 80))
        self.screen.blit(result_surface, result_rect)

        # Update the display
        pygame.display.flip()


    def read_info(self, file_path):
            info_dict = {}
            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        key, value = parts[0].strip(), parts[1].strip()
                        info_dict[key] = value
            return info_dict


def softmax(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))


if __name__ == '__main__':
    # Set saving directory
    save_dir = "./"

    # Parameters
    max_iteration = 5

    # Create simulation
    corridor_simulation = Q_RL(save_dir=save_dir)

    # Folders
    subdir = "hist_len_5/"
    non_dir = subdir + "Transfer_entropy_p1_x0_p2_x0/"
    pos_dir = subdir + "Transfer_entropy_p1_x0_p2_x10/"
    neg_dir = subdir + "Transfer_entropy_p1_x0_p2_x-10/"

    # subdir = "mix_train/"
    # non_dir = subdir + "Mixed_training_p2_x0/"
    # pos_dir = subdir + "Mixed_training_p2_x10/"
    # neg_dir = subdir + "Mixed_training_p2_x-10/"
    dir4all3 = [non_dir, pos_dir, neg_dir]

    # Shuffle the order of the list
    random.shuffle(dir4all3)

    # Start game
    for i, dir in enumerate(dir4all3):
        if dir == non_dir:
            model_type = 'non'
        elif dir == pos_dir:
            model_type = 'pos'
        else:
            model_type = 'neg'

        random_seed = random.randint(61, 63)
        dt = pos_dir + "seed-" + str(random_seed)
        corridor_simulation.test_(dt=dt, model_type=model_type, max_iteration=max_iteration)
        
    # Quit Pygame
    time.sleep(1)
    pygame.quit()
    sys.exit()