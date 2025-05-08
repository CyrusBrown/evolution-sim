import pygame
import random
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

pygame.init()


LOWEST_COLOR = 80

GLOBAL_RADITION = 0
GLOBAL_DECISIVENESS = 0.1
GLOBAL_ENERGY = 1


performance_calcs = {
    "move": 1,
    "reproduce": 10,
    "battle": 7,
    "eat": 4,
    "stay": -5
}


class CellBrain(nn.Module):
    def __init__(self):
        super(CellBrain, self).__init__()
        self.input_neuron_count = 4 * 8 + 4 * 5 + 4
        self.hidden_neuron_count = 16
        self.output_neuron_count = 9
        
        self.fc1 = nn.Linear(self.input_neuron_count, self.hidden_neuron_count)
        self.fc2 = nn.Linear(self.hidden_neuron_count, self.output_neuron_count)

        self.tuple_scale_factors = [255, 255, 255, 100]
        self.extra_scale_factors = [100, 1, 1, 1]
        
        spread = 6.0
        limit_in_hidden = np.sqrt(spread / (self.input_neuron_count + self.hidden_neuron_count))
        limit_hidden_out = np.sqrt(spread / (self.hidden_neuron_count + self.output_neuron_count))
        nn.init.uniform_(self.fc1.weight, -limit_in_hidden, limit_in_hidden)
        nn.init.zeros_(self.fc1.bias)
        nn.init.uniform_(self.fc2.weight, -limit_hidden_out, limit_hidden_out)
        nn.init.zeros_(self.fc2.bias)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.baseline = 0.0
        self.baseline_alpha = 0.1
    
    def update_learning_rate(self, new_lr):

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr * 0.01

    def forward(self, input_data):

        if len(input_data) < 8:
            raise ValueError("Input data must contain at least 8 tuples.")
        
        relu_values = []
        
        for tup in input_data[:(8+5)]:
            if len(tup) != len(self.tuple_scale_factors):
                raise ValueError("Each tuple must have the same number of values as there are tuple_scale_factors.")
            
            for i, value in enumerate(tup):
                scaled_value = value / self.tuple_scale_factors[i]
                relu_values.append(max(0, scaled_value))
        
        extras = input_data[(8+5):]
        if len(extras) > len(self.extra_scale_factors):
            raise ValueError("There are more extra values than extra_scale_factors provided.")

        for i, extra in enumerate(extras):
            scaled_extra = extra * self.extra_scale_factors[i]
            relu_values.append(max(0, scaled_extra))
        
        flattened = torch.tensor(relu_values, dtype=torch.float32)
        if flattened.numel() != self.input_neuron_count:
            print(f"Flattened data length: {flattened.numel()}, Expected input neuron count: {self.input_neuron_count}")
            print(f"Flattened data: {flattened}")
            print(f"Input data: {input_data}")
            raise ValueError("Flattened data length does not match the expected input neuron count.")
        
        hidden = self.fc1(flattened)
        hidden = F.relu(hidden)
        
        outputs = self.fc2(hidden)
        outputs = F.relu(outputs)
        return outputs
    
    def mutate(self, mutation_factor):
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.empty_like(param).uniform_(-mutation_factor, mutation_factor)
                param.add_(noise)

    def train_iteration(self, input_data, performance, action_taken_index):

        # I don't think this is even doing anything lmao
        self.optimizer.zero_grad()
        outputs = self.forward(input_data)
        log_probs = F.log_softmax(outputs, dim=0)

        advantage = performance - self.baseline
        self.baseline += self.baseline_alpha * advantage

        loss = -advantage * log_probs[action_taken_index]

        action_probs = F.softmax(outputs, dim=0)
        entropy = -torch.sum(action_probs * log_probs)
        loss -= 0.01 * entropy

        loss.backward()
        self.optimizer.step()


def softmax(values):
    exp_values = [math.exp(-v) for v in values]
    sum_exp = sum(exp_values)
    return [ex / sum_exp for ex in exp_values]

def decide_value(decided, possibilities, beta=1.0):
    if not possibilities:
        raise ValueError("The possibilities list must not be empty.")

    weights = []
    for candidate in possibilities:
        distance = abs(decided - candidate)
        sigma = max(0.00001, 1.0 / (beta + 1e-9 + GLOBAL_DECISIVENESS))  # avoid division by zero
        weight = math.exp(- (distance ** 2) / (2 * sigma ** 2)) + 0.0000001
        weights.append(weight)

    return random.choices(possibilities, weights=weights, k=1)[0]

def clamp(value, min_v=0, max_v=1):
    return max(min(value, max_v), min_v)

class Cell:
    def __init__(self):
        self.color = (max(LOWEST_COLOR, random.randint(0, 200)), 0, 0)
        self.active = 1
        self.type = 1
        self.brain = CellBrain()
        self.mutation_factor = 0.2
        self.energy = 30
        self.max_energy = 1000
        self.memory = (0, 0 ,0)
        self.age = 0
        self.brain.update_learning_rate(self.mutation_factor)
        self.highest_performance = 0
        self.past_performances = []
        self.avg_performance = 0
        self.sight_distance = 5
        self.look_direction = random.randint(0, 3)  # 0: up, 1: right, 2: down, 3: left
        self.last_action_index = 0
        self.decisiveness = 1
        self.moved = 0


    def change_energy(self, energy):
        self.energy += energy
        if self.energy > self.max_energy:
            self.energy = self.max_energy
        if self.energy < 0:
            self.energy = 0

    def prepare_input(self, neighbors):
        brain_input = copy.deepcopy(neighbors)
        brain_input.append(self.energy)
        brain_input.append(self.memory[0])
        brain_input.append(self.memory[1])
        brain_input.append(self.memory[2])
        return brain_input
    
    def activation(self, x):
        x = clamp(x.item(), -10, 10)
        return x # 1 / (1 + math.exp(-x))

    def cycle(self, neighbors):
        self.age += 1

        decisions = self.brain.forward(self.prepare_input(neighbors))

        action_taken_index = torch.argmax(decisions).item()  

        move = (decide_value(clamp(self.activation(decisions[0])), [0, 1], self.decisiveness) - decide_value(clamp(self.activation(decisions[1])), [0, 1], self.decisiveness),
                decide_value(clamp(self.activation(decisions[2])), [0, 1], self.decisiveness) - decide_value(clamp(self.activation(decisions[3])), [0, 1], self.decisiveness))
        reproduce = decide_value(self.activation(decisions[4]), [0, 1], self.decisiveness)
        self.memory = (self.activation(decisions[5]), self.activation(decisions[6]), self.activation(decisions[7]))
        self.look_direction = decide_value(decisions[8] * 3, [0, 1, 2, 3], self.decisiveness)

        action = {"move": move, "reproduce": reproduce, "look_direction": self.look_direction}

        self.brain.update_learning_rate(self.mutation_factor / (self.age / 4000))
        
        self.last_action_index = action_taken_index
    
        return action
    
    def learn(self, performance, neighbors):
        if performance > self.highest_performance:
            self.highest_performance = performance
        self.past_performances.append(performance)
        if len(self.past_performances) > 10:
            self.past_performances.pop(0)
        self.avg_performance = sum(self.past_performances) / len(self.past_performances)
        
        self.brain.train_iteration(self.prepare_input(neighbors), performance, self.last_action_index)
    
    def reproduce(self, child_energy):

        child = Cell()
        child.age = 0
        #me when torch doesnt let you deepcopy
        child.brain.load_state_dict(self.brain.state_dict())
        child.brain.mutate(self.mutation_factor / 10)
        child.color = tuple(
            int(max(LOWEST_COLOR, min(255, c + random.uniform(-self.mutation_factor, self.mutation_factor) / 10 * 255)))
            for c in self.color
        )
        child.energy = child_energy
        child.mutation_factor = clamp(self.mutation_factor + random.uniform(-self.mutation_factor, self.mutation_factor)) + GLOBAL_RADITION
        child.memory = (0, 0, self.memory[2])
        child.brain.update_learning_rate(child.mutation_factor)
        child.decisiveness += random.uniform(-self.mutation_factor, self.mutation_factor) / 5
        return child


empty_cell = Cell()
empty_cell.active = 0
empty_cell.type = 0
empty_cell.brain = None
empty_cell.color = (0, 0, 0)
empty_cell.energy = 0




PLANT_NATURAL_GROWTH = 200
class Plant(Cell):
    def __init__(self):
        super().__init__()
        self.color = (0, 255, 0)
        self.energy = 20
        self.type = 2
        self.active = 1
        self.brain = None

    def cycle(self, neighbors):
        self.age += 1
        return {"move": (0, 0), "reproduce": 0, "look_direction": 0}
    
    def learn(self, performance, neighbors):
        pass


class Grid:
    def __init__(self, rows, cols, scale, starting_density=0.5, padding=2):
        self.rows = rows
        self.cols = cols

        self.state = [[empty_cell for y in range(cols)] for x in range(rows)]
        self.scale = scale
        self.screen = pygame.display.set_mode((cols * scale, rows * scale))
        self.epoch = 0
        self.live_cells = rows * cols
        self.live_plants = 0

        for x in range(rows):
            for y in range(cols):
                if x < padding or y < padding or x > rows - padding - 1 or y > rows - padding - 1:
                    continue
                if random.random() < starting_density:
                    self.state[x][y] = Cell()


    def get_neighbors(self, x, y):
        neighbors = []
        for n_x in [-1, 0, 1]:
                    for n_y in [-1, 0, 1]:
                        if (n_x, n_y) == (0, 0):
                            continue
                        if x + n_x >= self.rows or x + n_x < 0 or y + n_y >= self.cols or y + n_y < 0:
                            neighbors.append((0, 0, 0, 0))
                            continue
                        n_cell = self.state[x + n_x][y + n_y]
                        neighbors.append(n_cell.color + (n_cell.energy,))

        return neighbors
    
    def get_sight(self, x, y, direction, distance):
        sight = []
        for i in range(distance):
            if direction == 0:
                if x - i < 0:
                    sight.append((0, 0, 0, 0))
                    continue
                n_cell = self.state[x - i][y]
            elif direction == 1:
                if y + i >= self.cols:
                    sight.append((0, 0, 0, 0))
                    continue
                n_cell = self.state[x][y + i]
            elif direction == 2:
                if x + i >= self.rows:
                    sight.append((0, 0, 0, 0))
                    continue
                n_cell = self.state[x + i][y]
            elif direction == 3:
                if y - i < 0:
                    sight.append((0, 0, 0, 0))
                    continue
                n_cell = self.state[x][y - i]
            sight.append(n_cell.color + (n_cell.energy,))

        return sight
    

    
    def update(self):

        new_state = [[None for y in range(self.cols)] for x in range(self.rows)]

        for x in range(self.rows):
            for y in range(self.cols):
                cell = self.state[x][y]
                if not cell.active:
                    continue
                cell_performance = 0
                neighbors = self.get_neighbors(x, y) + self.get_sight(x, y, cell.look_direction, 5)
                action = cell.cycle(neighbors)

                cell.look_direction = action["look_direction"]

                energy_consumption = (-GLOBAL_ENERGY * cell.moved) + GLOBAL_ENERGY / 2

                move = action["move"]

                if cell.energy > abs(move[0]) + abs(move[1]):

                    new_x = max(0, min(self.rows - 1, x + move[0]))
                    new_y = max(0, min(self.cols - 1, y + move[1]))
                    if abs(move[0]) + abs(move[1]) > 0:
                        cell_performance += performance_calcs["move"]
                    else:
                        cell_performance += performance_calcs["stay"]
                    
                    cell.moved = 1

                    energy_consumption += abs(move[0]) + abs(move[1])
                    cell.change_energy(-energy_consumption)
                
                else: 

                    cell.moved = 0

                    new_x = x
                    new_y = y
                
                if action["reproduce"] == 1 and cell.energy > 20 and not(new_x == x and new_y == y):
                    child_energy = int(cell.energy / 2)
                    cell_performance += performance_calcs["reproduce"]
                    cell.change_energy(-child_energy)
                    new_state[x][y] = cell.reproduce(child_energy)
                
                # if cell.energy == 0:
                #     continue
                cell_performance += cell.energy * 0.1
                if new_state[new_x][new_y] is None:
                    cell.learn(cell_performance, neighbors)
                    new_state[new_x][new_y] = cell
                else:
                    if new_x == x and new_y == y:
                        cell.learn(cell_performance, neighbors)
                        new_state[new_x][new_y] = cell
                    else:
                        if new_state[new_x][new_y].type == 2:
                            cell_performance += performance_calcs["eat"]
                            cell.learn(cell_performance, neighbors)
                            cell.change_energy(new_state[new_x][new_y].energy)
                            new_state[new_x][new_y] = cell
                        elif cell.energy > new_state[new_x][new_y].energy:
                            cell_performance += performance_calcs["battle"]
                            cell.learn(cell_performance, neighbors)
                            cell.change_energy(int(new_state[new_x][new_y].energy / 2))
                            new_state[new_x][new_y] = cell
                        else:
                            new_state[new_x][new_y].change_energy(int(cell.energy / 2))
                            continue

                    
        cell_count = 0
        plant_count = 0
        
        plant_cells = random.sample(range(0, self.rows*self.cols), min(max(0, PLANT_NATURAL_GROWTH - self.live_plants), self.rows * self.cols))
        grid_cell = 0
        for x in range(self.rows):
            for y in range(self.cols):
                if new_state[x][y] == None:

                    if grid_cell in plant_cells:
                        new_state[x][y] = Plant()
                    else:
                        new_state[x][y] = empty_cell
                else:
                    if grid_cell in plant_cells:
                        plant_cells.append(grid_cell + 1)
                    if new_state[x][y].type == 1:
                        cell_count += 1
                    elif new_state[x][y].type == 2:
                        plant_count += 1

                    # if random.random() > 0.9 + 5 * (self.live_cells / (self.rows * self.cols)):
                    #     new_state[x][y].change_energy(1)
                grid_cell += 1

        self.live_plants = plant_count
        self.live_cells = cell_count
        self.state = new_state
        self.epoch += 1

    def draw(self):

        self.screen.fill((0, 0, 0))

        for x in range(self.rows):
            for y in range(self.cols):
                cell = self.state[x][y]
                if cell == empty_cell:
                    continue
                else:
                    rect = pygame.Rect(
                            y * self.scale,
                            x * self.scale,
                            self.scale,
                            self.scale,
                    )
                    if 1 == 1:# if cell.type == 1:
                        pygame.draw.rect(self.screen, cell.color, rect)
                    else:
                        pygame.draw.circle(self.screen, cell.color, (rect.left + rect.width // 2, rect.top + rect.height // 2), rect.width // 2)


        pygame.display.flip()

    def print_stats(self):
        live_cells_count = 0
        total_energy = 0
        total_age = 0
        total_mutation = 0
        max_age = 0
        max_energy = 0
        min_energy = float('inf')
        max_mutation = 0
        min_mutation = float('inf')
        max_high_performance = float('-inf')
        min_high_performance = float('inf')
        max_avg_performance = float('-inf')
        min_avg_performance = float('inf')
        avg_high_performance = 0
        avg_avg_performance = 0
        max_decisiveness = float('-inf')
        min_decisiveness = float('inf')
        avg_decisiveness = 0
        color_sum = [0, 0, 0]

        for row in self.state:
            for cell in row:
                if cell.type == 1:
                    live_cells_count += 1
                    total_energy += cell.energy
                    total_age += cell.age
                    total_mutation += cell.mutation_factor
                    if cell.age > max_age:
                        max_age = cell.age
                    if cell.energy > max_energy:
                        max_energy = cell.energy
                    if cell.energy < min_energy:
                        min_energy = cell.energy
                    if cell.mutation_factor > max_mutation:
                        max_mutation = cell.mutation_factor
                    if cell.mutation_factor < min_mutation:
                        min_mutation = cell.mutation_factor
                    if cell.highest_performance > max_high_performance:
                        max_high_performance = cell.highest_performance
                    if cell.highest_performance < min_high_performance:
                        min_high_performance = cell.highest_performance
                    if cell.avg_performance > max_avg_performance:
                        max_avg_performance = cell.avg_performance
                    if cell.avg_performance < min_avg_performance:
                        min_avg_performance = cell.avg_performance
                    if cell.decisiveness > max_decisiveness:
                        max_decisiveness = cell.decisiveness
                    if cell.decisiveness < min_decisiveness:
                        min_decisiveness = cell.decisiveness
                    avg_decisiveness += cell.decisiveness
                    avg_high_performance += cell.highest_performance
                    avg_avg_performance += cell.avg_performance
                    
                        

                    color_sum[0] += cell.color[0]
                    color_sum[1] += cell.color[1]
                    color_sum[2] += cell.color[2]

        total_cells = self.rows * self.cols
        print("")
        print("Epoch:", self.epoch)
        print("Live cells:", live_cells_count)
        print("Live plants:", self.live_plants)
        print("Total cells:", total_cells)
        print("Density:", live_cells_count / total_cells if total_cells > 0 else 0)

        if live_cells_count > 0:
            avg_energy = total_energy / live_cells_count
            avg_age = total_age / live_cells_count
            avg_mutation = total_mutation / live_cells_count
            avg_high_performance = avg_high_performance / live_cells_count
            avg_avg_performance = avg_avg_performance / live_cells_count
            avg_decisiveness = avg_decisiveness / live_cells_count
            avg_color = (
                color_sum[0] // live_cells_count,
                color_sum[1] // live_cells_count,
                color_sum[2] // live_cells_count
            )
        else:
            avg_energy = avg_age = avg_mutation = 0
            max_age = max_energy = min_energy = 0
            max_mutation = min_mutation = 0
            avg_high_performance = avg_avg_performance = 0
            min_high_performance = 0
            min_avg_performance = 0
            max_decisiveness = min_decisiveness = 0
            avg_decisiveness = 0
            avg_color = (0, 0, 0)

        print("Average energy:", avg_energy)
        print("Average age:", avg_age)
        print("Average mutation factor:", avg_mutation)
        print("Oldest cell age:", max_age)
        print("Highest energy cell:", max_energy)
        print("Lowest energy cell:", min_energy if min_energy != float('inf') else 0)
        print("Highest mutation factor:", max_mutation)
        print("Lowest mutation factor:", min_mutation if min_mutation != float('inf') else 0)
        print("Average color:", avg_color)
        print("Highest performance:", max_high_performance)
        print("Lowest performance:", min_high_performance if min_high_performance != float('inf') else 0)
        print("Average performance:", avg_avg_performance)
        print("Average highest performance:", avg_high_performance)
        print("Lowest average performance:", min_avg_performance if min_avg_performance != float('inf') else 0)
        print("Highest average performance:", max_avg_performance)
        print("Lowest decisiveness:", min_decisiveness if min_decisiveness != float('inf') else 0)
        print("Highest decisiveness:", max_decisiveness)
        print("Average decisiveness:", avg_decisiveness)
        print("")



# grid = Grid(100, 100, 10, padding=10, starting_density=0.4)

GRID_SIZE = 80
grid = Grid(GRID_SIZE, GRID_SIZE, 1000 // GRID_SIZE, padding=0, starting_density=0.05)

grid.draw()

running = True

running = True
auto_cycle = False
limit_fps = False
stats = False

clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                grid.update()
                grid.draw()
            elif event.key == pygame.K_p:
                auto_cycle = not auto_cycle
                print("Auto cycle", "ON" if auto_cycle else "OFF")
            elif event.key == pygame.K_y:
                limit_fps = not limit_fps
                print("FPS limit", "ON" if limit_fps else "OFF")
            elif event.key == pygame.K_j:
                grid.print_stats()
            elif event.key == pygame.K_t:
                PLANT_NATURAL_GROWTH = int(PLANT_NATURAL_GROWTH * 2)
                if PLANT_NATURAL_GROWTH == 0:
                    PLANT_NATURAL_GROWTH = 1
                print("Plant natural growth:", PLANT_NATURAL_GROWTH)
            elif event.key == pygame.K_g:
                PLANT_NATURAL_GROWTH = max(0, int(PLANT_NATURAL_GROWTH / 2))
                print("Plant natural growth:", PLANT_NATURAL_GROWTH)
            elif event.key == pygame.K_r:
                GLOBAL_RADITION += 0.01
                print("Global radition:", GLOBAL_RADITION)
            elif event.key == pygame.K_f:
                GLOBAL_RADITION = 0
                print("Global radition:", GLOBAL_RADITION)
            elif event.key == pygame.K_e:
                GLOBAL_ENERGY += 0.1
                print("Global energy:", GLOBAL_ENERGY)
            elif event.key == pygame.K_d:
                GLOBAL_ENERGY -= 0.1
                if GLOBAL_ENERGY < 0:
                    GLOBAL_ENERGY = 0
                print("Global energy:", GLOBAL_ENERGY)
            elif event.key == pygame.K_q:
                GLOBAL_DECISIVENESS += 0.1
                print("Global decisiveness:", GLOBAL_DECISIVENESS)
            elif event.key == pygame.K_a:
                GLOBAL_DECISIVENESS -= 0.1
                print("Global decisiveness:", GLOBAL_DECISIVENESS)
    
    if auto_cycle:
        grid.update()
        grid.draw()
    pygame.display.flip()
    if limit_fps:
        clock.tick(30)

pygame.quit()