import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, Counter
import os
import matplotlib.pyplot as plt


# Define Rock Agent
class RockAgent:
    def decide_move(self, *args, **kwargs):
        return 0  # Always plays "Rock"


# Define Random Agent
class RandomAgent:
    def decide_move(self, *args, **kwargs):
        return random.randint(0, 2)  # Random move


# Define Statistical Agent
class StatisticalAgent:
    def __init__(self):
        self.history = []

    def decide_move(self, *args, **kwargs):
        if not self.history:
            return random.randint(0, 2)  # Random move if history is empty
        # Flatten the history list to just opponent moves
        opponent_moves = [move[0] for move in self.history]
        counts = Counter(opponent_moves)
        most_common_move = max(counts, key=counts.get)  # Get the most common opponent move
        return (most_common_move + 1) % 3  # Counter the most common move

    def update_history(self, opponent_move, agent_move):
        if (agent_move - opponent_move) % 3 == 1:  # Winning condition
            self.history.append((opponent_move, agent_move))


class MarkovAgent:
    def __init__(self):
        self.history = []

    def decide_move(self, *args, **kwargs):
        if len(self.history) < 2:
            return random.randint(0, 2)  # Play randomly if not enough history

        # Build the transition matrix
        transition_counts = np.zeros((3, 3))  # From move X to move Y
        for i in range(len(self.history) - 1):
            prev_move, next_move = self.history[i]
            transition_counts[prev_move][next_move] += 1

        # Calculate transition probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        transition_probs = transition_counts / np.maximum(row_sums, 1)

        # Predict the next move
        last_move = self.history[-1][1]  # Agent's last move
        predicted_move = np.argmax(transition_probs[last_move])

        # Counter the predicted move
        return (predicted_move + 1) % 3

    def update_history(self, opponent_move, agent_move):
        if (agent_move - opponent_move) % 3 == 1:  # Winning condition
            self.history.append((opponent_move, agent_move))


# Neural Network for DQN
class DQNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DQN Agent Class
class DQNAgent:
    def __init__(self, input_size, output_size, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=0.001,
                 model_path="dqn_model.pth"):
        self.input_size = input_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQNN(input_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=2000)  # Experience Replay Buffer
        self.model_path = model_path

        # Load pre-trained model if exists
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"Loaded model from {self.model_path}")

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:  # Exploration
            return random.randint(0, self.output_size - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()  # Exploitation

    def replay(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Current Q-values
        q_values = self.model(states).gather(1, actions).squeeze()

        # Target Q-values
        with torch.no_grad():
            max_next_q_values = self.model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Loss and optimization
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log Q-values for debugging
        print(f"Q-values (sample): {q_values[:5].tolist()}")  # Log first 5 Q-values for monitoring

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")


# New agents
class CycleAgent:
    def __init__(self):
        self.moves = [0, 1, 2]  # Rock, Paper, Scissors
        self.index = 0

    def decide_move(self, *args, **kwargs):
        move = self.moves[self.index]
        self.index = (self.index + 1) % 3  # Cycle through the moves
        return move


class MirrorAgent:
    def __init__(self):
        self.last_opponent_move = None

    def decide_move(self, *args, **kwargs):
        if self.last_opponent_move is None:
            return random.randint(0, 2)  # Play randomly if no history
        return self.last_opponent_move  # Mirror the opponent's previous move

    def update_history(self, opponent_move, agent_move):
        self.last_opponent_move = opponent_move


class AdaptiveAgent:
    def __init__(self):
        self.history = []

    def decide_move(self, *args, **kwargs):
        if len(self.history) < 2:
            return random.randint(0, 2)  # Play randomly if not enough history

        # Simple adaptation logic: play the move that has the best performance in past rounds
        win_count = [0, 0, 0]  # Rock, Paper, Scissors win counts
        for i in range(len(self.history)):
            agent_move, opponent_move = self.history[i]
            if (agent_move - opponent_move) % 3 == 1:  # Agent win
                win_count[agent_move] += 1

        return np.argmax(win_count)  # Play the move that has won the most times

    def update_history(self, opponent_move, agent_move):
        self.history.append((agent_move, opponent_move))


# Simulate the Game
def simulate_dqn(agent, opponent, rounds=5000, history_length=50):
    state = [0] * history_length  # Initial state (history of last moves, filled with zeros)
    scores = {"agent": 0, "opponent": 0, "ties": 0}  # Initialize the scores correctly
    score_history = {"agent": [], "opponent": [], "ties": []}

    for step in range(rounds):
        opponent_move = opponent.decide_move()
        action = agent.act(state)

        # Reward calculation
        reward = 0
        if action == opponent_move:
            scores["ties"] += 1
        elif (action - opponent_move) % 3 == 1:
            reward = 1
            scores["agent"] += 1
        else:
            reward = -1
            scores["opponent"] += 1

        # Update state
        next_state = state[1:] + [opponent_move]  # Add last opponent move to history
        done = step == rounds - 1
        agent.remember(state, action, reward, next_state, done)

        state = next_state

        # Replay experience for training
        agent.replay(batch_size=32)

        if hasattr(opponent, "update_history"):
            opponent.update_history(opponent_move, action)  # Pass both moves

        # Save score history for plotting
        score_history["agent"].append(scores["agent"])
        score_history["opponent"].append(scores["opponent"])
        score_history["ties"].append(scores["ties"])

    return scores, score_history  # Return scores and score history for plotting


# Plot Game Performance with Bar Chart
def plot_agent_performance(agent_name, score_history):
    # Line plot for the performance (Wins, Opponent Wins, Ties)
    plt.figure(figsize=(10, 6))
    plt.plot(score_history["agent"], label="Agent Wins", color="green")
    plt.plot(score_history["opponent"], label="Opponent Wins", color="red")
    plt.plot(score_history["ties"], label="Ties", color="blue")
    plt.title(f"Performance of {agent_name} (Line Plot)")
    plt.xlabel("Rounds")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Bar plot for final performance (Total Wins, Losses, and Ties)
    total_wins = score_history["agent"][-1]
    total_losses = score_history["opponent"][-1]
    total_ties = score_history["ties"][-1]

    # Bar chart
    plt.figure(figsize=(8, 6))
    labels = ['Wins', 'Losses', 'Ties']
    values = [total_wins, total_losses, total_ties]
    plt.bar(labels, values, color=['green', 'red', 'blue'])
    plt.title(f"Final Performance of {agent_name} (Bar Chart)")
    plt.ylabel("Count")
    plt.show()


# Calculate Averages
def calculate_averages(results):
    total_agent_wins = 0
    total_opponent_wins = 0
    total_ties = 0
    total_games = 0

    for agent_name, scores in results.items():
        total_agent_wins += scores["agent"]
        total_opponent_wins += scores["opponent"]
        total_ties += scores["ties"]
        total_games += scores["agent"] + scores["opponent"] + scores["ties"]

    avg_agent_wins = (total_agent_wins / total_games) * 100
    avg_opponent_wins = (total_opponent_wins / total_games) * 100
    avg_ties = (total_ties / total_games) * 100

    return avg_agent_wins, avg_opponent_wins, avg_ties


# Plot Pie Chart
def plot_pie_chart(results):
    total_agent_wins = sum([scores["agent"] for scores in results.values()])
    total_opponent_wins = sum([scores["opponent"] for scores in results.values()])
    total_ties = sum([scores["ties"] for scores in results.values()])

    labels = ['Agent Wins', 'Opponent Wins', 'Ties']
    sizes = [total_agent_wins, total_opponent_wins, total_ties]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.1, 0, 0)  # Explode the first slice (Agent Wins)

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140,
            wedgeprops={'edgecolor': 'black'})
    plt.title("Game Outcome Distribution")
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
    plt.show()


# Train DQN Against Different Agents
if __name__ == "__main__":
    input_size = 50  # Adjusted history length to match a more reasonable state dimension
    output_size = 3  # Rock, Paper, Scissors

    dqn_agent = DQNAgent(input_size, output_size)

    agents = {
        'RockAgent': RockAgent(),
        'RandomAgent': RandomAgent(),
        'StatisticalAgent': StatisticalAgent(),
        'MarkovAgent': MarkovAgent(),
        'CycleAgent': CycleAgent(),
        'MirrorAgent': MirrorAgent(),
        'AdaptiveAgent': AdaptiveAgent()
    }

    results = {}
    for agent_name, opponent in agents.items():
        print(f"Training against {agent_name}...")
        scores, score_history = simulate_dqn(dqn_agent, opponent, rounds=5000)  # Increased rounds to 5000
        results[agent_name] = scores

        # Plot performance for each agent
        plot_agent_performance(agent_name, score_history)

    # Calculate averages
    avg_agent_wins, avg_opponent_wins, avg_ties = calculate_averages(results)

    print(f"Average agent wins: {avg_agent_wins:.2f}%")
    print(f"Average opponent wins: {avg_opponent_wins:.2f}%")
    print(f"Average ties: {avg_ties:.2f}%")

    # Plot Pie Chart
    plot_pie_chart(results)
    dqn_agent.save_model()
