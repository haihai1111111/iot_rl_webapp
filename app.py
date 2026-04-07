from flask import Flask, render_template, jsonify, request
import numpy as np
import random
from collections import deque
from itertools import product
import threading
import time
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

np.random.seed(42)
random.seed(42)

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.gamma = 0.95
        self.alpha = 0.1
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def get_q(self, state):
        state_tuple = tuple(state)
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_size)
        return self.q_table[state_tuple]
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.get_q(state)
        return int(np.argmax(q_values))
    
    def learn(self, state, action, reward, next_state, done):
        current_q = self.get_q(state)
        next_q = self.get_q(next_state)
        target = reward if done else reward + self.gamma * np.max(next_q)
        current_q[action] += self.alpha * (target - current_q[action])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def reset(self):
        self.q_table = {}
        self.epsilon = 1.0

class IoTChannelGameRL:
    def __init__(self, n_players=4):
        self.n_players = n_players
        self.state_size = 3
        self.action_size = 2
        self.agents = [QLearningAgent(self.state_size, self.action_size) for _ in range(n_players)]
        self.transition_probs = self._init_transition_probs()
        self.rewards = self._init_rewards()
        self.training_rewards = [[] for _ in range(n_players)]
        self.training_progress = 0
        self.is_training = False
        
    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.training_rewards = [[] for _ in range(self.n_players)]
        self.training_progress = 0
        
    def _init_transition_probs(self):
        transition = {}
        action_combinations = list(product([0, 1], repeat=self.n_players))
        for s in [0, 1]:
            for action_tuple in action_combinations:
                num_transmitting = sum(action_tuple)
                if s == 0:
                    if num_transmitting == 0: prob_idle = 0.95
                    elif num_transmitting == 1: prob_idle = 0.85
                    elif num_transmitting == 2: prob_idle = 0.70
                    elif num_transmitting == 3: prob_idle = 0.50
                    else: prob_idle = 0.30
                else:
                    if num_transmitting == 0: prob_idle = 0.70
                    elif num_transmitting == 1: prob_idle = 0.60
                    elif num_transmitting == 2: prob_idle = 0.45
                    elif num_transmitting == 3: prob_idle = 0.30
                    else: prob_idle = 0.20
                transition[(s, action_tuple)] = {0: prob_idle, 1: 1 - prob_idle}
        return transition
    
    def _init_rewards(self):
        rewards = {}
        action_combinations = list(product([0, 1], repeat=self.n_players))
        for s in [0, 1]:
            for action_tuple in action_combinations:
                reward_per_device = []
                num_transmitting = sum(action_tuple)
                for player_id in range(self.n_players):
                    if action_tuple[player_id] == 1:
                        if s == 0:
                            if num_transmitting == 1: reward = 10.0
                            elif num_transmitting == 2: reward = 7.0
                            elif num_transmitting == 3: reward = 4.0
                            else: reward = 2.0
                        else:
                            if num_transmitting == 1: reward = 5.0
                            elif num_transmitting == 2: reward = 2.0
                            elif num_transmitting == 3: reward = -1.0
                            else: reward = -4.0
                    else:
                        if s == 1 and num_transmitting >= 3: reward = -1.0
                        elif s == 1 and num_transmitting <= 2: reward = 0.5
                        else: reward = 0.0
                    reward_per_device.append(reward)
                rewards[(s, action_tuple)] = reward_per_device
        return rewards
    
    def get_state(self, channel_state, player_id, history):
        avg_transmit = np.mean(history[-10:]) if len(history) > 0 else 0
        return np.array([channel_state, player_id / self.n_players, avg_transmit])
    
    def train_episode(self, max_steps=100):
        channel_state = 0
        episode_rewards = [0] * self.n_players
        history = [[] for _ in range(self.n_players)]
        step_data = []
        
        for step in range(max_steps):
            actions = []
            for i in range(self.n_players):
                state = self.get_state(channel_state, i, history[i])
                action = self.agents[i].act(state, training=True)
                actions.append(action)
                history[i].append(action)
            
            actions_tuple = tuple(actions)
            rewards = self.rewards[(channel_state, actions_tuple)]
            trans_probs = self.transition_probs[(channel_state, actions_tuple)]
            next_channel_state = np.random.choice([0, 1], p=[trans_probs[0], trans_probs[1]])
            
            for i in range(self.n_players):
                state = self.get_state(channel_state, i, history[i][:-1])
                next_state = self.get_state(next_channel_state, i, history[i])
                done = (step == max_steps - 1)
                self.agents[i].learn(state, actions[i], rewards[i], next_state, done)
                episode_rewards[i] += rewards[i]
            
            step_data.append({
                'step': step, 'channel_state': int(channel_state), 'actions': actions,
                'rewards': rewards, 'num_transmitting': int(sum(actions))
            })
            channel_state = next_channel_state
        
        return episode_rewards, step_data
    
    def train(self, episodes=500, callback=None):
        self.is_training = True
        print("Starting training...")
        
        for episode in range(episodes):
            episode_rewards, _ = self.train_episode()
            for i in range(self.n_players):
                self.training_rewards[i].append(np.mean(episode_rewards))
            
            self.training_progress = (episode + 1) / episodes * 100
            
            if callback:
                callback(episode + 1, episodes, self.training_rewards)
            
            if (episode + 1) % 100 == 0:
                avg_rewards = [np.mean(r[-100:]) for r in self.training_rewards]
                print(f"Episode {episode + 1}/{episodes} - Rewards: {[f'{r:.2f}' for r in avg_rewards]}")
        
        print("Training completed!")
        self.is_training = False
        return self.get_policy()
    
    def get_policy(self):
        policy = {0: [], 1: []}
        for state in [0, 1]:
            for player in range(self.n_players):
                state_vec = self.get_state(state, player, [0] * 10)
                action = self.agents[player].act(state_vec, training=False)
                policy[state].append(int(action))
            policy[state] = tuple(policy[state])
        return policy
    
    def simulate(self, policy, steps=200):
        channel_state = 0
        history = []
        for step in range(steps):
            actions = policy[channel_state]
            rewards = self.rewards[(channel_state, actions)]
            history.append({
                'step': step, 
                'state': 'Idle' if channel_state == 0 else 'Congested',
                'state_code': int(channel_state), 
                'actions': [int(a) for a in actions],
                'rewards': [float(r) for r in rewards],
                'num_transmitting': int(sum(actions)),
                'total_reward': float(sum(rewards))
            })
            trans_probs = self.transition_probs[(channel_state, actions)]
            channel_state = np.random.choice([0, 1], p=[trans_probs[0], trans_probs[1]])
        return history
    
    def get_training_data(self):
        return {
            'rewards': [[float(r) for r in rewards] for rewards in self.training_rewards],
            'epsilons': [float(agent.epsilon) for agent in self.agents],
            'progress': self.training_progress,
            'is_training': self.is_training
        }

# Create global instance
game = IoTChannelGameRL(n_players=4)
policy = game.train(episodes=500)
simulation_history = game.simulate(policy, steps=200)

training_thread = None
training_cancelled = False

def run_training(episodes=500):
    global training_cancelled, policy, simulation_history
    training_cancelled = False
    
    def update_progress(episode, total, rewards):
        if training_cancelled:
            return False
    
    game.reset()
    new_policy = game.train(episodes=episodes, callback=update_progress)
    if not training_cancelled:
        policy = new_policy
        simulation_history = game.simulate(policy, steps=200)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/policy')
def get_policy():
    return jsonify({
        'idle': [int(a) for a in policy[0]],
        'congested': [int(a) for a in policy[1]]
    })

@app.route('/api/simulation')
def get_simulation():
    return jsonify(simulation_history)

@app.route('/api/training')
def get_training():
    data = game.get_training_data()
    return jsonify({
        'rewards': data['rewards'],
        'epsilons': data['epsilons'],
        'progress': data['progress'],
        'is_training': data['is_training']
    })

@app.route('/api/stats')
def get_stats():
    idle_ratio = sum(1 for h in simulation_history if h['state_code'] == 0) / len(simulation_history)
    avg_transmitting = np.mean([h['num_transmitting'] for h in simulation_history])
    avg_rewards = np.mean([h['rewards'] for h in simulation_history], axis=0)
    return jsonify({
        'idle_ratio': float(idle_ratio * 100),
        'avg_transmitting': float(avg_transmitting),
        'avg_rewards': [float(r) for r in avg_rewards],
        'total_reward': float(sum(avg_rewards)),
        'steps': len(simulation_history)
    })

@app.route('/api/exploration')
def get_exploration():
    test_rewards = []
    for epsilon in [0.0, 0.1, 0.2, 0.5, 1.0]:
        test_agent = QLearningAgent(3, 2)
        test_agent.epsilon = epsilon
        temp_rewards = []
        for _ in range(10):
            channel = 0
            total_reward = 0
            for _ in range(50):
                action = test_agent.act([channel, 0, 0], training=False)
                if action == 1:
                    reward = 10 if channel == 0 else -2
                else:
                    reward = 0
                total_reward += reward
                channel = np.random.choice([0, 1], p=[0.7, 0.3])
            temp_rewards.append(total_reward)
        test_rewards.append(np.mean(temp_rewards))
    return jsonify({
        'epsilons': [0.0, 0.1, 0.2, 0.5, 1.0],
        'rewards': [float(r) for r in test_rewards]
    })

@app.route('/api/retrain', methods=['POST'])
def retrain():
    global training_thread
    data = request.get_json()
    episodes = data.get('episodes', 500)
    
    training_thread = threading.Thread(target=run_training, args=(episodes,))
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'status': 'started', 'episodes': episodes})

@app.route('/api/training-status')
def training_status():
    return jsonify({
        'is_training': game.is_training,
        'progress': game.training_progress
    })

@app.route('/api/chart-data')
def get_chart_data():
    """提供图表数据，用于前端简单绘图"""
    data = game.get_training_data()
    
    # 准备简化的图表数据
    chart_data = {
        'training': {
            'labels': list(range(len(data['rewards'][0]))),
            'datasets': []
        },
        'exploration': {
            'labels': [0.0, 0.1, 0.2, 0.5, 1.0],
            'rewards': []
        }
    }
    
    # 添加训练数据
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, rewards in enumerate(data['rewards']):
        chart_data['training']['datasets'].append({
            'label': f'Device {i+1}',
            'data': rewards,
            'color': colors[i % len(colors)]
        })
    
    # 添加探索数据
    exp_res = []
    for epsilon in [0.0, 0.1, 0.2, 0.5, 1.0]:
        test_agent = QLearningAgent(3, 2)
        test_agent.epsilon = epsilon
        temp_rewards = []
        for _ in range(10):
            channel = 0
            total_reward = 0
            for _ in range(50):
                action = test_agent.act([channel, 0, 0], training=False)
                if action == 1:
                    reward = 10 if channel == 0 else -2
                else:
                    reward = 0
                total_reward += reward
                channel = np.random.choice([0, 1], p=[0.7, 0.3])
            temp_rewards.append(total_reward)
        exp_res.append(np.mean(temp_rewards))
    
    chart_data['exploration']['rewards'] = exp_res
    
    return jsonify(chart_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)