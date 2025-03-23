import numpy as np
import gym
from gym import spaces

class FHVPricingEnv(gym.Env):
    def __init__(self, demand_data):
        super(FHVPricingEnv, self).__init__()
        self.action_space = spaces.Box(low=0.8, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        self.demand_data = demand_data
        self.current_step = 0
        self.state = None

    def reset(self):
        self.current_step = 0
        row = self.demand_data[self.current_step]
        self.state = np.array([row['demand'], row['supply'], row['hour'], 1.0])
        return self.state

    def step(self, action):
        row = self.demand_data[self.current_step]
        demand = row['demand']
        supply = row['supply']
        time = row['hour']
        price_multiplier = float(action[0])
        base_price = 10
        fulfilled_demand = min(demand * np.exp(-0.5 * (price_multiplier - 1.0)), supply)
        revenue = fulfilled_demand * base_price * price_multiplier
        unfulfilled_penalty = (demand - fulfilled_demand) * 2
        reward = revenue - unfulfilled_penalty
        self.current_step += 1
        done = self.current_step >= len(self.demand_data)
        if not done:
            next_row = self.demand_data[self.current_step]
            next_state = np.array([next_row['demand'], next_row['supply'], next_row['hour'], price_multiplier])
        else:
            next_state = None
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass
