# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple episode runner using the RL environment."""

from __future__ import print_function

import sys
import os
import getopt
from hanabi_learning_environment import env
from hanabi_learning_environment.agents.llm_agent import LLMAgent
import time
import datetime
import numpy as np

AGENT_CLASSES = {'LLMAgent': LLMAgent}

shared_memory=[]

class Runner(object):
  """Runner class."""

  def __init__(self, flags):
    """Initialize runner."""
    self.flags = flags
    self.agent_config = {'players': flags['players'], 'shared_memory': shared_memory, 'max_memory': flags['max_memory'], 'context_level': flags['context_level'], 'weighting': flags['weighting'], 'deck_weights': flags['deck_weights'], 'weight_type': flags['weight_type'], 'log_file': flags['log_file'], 'model': flags['model'], 'logprobs': flags['logprobs']} # max_memory is the number of previous actions to include in the prompt for the LLM. This is to provide context to the LLM about the recent history of the game, which can help it make more informed decisions. By including a limited number of previous actions, we can give the LLM enough information to understand the current state of the game and the strategies being employed, without overwhelming it with too much information that may not be relevant to the current decision.
    self.environment = env.make('Hanabi-Full', num_players=flags['players'])
    self.agent_class = AGENT_CLASSES[flags['agent_class']]
    self.log_file = flags['log_file']

  def run(self):
    """Run episodes."""
    rewards = []
    for episode in range(self.flags['num_episodes']): # num_episodes == number of games to play
      self.agent_config['shared_memory'].clear()
      observations = self.environment.reset()
      agents = [self.agent_class(self.agent_config)
                for _ in range(self.flags['players'])]
      done = False
      episode_reward = 0
      turn = 0 
      while not done:
        for agent_id, agent in enumerate(agents):
          observation = observations['player_observations'][agent_id]
          action = agent.act(observation) 
          if observation['current_player'] == agent_id:
            assert action is not None
            current_player_action = action
          else:
            assert action is None
        # Make an environment step.
        observations, reward, done, unused_info = self.environment.step(
            current_player_action, self.log_file)
        
        # Print and log current gameplay
        if 'target_offset' in current_player_action: # If the action is a hint, log which player recieved the hint
          target_offset = current_player_action['target_offset']
          target_player = (observation['current_player'] + target_offset) % self.flags['players']
          target_player_string = 'Target player: {}'.format(target_player)
        else: 
          target_player_string = ''
          
        print('Agent: {} action: {}. {}'.format(observation['current_player'],
                                            current_player_action, target_player_string))
        # check for target offset in the current_player_action and adjust if necessary
        episode_reward += reward
        if self.log_file:
          with open('logs/full_logs/'+self.log_file, 'a') as f, open('logs/gameplay/'+self.log_file, 'a') as f_gameplay:
            f.write('Agent: {} action: {}. {}\n'.format(observation['current_player'],
                                            current_player_action, target_player_string))
            f_gameplay.write('Agent: {} action: {}. {}\n'.format(observation['current_player'],
                                            current_player_action, target_player_string))
          action_taken = current_player_action['action_type'] if current_player_action else 'None'
          with open('logs/stats/'+self.log_file+'.csv', 'a') as f:
            f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(turn, observation['current_player'], action_taken, observation['life_tokens'], observation['information_tokens'], observation['fireworks']['R'], observation['fireworks']['Y'], observation['fireworks']['G'], observation['fireworks']['W'], observation['fireworks']['B'], episode_reward))
        # Every 10 turns print the current state of the game.
        #if turn % 10 == 0:
        print('Current state of the game: {}'.format(self.environment.state))
        #print('Current memory of the agent: {}'.format(shared_memory[-self.agent_config['max_memory']:]))
        if self.log_file:
          with open('logs/full_logs/'+self.log_file, 'a') as f, open('logs/gameplay/'+self.log_file, 'a') as f_gameplay:
            f.write('Current state of the game: {}\n'.format(self.environment.state))
            f_gameplay.write('Current state of the game: {}\n'.format(self.environment.state))
            if self.flags['weighting'] == True: 
              for agent_id in range(self.flags['players']):
                current_deck = [observation['fireworks']['R'], observation['fireworks']['Y'], observation['fireworks']['G'], observation['fireworks']['W'], observation['fireworks']['B']]
                player_weights = self.flags['deck_weights'][agent_id]
                local_score = []
                [local_score.append(current_deck[i] * player_weights[i]) for i in range(len(current_deck))]
                print('Local score of player {}: {}'.format(agent_id, local_score))
                f.write('Local score of player {}: {}\n'.format(agent_id, local_score))
                f_gameplay.write('Local score of player {}: {}\n'.format(agent_id, local_score))
        turn += 1 
      rewards.append(episode_reward)
      print('Running episode: %d' % episode)
      print('Max Reward: %.3f' % max(rewards))
      if self.log_file:
        with open('logs/full_logs/'+self.log_file, 'a') as f, open('logs/gameplay/'+self.log_file, 'a') as f_gameplay:
          f.write('Running episode: %d\n' % episode)
          f.write('Max Reward: %.3f\n' % max(rewards))
          # write the local score of each player to the log file 
    return rewards
  
  def get_weights(n_players, weight_type): 
    np.random.seed(0)
    n_decks = 5
    weights_comp = [[1.0, 0.0, 0.0, 0.5, 0.5],[0.0, 1.0, 0.0, 0.5,0.5],[0.0, 0.0, 1.0, 0.5, 0.5]]
    weights_mix = [[0.666, 0.333, 0.333, 0.5, 0.5],[0.333, 0.666, 0.333, 0.5, 0.5],[0.333, 0.333, 0.666, 0.5, 0.5]]

    if weight_type == 'fullcoop':
      weights = [[1.0, 1.0, 1.0, 0.5, 0.5] for _ in range(n_players)]
    elif weight_type == 'fullcomp':
      weights = weights_comp[:n_players]
    elif weight_type == 'mixcoop':
      weights = weights_mix[:n_players]
    else:
      raise ValueError("Invalid weight_type. Must be 'fullcoop', 'fullcomp', or 'mixcoop'.")
    return weights

if __name__ == "__main__":
  players = 3
  num_episodes = 1
  agent_class = 'LLMAgent'
  max_memory = 6
  context_level = 0
  weighting = True
  model = "Kimi-K2.5"
  logprobs = True 
  weight_type = 'fullcoop' # fullcoop, mixcoop, fullcomp

  flags = {'players': players, 'num_episodes': num_episodes, 'agent_class': agent_class, 'max_memory': max_memory, 'context_level': context_level, 'weighting': weighting, 'model':model, 'logprobs': logprobs, 'weight_type': weight_type}
  options, arguments = getopt.getopt(sys.argv[1:], '',
                                     ['players=',
                                      'num_episodes=',
                                      'agent_class=',
                                      'max_memory=',
                                      'context_level=',
                                      'weighting=',
                                      'log_file=',
                                      'model=',
                                      'logprobs=',
                                      'weight_type='])
  if arguments:
    sys.exit('usage: single_agent_hanabi.py [options]\n'
             '--players       number of players in the game.\n'
             '--num_episodes  number of game episodes to run.\n'
             '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
    
  for flag, value in options:
      flag = flag[2:]
      if isinstance(flags[flag], bool):
          flags[flag] = value.lower() in ('true', '1')
      else:
          flags[flag] = type(flags[flag])(value)

  log_file = "single_agent_hanabi_{}players_{}episodes_{}model_{}context_{}weighting_{}type_{}maxmemory_{}_{}".format(
      flags['players'], flags['num_episodes'], flags['model'], flags['context_level'],
      flags['weighting'], flags['weight_type'], flags['max_memory'], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), os.getpid())
  flags['log_file'] = log_file

  if flags['weighting']:
    flags['deck_weights'] = Runner.get_weights(flags['players'], flags['weight_type'])
  else:
    flags['deck_weights'] = None

  for d in ['logs', 'logs/full_logs', 'logs/stats', 'logs/gameplay', 'logs/logprobs', 'logs/illegal_moves', 'logs/api_errors']:
    os.makedirs(d, exist_ok=True)
  with open('logs/stats/'+log_file+'.csv', 'a') as f:
    f.write('turn,player_no,action,life_tokens,information_tokens,firework_R,firework_Y,firework_G,firework_W,firework_B,reward\n')
  start = time.time()
  runner = Runner(flags)
  runner.run()
  end = time.time()
  print('Total time taken: {} seconds'.format(end - start))
