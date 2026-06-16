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
"""Simple Agent."""

from autogen import ConversableAgent, LLMConfig
from hanabi_learning_environment.env import Agent
from dotenv import load_dotenv 
import os
import ast
import re
import time
import datetime
import numpy as np
load_dotenv()
_API_VERSION_MAP = {
  'DeepSeek-R1': os.getenv('DEEPSEEKR1_API_VERSION'),
  'Phi-4': os.getenv('PHI4_API_VERSION'),
  'o4-mini': os.getenv('O4MINI_API_VERSION'),
  'Kimi-K2.5': os.getenv('KIMI_API_VERSION'),
}
api_key = os.getenv('AZURE_API_KEY')
api_env = os.getenv('AZURE_API_ENV')

class LLMAgent(Agent):
  """Agent that calls LLM."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the agent."""
    self.config = config
    self.memory = config.get('shared_memory', []) # This is a list that is shared across all agents, which can be used to store information that all agents can access. This is useful for multi-agent coordination and communication. For example, agents can write to this shared memory to share their intentions, observations, or plans with other agents. This allows for more complex and coordinated behavior among multiple agents.  
    self.max_memory=config.get('max_memory', 0)
    self.context_level=config.get('context_level', 0) # This is a dictionary that can be used to store any additional context or information that the agent may need to access. This can include things like the current state of the environment, the agent's goals, or any other relevant information that the agent may need to make decisions. The context is specific to each agent and is not shared across agents, unlike the memory which is shared.
    self.weighting = config.get('weighting', False) # Whether to include deck weighting information in the prompt to the LLM. This can help the LLM make more informed decisions by providing it with additional context about the current state of the game and which fireworks piles are more likely to be completed soon based on the remaining cards in the deck. If weighting is True, the prompt will include information about the deck weighting for the player's personal score and how it should influence their decision making.
    self.deck_weights = config.get('deck_weights', None) # This is a list of lists, where each inner list contains the weights for the 5 fireworks piles for a particular player. The weights represent the importance of each fireworks pile for the player's personal score, with higher weights indicating that completing that pile would contribute more to the player's score. This information can be used by the LLM to prioritize certain actions that would help complete higher weighted piles, while still considering the overall game state and the need to maintain information and life tokens.
    self.log_file = config.get('log_file', None) # This is a string that specifies the file path where the agent's actions and observations will be logged. This can be useful for debugging, analysis, and improving the agent's performance over time. The log file can contain information such as the current game state, the actions taken by the agent, the reasoning behind those actions (if context_level > 0), and any other relevant information that can help understand the agent's behavior and decision-making process.
    self.record_logprobs = config.get('logprobs', False)
    logprobs_config = {"logprobs": True, "top_logprobs": 5} if self.record_logprobs else {}
    llm_config = LLMConfig({"api_type":"azure", #free tokens
                    #"model": "DeepSeek-V3.2",
                    "model": config.get('model'),
                    "base_url": api_env,
                    "api_key": api_key,
                    "api_version": _API_VERSION_MAP.get(config.get('model')),
                    "price": [0.0, 0.0],  # [prompt, completion] price per 1k tokens
                    **logprobs_config
}
)
    self.player_agent = ConversableAgent(name="hanabi_player", 
                                         llm_config=llm_config, 
                                         human_input_mode="NEVER",
                                         system_message=f"""
You are a player in a {config.get('players')}-player cooperative game of Hanabi. Your team scores points by playing cards to build five fireworks (R, Y, G, W, B), each requiring cards played in order from rank 0 to rank 4.

IMPORTANT RULES:
- You CANNOT see your own cards, but you CAN see your teammates' cards.
{ self.rank_explain(False)}
- A card is playable if its rank equals the current fireworks value for its color (e.g. if fireworks R=2, you need rank 2 to play next on red).
- Information tokens are spent to give hints and recovered by discarding. Discarding when tokens are full would be wasteful.
- Life tokens are lost when an unplayable card is played (e.g. if you play a blue rank 3 card when fireworks B=4). You only have 3 life tokens.
- Score is determined by adding together the current fireworks values (e.g. if fireworks are R=2, Y=1, G=0, W=3, B=1, score is 2+1+0+3+1=7). The maximum score is 25.

{self.reading_your_observation(False)}

{self.strategic_considerations(False)}
""")
    # Extract max info tokens or set default to 8.
    self.max_information_tokens = config.get('information_tokens', 8)

  @staticmethod
  def playable_card(card, fireworks): 
    """A card is playable if it can be placed on the fireworks pile."""
    return card['rank'] == fireworks[card['color']] # Return True if card rank is equivalent to the number of cards in the fireworks pile for that color, meaning it can be played.
  
  def _format_observation(self, obs):
    fireworks = obs['fireworks']
    
    my_knowledge = []
    for i, hint in enumerate(obs['card_knowledge'][0]):
        my_knowledge.append(f"Card {i}: known color={hint['color']}, known rank={hint['rank']}") # (i.e. card value {hint['rank']+1 if hint['rank'] is not None else '?'})")
    
    teammate_hands = []
    for player_offset in range(1, obs['num_players']):
        teammate_hands.append(f"Hand of player at target_offset={player_offset}:")
        for i, card in enumerate(obs['observed_hands'][player_offset]):
            playable_now = card['color'] and card['rank'] == fireworks[card['color']]
            teammate_hands.append(f"  Card {i}: {card['color']}{card['rank']} {'*** PLAYABLE NOW ***' if playable_now else ''}")
            #teammate_hands.append(f"  Card {i}: {card['color']}{card['rank']}")
    return f"""     
Firework deck status: R={fireworks['R']}, Y={fireworks['Y']}, G={fireworks['G']}, W={fireworks['W']}, B={fireworks['B']}. This is a summary of the current fireworks piles, where each value represents the NEXT RANK NEEDED FOR THAT COLOR. For example, if fireworks B=2, that means the next card needed to play on the blue pile is rank 2.
Life tokens: {obs['life_tokens']} | Info tokens: {obs['information_tokens']}
Your hand (what you know): {my_knowledge}
{self.belief_state(obs, include=False)}
{self.format_all_beliefs(obs, include=False)}
Teammates' hands: {teammate_hands}
Discard pile: {obs['discard_pile']}
Legal moves: {obs['legal_moves']}

{self.last_round(obs, True)}
"""
  def belief_state(self, obs, include):
    if not include:
        return ""
    colors = ['R', 'Y', 'G', 'W', 'B']
    lines = ["Based on hints you've received, here's what you know about your hand:"]
    for i, card_hint in enumerate(obs['pyhanabi'].card_knowledge()[0]):
        possible_colors = [c for j, c in enumerate(colors) if card_hint.color_plausible(j)]
        possible_values = [r + 1 for r in range(5) if card_hint.rank_plausible(r)]
        lines.append(f"  Card {i}: possible colors={possible_colors}, possible values={possible_values}")
    return "\n".join(lines)
  def reading_your_observation(self, include):
    if include:
      return """READING YOUR OBSERVATION:
- observed_hands[0] is YOUR hand - your cards are hidden (shown as None/-1).
- observed_hands[1], [2], etc. are your teammates' hands - you can see these fully.
- card_knowledge[0] tells you what hints YOU have received about your own cards (color and/or rank).
- card_knowledge[1], [2], etc. tells you what hints teammates have received.
- legal_moves lists every action currently available to you - you MUST choose from this list."""
    else:
      return ""
  def strategic_considerations(self, include):
    if include:
      return """STRATEGY PRIORITIES:
1. If card_knowledge tells you a card in your hand is definitely playable (its known color and rank match what is needed on the fireworks), PLAY it. 
2. If you can see a teammate has a card that is immediately playable on the fireworks and they have not been hinted about it, give them a REVEAL_COLOR or REVEAL_RANK hint.
3. If information tokens are low (<=2), prefer DISCARD over giving hints. Prioritise discarding cards that are less likely to be needed soon (e.g. high rank cards of colors that are far from being completed on the fireworks).
4. Do NOT discard blindly every turn - this wastes the information your teammates gave you and risks discarding cards that may be needed soon. 
5. Consider why a player might be hinting certain information to you - they may be trying to signal that a card is playable or that you should discard it. Use the pattern of hints and the game state to infer their intentions.
6. Only play a card if you are reasonably certain it is playable - avoid risky plays that could lose life tokens.
7. When you are on one life token, only play cards that you are 100% certain are playable.
8. Before giving hints, check your memory to see if you or a previous player has already hinted the same information to the same players - if so, giving the same hint again is redundant and wastes information tokens. 
"""
    else:
      return ""
  def print_belief_state(self, include):
    if include:
      return "First, state what you currently believe your hidden cards are, with your confidence for each"
    else:
      return ""
  def format_all_beliefs(self, obs, include):
      if not include:
          return ""
      colors = ['R', 'Y', 'G', 'W', 'B']
      pyhanabi_obs = obs['pyhanabi']
      all_knowledge = pyhanabi_obs.card_knowledge()
      lines = ["What each teammate currently knows about their own hand:"]
      for player_offset, player_knowledge in enumerate(all_knowledge):
          if player_offset == 0:
              continue
          lines.append(f"  Player at target_offset={player_offset}:")
          for i, card_hint in enumerate(player_knowledge):
              possible_colors = [c for j, c in enumerate(colors) if card_hint.color_plausible(j)]
              possible_values = [r + 1 for r in range(5) if card_hint.rank_plausible(r)]
              color_str = possible_colors if len(possible_colors) < 5 else "any"
              value_str = possible_values if len(possible_values) < 5 else "any"
              lines.append(f"    Card {i}: colors={color_str}, values={value_str}")
      lines.append("Use this information to infer what your teammates know and how they might interpret hints you give them.")
      return "\n".join(lines) 
  def rank_explain(self, include): 
    if include: 
      return """- Ranks are 0-indexed: rank 0 = card "1", rank 1 = card "2", ..., rank 4 = card "5"."""
    else:
      return ""
  def get_few_shot_examples(self, few_shot):
    """Return a string of few-shot examples to include in the prompt."""
    if few_shot == True: 
      return """Here are some examples of good actions in different situations:
  Example 1 
  Fireworks: R1 Y0 G0 W0 B0 | Info tokens: 4 | Life tokens: 3
  My card_knowledge: Card 0: color=R, rank=1 | Teammates' hands: ...

  Reasoning:
  - card_knowledge[0] tells me Card 0 is Red, rank 1.
  - Fireworks R=1 means the next needed Red card is rank 1.
  - Card 0 is therefore definitely playable.
  - I have enough life tokens to play safely.
  Action: {'action_type': 'PLAY', 'card_index': 0}

  Example 2 
  Fireworks: R0 Y0 G0 W0 B0 | Info tokens: 7 | Life tokens: 3
  My card_knowledge: all unknown
  Player at target_offset=1 hand: Card 0: R1 *** PLAYABLE NOW ***, Card 1: G3, Card 2: B2, Card 3: Y4, Card 4: W2
  Memory: no previous hints given to this player

  Reasoning:
  - I cannot play — I have no card_knowledge about my own cards.
  - Player at offset=1 has R1 at index 0 which is playable (fireworks R=0 needs rank 0).
  - Checking memory — no colour or rank hints have been given to this player yet.
  - I have 7 info tokens so hinting is not wasteful.
  - Hinting Red will identify the playable card without revealing unnecessary information about the others.
  Action: {'action_type': 'REVEAL_COLOR', 'color': 'R', 'target_offset': 1}

  Example 3 
  Fireworks: R1 Y0 G0 W0 B0 | Info tokens: 5 | Life tokens: 3
  Player at target_offset=1 hand: Card 0: R2, Card 1: G3, Card 2: Y1 *** PLAYABLE NOW ***
  Memory: [player 0 REVEAL_COLOR Y target_offset=1, player 1 DISCARD, player 2 REVEAL_RANK 0 target_offset=1]

  Reasoning:
  - Player at offset=1 has Y1 at index 2 which is playable (fireworks Y=0 needs rank 0).
  - Checking memory — player 0 already gave a Yellow hint to this player two moves ago.
  - Giving Yellow again wastes an info token on information they already have.
  - Player at offset=2 has no playable cards visible.
  - I have no certain plays from my own card_knowledge.
  - Info tokens are 5, not critically low.
  - Best action is to discard my least-known card to recover a token rather than waste one.
  Action: {'action_type': 'DISCARD', 'card_index': 4}

  Example 4 
  Fireworks: R2 Y1 G3 W1 B2 | Info tokens: 1 | Life tokens: 2
  My card_knowledge: Card 0: color=None rank=None, Card 1: color=B rank=None, Card 2: color=None rank=4
  Memory: no hints about my cards recently

  Reasoning:
  - Info tokens at 1 — I should not hint, that would leave 0 tokens.
  - card_knowledge shows Card 2 is rank 4 (value 5). Fireworks are all below 4, so it cannot be played yet and is low priority to keep.
  - Card 1 is Blue — fireworks B=2 so I would need rank 2. I don't know the rank so I cannot safely play it.
  - Card 0 is completely unknown — risky to play with only 2 life tokens.
  - Safest discard is Card 2 (rank 4 is unlikely to be needed soon) to recover an info token.
  Action: {'action_type': 'DISCARD', 'card_index': 2}

  Example 5
  Fireworks: R2 Y2 G1 W0 B3 | Info tokens: 3 | Life tokens: 1
  My card_knowledge: Card 0: color=G rank=1, Card 1: color=None rank=None
  Memory: ...

  Reasoning:
  - Only 1 life token remaining — a wrong play ends the game immediately.
  - Card 0 is Green rank 1. Fireworks G=1 needs rank 1. This is a certain safe play.
  - Card 1 is completely unknown — never play an unknown card on 1 life token.
  - Playing Card 0 is the only action I am 100% certain about.
  Action: {'action_type': 'PLAY', 'card_index': 0}"""
    else:
      return ""
  def get_history(self, max_memory): 
    if max_memory > 0 and len(self.memory) > 0:
      return f"Here are the previous {max_memory} moves made in the game: {self.memory[-max_memory:]}. Use this history to inform your understanding of the other players' strategies and how they might interpret your actions."
    else:
      return ""
  def last_round(self, obs, include):
    if include and obs['deck_size'] == 0:
      # if deck is empty 
      return "This is the last round of the game since the deck is empty. Think carefully how to maximize the score in this final turn."
    else:
      return ""
  def get_prompt_string(self, turn, obs): 
    if self.weighting == True: 
      weight_string = "The deck weighting for you is as follows: R:" + str(self.deck_weights[turn][0]) + ", Y:" + str(self.deck_weights[turn][1]) + ", G:" + str(self.deck_weights[turn][2]) + ", W:" + str(self.deck_weights[turn][3]) + ", B:" + str(self.deck_weights[turn][4]) + ". A good score in higher weighted decks is better for your personal success than a good score in lower weighted decks. However, a poor global deck will negatively impact your score, so you should still consider the global deck state in your decision making."
    else: 
      weight_string = ""
    
    return f"""You are player {turn}. 
    {weight_string}
    The current state of the game is as follows: 
    {obs}.

    {self.get_history(max_memory=self.max_memory)}

    Choose your next action. This must be one of the following:
1. PLAY: You need to specify the index of the card in your hand that you want
2. DISCARD: You need to specify the index of the card in your hand that you want to discard
3. REVEAL_COLOR: You need to specify the target player and the color you want
4. REVEAL_RANK: You need to specify the target player and the rank you want to reveal
    Only output the action in the following format:
    For PLAY and DISCARD: {{'action_type': 'PLAY', 'card_index': 0}} or {{'action_type': 'DISCARD', 'card_index': 3}}
    For REVEAL_COLOR: {{'action_type': 'REVEAL_COLOR', 'color': 'R', 'target_offset': 1, 'reasoning': <reasoning>, 'context': <context>}} where target_offset refers to the player shown as "Player at target_offset=1" in your observation.
    For REVEAL_RANK: {{'action_type': 'REVEAL_RANK', 'rank': 3, 'target_offset': 2, 'reasoning': <reasoning>, 'context': <context>}} where target_offset refers to the player shown as "Player at target_offset=2" in your observation.
    When asked for <reasoning>: provide one sentence explaining your intention behind giving this hint. For example, if you give a REVEAL_RANK hint to player at offset=2, it might be because there is only one deck that requires a rank 2 card, and you are trying to communicate to the other player this card is playable. Therefore your reasoning might be "I want to signal to player at offset=2that they have a playable card in their hand, which is the only card that can be played on the fireworks at the moment, so they know it is safe to play this card on their next turn."
    When asked for <context>: provide a sentence on your motivation for picking this deck specifically. For example, if each player has a playable card but one of the players has a playable card for a deck that is higher weighted for you, you might want to hint that card to increase the chances of that deck being completed, which would increase your personal score. In this case your context might be "I want to increase the chances of the red fireworks being completed soon, because red has the highest deck weighting for my personal score among currently playable colors."
    {self.get_few_shot_examples(False)}
    
    {self.print_belief_state(True)}
    Before choosing your action, reason through:
1. Can I see any immediately playable cards in my teammate's hand?
2. Does my card_knowledge tell me any of my cards are definitely playable?
3. Are info tokens low enough that I should discard instead of hint?
Then output ONLY the action dict.
    """
  
  # def strip_think_block(self, response): #strip any thinking from DeepSeek responses
  #   end = response.find('</think>')
  #   if end != -1:
  #       return response[end + len('</think>'):].strip()
  #   return response

  def act(self, observation):
    """Act based on an observation."""
    if observation['current_player_offset'] != 0: # Only act if it's our turn.
      return None
    
    self.player_agent.clear_history()
    
    turn = observation['current_player'] # Get the current player index
    obs = self._format_observation(observation) # Format the observation into a string for the LLM
    prompt_string = self.get_prompt_string(turn, obs) # Get the prompt string based on the formatted observation. This can include instructions, strategy hints, or any other relevant information to guide the LLM in making a decision. The prompt should be designed to help the LLM understand the current game state and what actions are available, as well as any strategic considerations it should take into account when choosing an action.
    # Get action from LLM
    legal=False
    retries = 0
    max_retries = 10
    MAX_API_RETRIES = 3
    RETRY_WAIT = 60
    while legal is False and retries < max_retries: # Loop until a legal action is chosen
      messages = [
        {"role": "system", "content": self.player_agent.system_message},
        {"role": "user", "content": prompt_string}
      ]
      for api_attempt in range(MAX_API_RETRIES):
        try:
          raw_response = self.player_agent.client.create(messages=messages)
          break
        except Exception as e:
          is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
          with open("logs/api_errors/{}".format(self.log_file), "a") as err_f:
            err_f.write(f"{datetime.datetime.now()} | attempt={api_attempt+1} | rate_limit={is_rate_limit} | error={e}\n")
          if api_attempt < MAX_API_RETRIES - 1:
            time.sleep(RETRY_WAIT)
          else:
            response = None
            raw_response = None
      if raw_response is None:
        retries += 1
        prompt_string += "No response received. Please output a dictionary in the correct format. Try again."
        continue
      response = raw_response.choices[0].message.content
      with open("logs/{}".format(self.log_file), "a") as f:
        f.write(f"Player {turn} response: {response}\n")
      if self.record_logprobs and raw_response.choices[0].logprobs:
        import json
        logprobs_data = [
          {"token": t.token, "logprob": t.logprob,
           "top": [{"token": tl.token, "logprob": tl.logprob} for tl in (t.top_logprobs or [])]}
          for t in raw_response.choices[0].logprobs.content
        ]
        with open("logs/logprobs/{}".format(self.log_file), "a") as f:
          f.write(json.dumps({"turn": turn, "retry": retries, "logprobs": logprobs_data}) + "\n")
      retries += 1
      if response is None:
        prompt_string += "No response received. Please output a dictionary in the correct format. Try again."
        continue
      if isinstance(response, dict): # If LLM returns dictionary directly (e.g. with Phi-4), extract content
        response = response.get('content', '')
      matches = re.findall(r"\{[^{}]*['\"]action_type['\"][^{}]*\}", response, re.DOTALL)
      if not matches:
        prompt_string += "Your response was not in the correct format. Please output a dictionary in the correct format. Try again."
        continue
      try:
        action_dict = ast.literal_eval(matches[-1])
      except (ValueError, SyntaxError):
        prompt_string += "Your response could not be parsed. Please output a valid Python dictionary. Try again."
        continue
      # Check if action is legal
      action_dict_no_reasoning = action_dict.copy()
      action_dict_no_reasoning.pop('reasoning', None) # Remove reasoning from action dict if it exists, as legal_moves will not include reasoning
      action_dict_no_reasoning_or_context = action_dict_no_reasoning.copy()
      action_dict_no_reasoning_or_context.pop('context', None) # Remove context from action dict if it exists, as legal_moves will not include context
      # print the deck in action_dict
      legal = action_dict_no_reasoning_or_context in observation['legal_moves']
      if not legal:
        with open("logs/{}".format(self.log_file), "a") as f, open("logs/gameplay/{}".format(self.log_file), "a") as f_gameplay:
          f.write(f"Illegal move: {action_dict_no_reasoning_or_context}\n")
          f_gameplay.write(f"Illegal move: {action_dict_no_reasoning_or_context}\n")
        prompt_string += "Illegal move. Check list of legal moves and try again."
        print("Illegal move. Check list of legal moves and try again.")

    if not legal:
      action_dict = observation['legal_moves'][0]  # fallback to first legal move if LLM fails repeatedly
      action_dict_no_reasoning = action_dict.copy()
      action_dict_no_reasoning.pop('reasoning', None)
      action_dict_no_reasoning_or_context = action_dict_no_reasoning.copy()
      action_dict_no_reasoning_or_context.pop('context', None)
      with open("logs/{}".format(self.log_file), "a") as f, open("logs/gameplay/{}".format(self.log_file), "a") as f_gameplay:
        f.write(f"LLM failed to provide a legal move after {max_retries} attempts. Defaulting to first legal move: {action_dict}\n")
        f_gameplay.write(f"LLM failed to provide a legal move after {max_retries} attempts. Defaulting to first legal move: {action_dict}\n")

    # trim action_dict according to context level
    if self.context_level == 0:
      action_dict.pop('reasoning', None)
      action_dict.pop('context', None)
    elif self.context_level == 1:
      action_dict.pop('context', None)
    elif self.context_level == 2:
      pass # keep everything in the action dict
    else:
      raise ValueError("Invalid context level. Must be 0, 1, or 2.")
    self.memory.append({'player': turn, 'action': action_dict}) # Add the current action to memory with appropriate context level
    return action_dict # should be in format {'action_type': 'PLAY', 'card_index': 0} or {'action_type': 'REVEAL_COLOR', 'color': R, 'target_offset': 1} etc. depending on the action type