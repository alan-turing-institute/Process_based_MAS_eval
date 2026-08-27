This code is adapted from Google DeepMind's Hanabi implementation (available at: https://github.com/google-deepmind/hanabi-learning-environment). 

## How to run: 
Gameplay is started through the single_agent_hanabi.py file. Once this file is ran from the terminal it will play $n_{episodes}$ games of Hanabi to completion. This contains the runner for the game, as well as the flags that control the parameters of the gameplay (e.g. players, context level, etc...). 
The parameters of the game can either be set through the terminal: 

```bash 
python single_agent_hanabi.py --model="Kimi-K2.5" --players=3 --num_episodes=1
```
or within script. 

The list of controllable variables are as follows: 
|Flag | Data Type | Description|
|-|-|-|
|players |int| number of players in each game |
|num_episodes |int| the number of episodes played while the script is running (NOTE: these will all be recorded in the same log files, including the same *stats* file which tracks different statistics of interested over the course of a game)|
|agent_class |str| which agent class to implement as the players. This currently only includes LLMAgent, but additional classes can be added by the user. |
|max_memory |int| the maximum previous number of moves passed to the next player |
|context_level |int| level of information a player can include when giving a hint |
|weighting |bool| whether or not weighting is included |
|weight_type |str| weighting type to be used|
|logprobs |bool| whether logprobs should be recorded or not |


This setup allows for one model to be used for all players, which is set up in llm_agent.py. This uses the *autogen* package to connect to an Azure-deployed model, where the API key and environment are pulled from the .env file. When a model requires a version be specified, this can be specified in the .env file, and this is then mapped in llm_agent.py to that model name. 

## Log Files
The game progress for an execution of single_agent_hanabi.py is recorded in several ways in the ```hanabi-learning-environment/logs/``` directory: 
|folder name | file content |
|-|-|
|api_errors | any failed attempts to connect to the agent API, recording date and time, number of attempts, rate limit, and error message |
| full_logs | the complete output from each agent followed by a JSON-formatted summary of their move each turn, and printouts of the current game "board" |
| gameplay |  a JSON-formatted summary of an agent's move each turn, and printouts of the current game "board" |
| logprobs | if logprobs are enabled, records the logprobs of agent responses each turn |
| stats | a turn-by-turn record of various game statistics (turn number, current player, action made, life tokens, info. tokens, current deck values, current total score) |
## Notable Changes
Several changes were made from the DeepMind code to adapt this set-up for our purposes. These are as follows: 
### Memory system
As part of the prompt given to an agent on their turn, the prompt returns a "memory" of the previous $max$ _ $memory$ turns, where $max$ _ $memory$ is a variable that is set at the through flags. This can provide the agent with information such as what hints have already been recently given, or if they've received multiple hints on the same card. 

```bash 
python single_agent_hanabi.py --max_memory=6
```

### Context levels 
We use the context levels to control how much additional information can be passed between players during hints. The base game only allows hints within very specific constraints, but with context levels we allow for the agents to include reasoning and motivation for giving particular hints. The context level is controlled through the flags at the initialisation of the game:

```bash 
python single_agent_hanabi.py --context_level=2
```
The levels are defined as follows: 

| Level | Included |
|-------|----------|
| L0    | Hint only |
| L1    | Hint and reasoning |
| L2    | Hint, reasoning, and context |

### Deck weightings
We include the option to give a weighting to the decks. This gives each player a specific weighting that gives some decks higher value than others, which gives each player a personal score that is determined by that weighting. This is then fed back to them during prompting. This can be included by running:
```bash 
python single_agent_hanabi.py --weighting=True --weight_type=<WEIGHT_TYPE>
```
The current version of the script includes three options for #WEIGHT_TYPE#: "fullcoop", "mixcoop", and "fullcomp". These are for 3-player games, but these can be modified from the *get_weights* function in single_agent_hanabi.py. 


