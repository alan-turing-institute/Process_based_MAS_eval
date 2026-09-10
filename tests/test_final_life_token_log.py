import importlib
import sys
import types


def _load_runner_module(monkeypatch):
    fake_env = types.ModuleType("hanabi_learning_environment.env")
    fake_env.make = lambda *_args, **_kwargs: None
    fake_llm_agent = types.ModuleType("hanabi_learning_environment.agents.llm_agent")
    fake_llm_agent.LLMAgent = object
    monkeypatch.setitem(sys.modules, "hanabi_learning_environment.env", fake_env)
    monkeypatch.setitem(
        sys.modules, "hanabi_learning_environment.agents.llm_agent", fake_llm_agent
    )
    sys.modules.pop("single_agent_hanabi", None)
    return importlib.import_module("single_agent_hanabi")


class _Agent:
    def __init__(self, _config):
        pass

    def act(self, observation):
        if observation["current_player"] == 0:
            return {"action_type": "PLAY"}
        return None


class _Environment:
    state = "terminal"

    def __init__(self, before, after):
        self.before = before
        self.after = after

    def reset(self):
        return {"player_observations": [self.before]}

    def step(self, _action, _log_file):
        return {"player_observations": [self.after]}, 0, True, {}


def _observation(life_tokens):
    return {
        "current_player": 0,
        "life_tokens": life_tokens,
        "information_tokens": 5,
        "fireworks": {"R": 1, "Y": 2, "G": 3, "W": 4, "B": 5},
    }


def test_final_life_token_is_recorded(tmp_path, monkeypatch):
    module = _load_runner_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    for log_dir in ("stats", "full_logs", "gameplay"):
        (tmp_path / "logs" / log_dir).mkdir(parents=True)

    runner = module.Runner.__new__(module.Runner)
    runner.flags = {"num_episodes": 1, "players": 1, "weighting": False}
    runner.agent_config = {"shared_memory": []}
    runner.agent_class = _Agent
    runner.environment = _Environment(_observation(1), _observation(0))
    runner.log_file = "game"

    runner.run()

    rows = (tmp_path / "logs" / "stats" / "game.csv").read_text().splitlines()
    assert len(rows) == 2
    assert rows[0].split(",")[3] == "1"
    assert rows[1].split(",")[2:4] == ["GAME_OVER", "0"]
