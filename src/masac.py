from ray.tune.registry import register_env
from ray.rllib.algorithms import SACConfig
from ray.rllib.policy.policy import PolicySpec
from multi_agent_env import UltraDenseHetNetEnvironment
from ray.rllib.models import ModelCatalog
from green_cnn_policy import GreenCNNPolicy
from pprint import pprint

def env_creator(env_config):
    return UltraDenseHetNetEnvironment(config=env_config)

register_env("sleep_switch_env", env_creator)

ModelCatalog.register_custom_model("green_cnn_policy", GreenCNNPolicy)

config = (
    SACConfig()
    .environment("sleep_switch_env")
    .env_runners(
        num_env_runners=4,
        num_envs_per_env_runner=2
    )
    .multi_agent(
        policies={"shared_policy": PolicySpec(config={
                    "model": {
                        "custom_model": "green_cnn_policy",
                    }
                })},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
    )
    .evaluation(
        evaluation_num_env_runners=1,
        evaluation_interval=1,
        evaluation_duration=10,
        evaluation_force_reset_envs_before_iteration=True
    )
)
algo = config.build()

for _ in range(5):
    algo.train()

pprint(algo.evaluate())
algo.stop()