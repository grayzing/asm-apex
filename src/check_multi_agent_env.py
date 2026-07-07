from ray.rllib.utils.pre_checks.env import check_multiagent_environments
from multi_agent_env import HeterogenousNetworkMultiAgentSleepEnv

# 1. Instantiate your environment
env = HeterogenousNetworkMultiAgentSleepEnv(config={})

# 2. Run the check
# This will raise a descriptive ValueError if your environment 
# doesn't follow the MultiAgentEnv API.
check_multiagent_environments(env)