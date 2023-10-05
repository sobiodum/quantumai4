import ray, random, os 
ray.init(address="auto")
from ray import air, tune
from ray.tune import CLIReporter
from multi_agent import MultiAgent
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print

from ray.tune.schedulers import ASHAScheduler

from ray.tune.logger import pretty_print

env = MultiAgent()
def env_creator(env_config):
    return MultiAgent()  

register_env("MultiAgent", env_creator)

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    max_t=2000,
    grace_period=200,
    reduction_factor=3,
    brackets=1,
)

def create_policy_spec(agent_id):
    # print(f"Creating policy for {worker_id} with obs space {env.observation_space[worker_id]} and action space {env.action_space[worker_id]}")
    return PolicySpec(
        observation_space=env.observation_space[agent_id],
        action_space=env.action_space[agent_id],
        config={}
    )

controller_policy_spec = PolicySpec(
    observation_space=env.observation_space['controller'],
    action_space=env.action_space['controller'],
    config={}
)

policies = {
    "controller_policy": controller_policy_spec,
}

for agent_id in env.agents:
    policies[agent_id] = create_policy_spec(agent_id)

def policy_mapping_fn(agent_id, episode=None, agent=None, **kwargs):
    if agent_id == 'controller':
        # print(f"!!!!!! policy mapping manager: {agent_id}")
        return "controller_policy"
    elif agent_id in env.agents:
        return agent_id
    else:
        print("defaul policy triggered")
        return "default_policy"

param_space = {
     "env": "MultiAgent",
    "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "rollout_fragment_length": "auto",
        "framework": "tf2",
        "lr": tune.uniform(1e-5,1e-4),
        "gamma": tune.uniform(0.95, 0.9999),
        "lambda": tune.uniform(0.9,1.0),
        "entropy_coeff": tune.uniform(0.01,0.1),
        "vf_loss_coeff": tune.uniform(0.1,0.3),
        "num_workers": 7, 
        #Change for Debugging
        "log_level": "ERROR",
        "output": "logdir",
        "monitor": True,
}
@ray.remote
def tuning_func():
    analysis = tune.run(
        "A2C", 
        metric="episode_reward_mean", 
        num_samples=10,
        resume=False,
        mode="max",
        config=param_space, 
        local_dir="./temp",
        search_alg=None,
        scheduler=asha_scheduler,
        progress_reporter=CLIReporter(max_progress_rows=10,max_report_frequency=120),
        max_concurrent_trials=1,
        #checkpoint_config not checked yet
        checkpoint_config={
            "num_to_keep": 1,
            "checkpoint_score_attribute": "episode_reward_mean",
            "checkpoint_score_order": "max",
            "checkpoint_frequency": 10
        }
        )
    return analysis

analysis = ray.get(tuning_func.remote())
print("Best hyperparameters found were: ", pretty_print(analysis.best_config))