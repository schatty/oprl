from collections import defaultdict

from oprl.environment import EnvProtocol, DMControlEnv, SafetyGym
from oprl.environment.gymnasium import Gymnasium



ENV_MAPPER: defaultdict[str, set[str]] = defaultdict(set)
ENV_MAPPER["dm_control"] = set(
    [
        "acrobot-swingup",
        "ball_in_cup-catch",
        "cartpole-balance",
        "cartpole-swingup",
        "cheetah-run",
        "finger-spin",
        "finger-turn_easy",
        "finger-turn_hard",
        "fish-upright",
        "fish-swim",
        "hopper-stand",
        "hopper-hop",
        "humanoid-stand",
        "humanoid-walk",
        "humanoid-run",
        "pendulum-swingup",
        "point_mass-easy",
        "reacher-easy",
        "reacher-hard",
        "swimmer-swimmer6",
        "swimmer-swimmer15",
        "walker-stand",
        "walker-walk",
        "walker-run",
    ]
)


ENV_MAPPER["gymnasium"] = set(
    [
        "Ant-v4",
        "Hopper-v4",
        "HalfCheetah-v4",
        "HumanoidStandup-v4",
        "Humanoid-v4",
        "InvertedDoublePendulum-v4",
        "InvertedPendulum-v4",
        "Pusher-v4",
        "Reacher-v4",
        "Swimmer-v4",
        "Walker2d-v4",
    ]
)


ENV_MAPPER["safety_gymnasium"] = set(
    [
        "SafetyPointGoal1-v0",
        "SafetyPointGoal2-v0",
        "SafetyPointButton1-v0",
        "SafetyPointButton2-v0",
        "SafetyPointPush1-v0",
        "SafetyPointPush2-v0",
        "SafetyPointCircle1-v0",
        "SafetyPointCircle2-v0",
        "SafetyCarGoal1-v0",
        "SafetyCarGoal2-v0",
        "SafetyCarButton1-v0",
        "SafetyCarButton2-v0",
        "SafetyCarPush1-v0",
        "SafetyCarPush2-v0",
        "SafetyCarCircle1-v0",
        "SafetyCarCircle2-v0",
        "SafetyAntGoal1-v0",
        "SafetyAntGoal2-v0",
        "SafetyAntButton1-v0",
        "SafetyAntButton2-v0",
        "SafetyAntPush1-v0",
        "SafetyAntPush2-v0",
        "SafetyAntCircle1-v0",
        "SafetyAntCircle2-v0",
        "SafetyDoggoGoal1-v0",
        "SafetyDoggoGoal2-v0",
        "SafetyDoggoButton1-v0",
        "SafetyDoggoButton2-v0",
        "SafetyDoggoPush1-v0",
        "SafetyDoggoPush2-v0",
        "SafetyDoggoCircle1-v0",
        "SafetyDoggoCircle2-v0",
    ]
)


def make_env(name: str, seed: int) -> EnvProtocol:
    for env_type, env_set in ENV_MAPPER.items():
        if name in env_set:
            if env_type == "dm_control":
                return DMControlEnv(name, seed=seed)
            elif env_type == "safety_gymnasium":
                return SafetyGym(name, seed=seed)
            elif env_type == "gymnasium":
                return Gymnasium(name, seed=seed)
    raise ValueError(f"Unsupported environment: {name}")
