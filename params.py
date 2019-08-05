import gym


class train_params:
    # Environment parameters
    ENV = 'Pendulum-v0'  # Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)
    RENDER = False  # Whether or not to display the environment on the screen during training
    RANDOM_SEED = 99999999  # Random seed for reproducability
    NUM_AGENTS = 4  # Number of distributed agents to run simultaneously

    # Create dummy environment to get all environment params
    dummy_env = gym.make(ENV)
    STATE_DIMS = dummy_env.observation_space.shape
    STATE_BOUND_LOW = dummy_env.observation_space.low
    STATE_BOUND_HIGH = dummy_env.observation_space.high
    ACTION_DIMS = dummy_env.action_space.shape
    ACTION_BOUND_LOW = dummy_env.action_space.low
    ACTION_BOUND_HIGH = dummy_env.action_space.high
    del dummy_env

    # Training parameters
    BATCH_SIZE = 256
    NUM_STEPS_TRAIN = 200  # Number of steps to train for
    MAX_EP_LENGTH = 1000  # Maximum number of steps per episode
    REPLAY_MEM_SIZE = 1000000  # Soft maximum capacity of replay memory
    REPLAY_MEM_REMOVE_STEP = 200  # Check replay memory every REPLAY_MEM_REMOVE_STEP training steps and remove samples over REPLAY_MEM_SIZE capacity
    PRIORITY_ALPHA = 0.6  # Controls the randomness vs prioritisation of the prioritised sampling (0.0 = Uniform sampling, 1.0 = Greedy prioritisation)
    PRIORITY_BETA_START = 0.4  # Starting value of beta - controls to what degree IS weights influence the gradient updates to correct for the bias introduced by priority sampling (0 - no correction, 1 - full correction)
    PRIORITY_BETA_END = 1.0  # Beta will be linearly annealed from its start value to this value throughout training
    PRIORITY_EPSILON = 0.00001  # Small value to be added to updated priorities to ensure no sample has a probability of 0 of being chosen
    NOISE_SCALE = 0.3  # Scaling to apply to Gaussian noise
    NOISE_DECAY = 0.9999  # Decay noise throughout training by scaling by noise_decay**training_step
    DISCOUNT_RATE = 0.99  # Discount rate (gamma) for future rewards
    N_STEP_RETURNS = 5  # Number of future steps to collect experiences for N-step returns
    UPDATE_AGENT_EP = 1  # Agent gets latest parameters from learner every update_agent_ep episodes

    # Network parameters
    CRITIC_LEARNING_RATE = 0.0001
    ACTOR_LEARNING_RATE = 0.0001
    CRITIC_L2_LAMBDA = 0.0  # Coefficient for L2 weight regularisation in critic - if 0, no regularisation is performed
    DENSE1_SIZE = 400  # Size of first hidden layer in networks
    DENSE2_SIZE = 300  # Size of second hidden layer in networks
    FINAL_LAYER_INIT = 0.003  # Initialise networks' final layer weights in range +/-final_layer_init
    NUM_ATOMS = 51  # Number of atoms in output layer of distributional critic
    V_MIN = -20.0  # Lower bound of critic value output distribution
    V_MAX = 0.0  # Upper bound of critic value output distribution (V_min and V_max should be chosen based on the range of normalised reward values in the chosen env)
    TAU = 0.001  # Parameter for soft target network updates
    USE_BATCH_NORM = False  # Whether or not to use batch normalisation in the networks

    # Files/Directories
    SAVE_CKPT_STEP = 10000  # Save checkpoint every save_ckpt_step training steps
    CKPT_DIR = './ckpts/' + ENV  # Directory for saving/loading checkpoints
    CKPT_FILE = None  # Checkpoint file to load and resume training from (if None, train from scratch)
    LOG_DIR = './logs/train/' + ENV  # Directory for saving Tensorboard logs (if None, do not save logs)


class test_params:
    # Environment parameters
    ENV = train_params.ENV  # Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)
    RENDER = False  # Whether or not to display the environment on the screen during testing
    RANDOM_SEED = 999999  # Random seed for reproducability

    # Testing parameters
    NUM_EPS_TEST = 100  # Number of episodes to test for
    MAX_EP_LENGTH = 1000  # Maximum number of steps per episode

    # Files/directories
    CKPT_DIR = './ckpts/' + ENV  # Directory for saving/loading checkpoints
    CKPT_FILE = None  # Checkpoint file to load and test (if None, load latest ckpt)
    RESULTS_DIR = './test_results'  # Directory for saving txt file of results (if None, do not save results)
    LOG_DIR = './logs/test/' + ENV  # Directory for saving Tensorboard logs (if None, do not save logs)


class play_params:
    # Environment parameters
    ENV = train_params.ENV  # Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)
    RANDOM_SEED = 999999  # Random seed for reproducability

    # Play parameters
    NUM_EPS_PLAY = 5  # Number of episodes to play for
    MAX_EP_LENGTH = 1000  # Maximum number of steps per episode

    # Files/directories
    CKPT_DIR = './ckpts/' + ENV  # Directory for saving/loading checkpoints
    CKPT_FILE = None  # Checkpoint file to load and run (if None, load latest ckpt)
    RECORD_DIR = './video'  # Directory to store recorded gif of gameplay (if None, do not record)
