# import classes to module level
from .simpletrainer import SimpleTrainer
from .evolutiontrainer import EvolutionTrainer

# import the most popular helper functions
from .helpers import optimistic_restore, create_save_var_dict, get_gpu_count, average_gradients, combine_loss_dicts
from .easing import *

