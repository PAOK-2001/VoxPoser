import openai
from voxposer.arguments import get_config
from voxposer.interfaces import setup_LMP
from voxposer.visualizers import ValueMapVisualizer
from voxposer.envs.rlbench_env import VoxPoserRLBench
from voxposer.utils import set_lmp_objects

import numpy as np
from rlbench import tasks

config = get_config(config_path='src/voxposer/configs/rlbench_config.yaml')
# uncomment this if you'd like to change the language model (e.g., for faster speed or lower cost)
# for lmp_name, cfg in config['lmp_config']['lmps'].items():
#     cfg['model'] = 'gpt-3.5-turbo'
# initialize env and voxposer ui
task = str(input("Enter the task you want to run: "))
task = task.lower()

visualizer = ValueMapVisualizer(config['visualizer'])
env = VoxPoserRLBench(visualizer=visualizer)
lmps, lmp_env = setup_LMP(env, config, debug=False)
voxposer_ui = lmps['plan_ui']

task_dict = {
    "buttons": tasks.PushButton,
    "lamp": tasks.LampOff,
    "wine": tasks.OpenWineBottle,
    "rubbish": tasks.PutRubbishInBin,
    "umbrella": tasks.TakeUmbrellaOutOfUmbrellaStand,
    "scales": tasks.TakeOffWeighingScales,
    "grill": tasks.MeatOffGrill,
    "slide": tasks.SlideBlockToTarget,
    "saucepan": tasks.TakeLidOffSaucepan,
    "cups": tasks.StackCups,
    "slide block": tasks.SlideBlockToTarget,
    "door": tasks.OpenDoor,
}


env.load_task(task_dict[task])

descriptions, obs = env.reset()
set_lmp_objects(lmps, env.get_object_names())  # set the object names to be used by voxposer

instruction = np.random.choice(descriptions)
voxposer_ui("close gripper and move above the object")