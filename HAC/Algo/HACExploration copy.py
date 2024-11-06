import os
import numpy
from gym import spaces
from stable_baselines3 import HAC 
from stable_baselines3.hac.policies import MlpPolicy  
from rl_interaction.algorithms.ExplorationAlgorithm import ExplorationAlgorithm
from rl_interaction.utils.TimerCallback import TimerCallback
from rl_interaction.utils.wrapper import TimeFeatureWrapper


class HACAlgorithm(ExplorationAlgorithm):

    @staticmethod
    def explore(app, emulator, appium, timesteps, timer, save_policy=False, app_name='', reload_policy=False,
                policy_dir='.', cycle=0, num_layers=2, train_freq=5, **kwargs):
        try:
            env = TimeFeatureWrapper(app)
        
            if reload_policy and (os.path.isfile(f'{policy_dir}{os.sep}{app_name}.zip')):
                temp_dim = env.action_space.high[0]
                env.action_space.high[0] = env.env.ACTION_SPACE
                print(f'Reloading Policy {app_name}.zip')
                model = HAC.load(f'{policy_dir}{os.sep}{app_name}', env)
                env.action_space.high[0] = temp_dim
            else:
                print('Starting training from zero')
                model = HAC(MlpPolicy, env, verbose=1, num_layers=num_layers, train_freq=train_freq)
            model.env.envs[0].check_activity()
            callback = TimerCallback(timer=timer, app=app)
            model.learn(total_timesteps=timesteps, callback=callback)
            if save_policy:
                print('Saving Policy...')
                model.action_space.high[0] = model.env.envs[0].ACTION_SPACE
                model.save(f'{policy_dir}{os.sep}{app_name}')
            return True
        except Exception as e:
            print(e)
            appium.restart_appium()
            if emulator is not None:
                emulator.restart_emulator()
            return False
