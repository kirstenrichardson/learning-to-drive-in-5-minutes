import os

import numpy as np
import cv2


class Recorder(object):
    """
    Class to record images for offline VAE training
    and expert data for pretraining.

    :param env: (Gym env)
    :param folder: (str)
    :param start_recording: (bool)
    :param verbose: (int)
    """

    def __init__(self, env, folder='logs/recorded_data/', start_recording=False, verbose=0):
        super(Recorder, self).__init__()
        self.env = env
        self.is_recording = start_recording
        self.folder = folder
        self.current_idx = 0
        self.verbose = verbose
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.actions = []
        self.observations = []
        self.rewards = []
        self.episode_returns = []
        self.episode_starts = []
        self.reward_sum = 0.0
        self.archive_path = os.path.join(folder, 'expert_dataset.npz')
        self.traj_data = None

        # Create folder if needed
        os.makedirs(folder, exist_ok=True)

        if os.path.isfile(self.archive_path):
            # Open previous experiment
            self.traj_data = np.load(self.archive_path)
            self.current_idx = len(self.traj_data['obs'])

        if verbose > 0:
            print("Recorder current idx: {}".format(self.current_idx))

    def reset(self):
        obs = self.env.reset()
        self.reward_sum = 0.0
        # TODO: handle reset properly
        # if self.is_recording:
        #     self.episode_starts.append(True)
        #     self.save_image()
        return obs

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def store_infos(self, action, reward, done=False):
        if self.is_recording:
            self.save_image()
            self.actions.append(action)
            self.rewards.append(reward)
            self.episode_starts.append(done)
            self.reward_sum += reward
            if done:
                self.episode_returns.append(self.reward_sum)

    def save_image(self):
        image = self.env.render(mode='rgb_array')
        # Convert RGB to BGR
        image = image[:, :, ::-1]
        image_path = os.path.join(self.folder, "{}.jpg".format(self.current_idx))
        cv2.imwrite(image_path, image)
        self.observations.append(image_path)
        if self.verbose >= 2:
            print("Saving", "{}".format(image_path))
        self.current_idx += 1

    def set_recording_status(self, is_recording):
        self.is_recording = is_recording
        if self.verbose > 0:
            print("Setting recording to {}".format(is_recording))

    def toggle_recording(self):
        self.set_recording_status(not self.is_recording)
        # TODO: handle that properly
        if self.is_recording and len(self.observations) > 0:
            self.episode_starts[-1] = True

        if len(self.observations) > 0 and not self.is_recording:
            self._save_archive()

    def exit_scene(self):
        self.env.exit_scene()

    def _save_archive(self):
        observations = np.array(self.observations)
        actions = np.concatenate(self.actions).reshape((-1,) + self.env.action_space.shape)
        rewards = np.array(self.rewards)
        self.episode_starts[0] = True
        episode_starts = np.array(self.episode_starts)
        episode_returns = np.array(self.episode_returns)

        assert len(observations) == len(actions)
        assert len(observations) == len(episode_starts)

        numpy_dict = {
            'actions': actions,
            'obs': observations,
            'rewards': rewards,
            'episode_returns': episode_returns,
            'episode_starts': episode_starts
        }
        # Merge previous expert data
        if self.traj_data is not None:
            for key in numpy_dict.keys():
                numpy_dict[key] = np.concatenate((numpy_dict[key], self.traj_data[key]))
        if os.path.isfile(self.archive_path):
            os.remove(self.archive_path)
        # Save archive
        np.savez(self.archive_path, **numpy_dict)
