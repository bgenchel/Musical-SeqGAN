"""
taken from https://github.com/ZiJianZhao/SeqGAN-PyTorch
"""
import copy
import numpy as np

class Rollout(object):
    """
    Monte-Carlo Rollout Policy
    """
    def __init__(self, model, update_rate):
        self.model = model
        self.update_rate = update_rate

    def get_reward(self, data, rollout_num, discriminator):
        """
        args:
            data: input data (batch_size, seq_len)
            rollout_num: roll-out number
            discriminator: discriminator model
        """
        rewards = []
        batch_size, seq_len = data.size()
        for i in range(rollout_num):
            for l in range(1, seq_len + 1):
                data_subseqs = data[:, :l]
                samples = self.model.sample(batch_size, seq_len, data_subseqs)
                pred = discriminator(samples)
                pred = pred.cpu().data[:, 1].numpy() # why cpu?
                if i == 0:
                    rewards.append(pred)
                else:
                    # rewards are summed over all rollouts?
                    rewards[l-1] += pred

        rewards = np.transpose(np.array(rewards)) / float(rollout_num)
        return rewards

    def update_params(self):
        """
        seems like this is transferring parameter values from the original_model
        to my_model, based on the update rate. Doesn't look like this has anything
        to do with the rewards being calculated though ... not sure what the purpose
        of this is
        """
        for name, param in self.model.named_parameters():
            if name.startswith('embed'):
                continue
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * param.data
