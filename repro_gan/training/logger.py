"""
Implementation of the Logger object for performing training logging and visualisation.
"""
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    Writes summaries and visualises training progress.
    
    Attributes:
        log_dir (str): The path to store logging information.
        num_steps (int): Total number of training iterations (epoch).
        dataset_size (int): The number of examples in the dataset.
        device (Device): Torch device object to send data to.
        flush_secs (int): Number of seconds before flushing summaries to disk.
        writers (dict): A dictionary of tensorboard writers with keys as metric names.
    """
    def __init__(self,
                 log_dir,
                 num_steps,
                 dataset_size,
                 device,
                 flush_secs=120,
                 **kwargs):
        self.log_dir = log_dir
        self.num_steps = num_steps
        self.dataset_size = dataset_size
        self.flush_secs = flush_secs
        self.device = device
        self.writers = {}

        # Create log directory if haven't already
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _build_writer(self, metric):
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'data', metric),
                               flush_secs=self.flush_secs)

        return writer

    def write_summaries(self, log_data, global_step):
        """
        Tasks appropriate writers to write the summaries in tensorboard. Creates additional
        writers for summary writing if there are new scalars to log in log_data.

        Args:
            log_data (MetricLog): Dict-like object to collect log data for TB writing.
            global_step (int): Global step variable for syncing logs.

        Returns:
            None
        """
        for metric, data in log_data.items():
            if metric not in self.writers:
                self.writers[metric] = self._build_writer(metric)

            # Write with a group name if it exists
            name = log_data.get_group_name(metric) or metric
            self.writers[metric].add_scalar(name,
                                            log_data[metric],
                                            global_step=global_step)

    def close_writers(self):
        """
        Closes all writers.
        """
        for metric in self.writers:
            self.writers[metric].close()

    def print_log(self, global_step, log_data, time_taken):
        """
        Formats the string to print to stdout based on training information.

        Args:
            log_data (MetricLog): Dict-like object to collect log data for TB writing.
            global_step (int): Global step variable for syncing logs.
            time_taken (float): Time taken for one training iteration.

        Returns:
            str: String to be printed to stdout.
        """
        # Basic information
        log_to_show = [
            "INFO: [Epoch: {:d}/{:d}]".format(global_step, self.num_steps)
        ]

        # Display GAN information as fed from user.
        GAN_info = [""]
        metrics = sorted(log_data.keys())

        for metric in metrics:
            GAN_info.append('{}: {}'.format(metric, log_data[metric]))

        # Add train step time information
        GAN_info.append("({:.4f} sec/idx)".format(time_taken))

        # Accumulate to log
        log_to_show.append("\n| ".join(GAN_info))

        # Finally print the output
        ret = " ".join(log_to_show)
        print(ret)

        return ret

    def _get_fixed_noise(self, nz, num_signals, output_dir=None):
        """
        Produce the fixed gaussian noise vectors used across all models
        for consistency.
        """
        if output_dir is None:
            output_dir = os.path.join(self.log_dir, 'viz')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir,
                                   'fixed_noise_nz_{}.pth'.format(nz))

        if os.path.exists(output_file):
            noise = torch.load(output_file)

        else:
            noise = torch.randn((num_signals, nz))
            torch.save(noise, output_file)

        return noise.to(self.device)

    def _get_fixed_labels(self, num_signals, num_classes):
        """
        Produces fixed class labels for generating fixed signals.
        """
        labels = np.array([i % num_classes for i in range(num_signals)])
        labels = torch.from_numpy(labels).to(self.device)

        return labels
