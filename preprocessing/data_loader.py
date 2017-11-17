from __future__ import print_function, division, absolute_import

import sys
import time

import cv2
import numpy as np
import torch as th
import multiprocessing
from torch.autograd import Variable

from .utils import preprocessInput
from .preprocess import IMAGE_WIDTH, IMAGE_HEIGHT, N_CHANNELS

if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue


def imageWorker(image_queue, output_queue, exit_event):
    while not exit_event.is_set():
        idx, image_path = image_queue.get()

        if idx is None:
            image_queue.put((None, None))
            break

        im = cv2.imread(image_path)
        im = cv2.resize(im, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        im = preprocessInput(im.astype(np.float32), mode="image_net")
        output_queue.put((idx, im))
        del im
    print("Worker exiting...")


# TODO: prefetch data + cache using dict
class BaxterImageLoader(object):
    """
    :param minibatches: [[int]] list of list of int (observations ids)
    :param images_path: (numpy 1D array of str)
    """

    def __init__(self, minibatches, images_path, same_actions,
                 dissimilar, test_batch_size=512, n_workers=8,
                 n_queues=4,
                 is_training=True, mode="image_net"):
        super(BaxterImageLoader, self).__init__()
        self.minibatches = minibatches[:]
        self.n_minibatches = len(minibatches)
        self.n_samples = len(images_path)
        print("{} samples".format(self.n_samples))
        # Save minibatches original order
        self.original_minibatches = minibatches[:]
        self.images_path = images_path
        self.mode = mode
        self.dissimilar = dissimilar[:]
        self.same_actions = same_actions[:]
        self.current_idx = 0
        self.is_training = is_training
        self.test_batch_size = test_batch_size

        # Multiprocessing
        self.n_workers = n_workers
        n_queues = n_queues if n_queues > 0 else n_workers // 2
        self.n_queues = min(n_queues, n_workers)
        self.image_queues = [multiprocessing.Queue() for _ in range(n_queues)]
        self.output_queues = [multiprocessing.Queue() for _ in range(n_queues)]
        self.exit_event = multiprocessing.Event()
        self.n_sent, self.n_received = 0, 0
        self.shutdown = False

        if self.n_workers <= 0:
            raise ValueError("n_workers <= 0")
        print("{} workers {} queues".format(self.n_workers, self.n_queues))

        self.workers = []
        for i in range(self.n_workers):
            w = multiprocessing.Process(target=imageWorker,
                                        args=(self.image_queues[i % self.n_queues],
                                              self.output_queues[i % self.n_queues],
                                              self.exit_event)
                                        )
            w.daemon = True  # ensure that the worker exits on process exit
            w.start()
            self.workers.append(w)

    def trainMode(self):
        self.is_training = True
        self.minibatches = self.original_minibatches[:]

    def testMode(self):
        self.is_training = False
        self.minibatches = []
        for i in range(self.n_samples // self.test_batch_size + 1):
            start_idx = i * self.test_batch_size
            end_idx = min(self.n_samples, (i + 1) * self.test_batch_size)
            self.minibatches.append(np.arange(start_idx, end_idx))

    def reset(self):
        self.current_idx = 0

    def resetAndShuffle(self):
        self.current_idx = 0
        self.shuffleMinitbatchesOrder()

    def shuffleMinitbatchesOrder(self):
        np.random.shuffle(self.minibatches)

    def resetQueues(self):
        """
        Reset sent and received count
        and empty queues
        """
        self.n_sent, self.n_received = 0, 0
        # Clear queues
        for q in self.image_queues + self.output_queues:
            while not q.empty():
                idx, _ = q.get()
                if idx is not None:
                    print("Warning, queue not empty")

    def __iter__(self):
        return self

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        if self.current_idx < len(self.minibatches):
            if self.current_idx == 0:
                self.total_time = 0
            start_time = time.time()
            print('{}/{}'.format(self.current_idx, len(self.minibatches)))
            # Alias to improve readability
            i = self.current_idx
            obs_indices = self.minibatches[i]

            batch_size = len(obs_indices)

            if self.is_training:
                diss = self.dissimilar[i][np.random.permutation(self.dissimilar[i].shape[0])]
                same = self.same_actions[i][np.random.permutation(self.same_actions[i].shape[0])]
                # Convert to torch tensor
                diss, same = th.from_numpy(diss), th.from_numpy(same)

                # Retrieve observations
                # Define a dict to modify it in the for loop
                obs_dict = {'obs': None, 'next_obs': None}
                indices_list = [obs_indices, obs_indices + 1]
            else:
                obs_dict = {'obs': None}
                indices_list = [obs_indices]

            for indices, key in zip(indices_list, obs_dict.keys()):
                obs = np.zeros((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, N_CHANNELS), dtype=np.float32)
                # Reset queues and received count
                self.resetQueues()
                self._putImages(indices)

                while self.n_received < self.n_sent:
                    for q in self.output_queues:
                        try:
                            j, im = q.get_nowait()
                        except queue.Empty:
                            continue
                        obs[j, :, :, :] = im
                        self.n_received += 1

                obs = np.transpose(obs, (0, 3, 2, 1))
                obs_dict[key] = Variable(th.from_numpy(obs))
                # Free memory
                del obs

            delta = time.time() - start_time
            self.total_time += delta
            self.current_idx += 1
            if self.is_training:
                return obs_dict['obs'], obs_dict['next_obs'], diss, same
            else:
                return obs_dict['obs']

        else:
            print("Preprocessing took {:.2f}s".format(self.total_time))
            raise StopIteration

    next = __next__  # Python 2 compatibility

    def _putImages(self, indices):
        """
        Put images to be processed in the queues
        :param indices: (int)
        """
        for j, idx in enumerate(indices):
            image_path = 'data/{}'.format(self.images_path[idx])
            self.image_queues[j % self.n_queues].put((j, image_path))
            self.n_sent += 1

    def _shutdown_workers(self):
        """
        Method used to shutdown processes
        It set the exit_event and release the queues
        by sending `None`
        """
        if not self.shutdown:
            self.shutdown = True
            self.exit_event.set()
            for i in range(len(self.workers)):
                self.image_queues[i % self.n_queues].put((None, None))
            for w in self.workers:
                w.join()

    def __del__(self):
        """
        Shutdown processes when the object is deleted
        """
        if self.n_workers > 0:
            self._shutdown_workers()
