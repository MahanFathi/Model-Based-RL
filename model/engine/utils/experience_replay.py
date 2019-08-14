import torch
import torch.utils.data
import random


class StateQueue(object):

    def __init__(self, size):
        self.size = size
        self.queue = []

    def __getitem__(self, idx):
        idx = idx % len(self.queue)
        return self.queue[idx]

    def __len__(self):
        return len(self.queue) or self.size

    def add(self, state):
        self.queue.append(state)
        self.queue = self.queue[-self.size:]

    def add_batch(self, states):
        self.queue.extend([state for state in states])
        self.queue = self.queue[-self.size:]

    def get_item(self):
        return random.choice(self.queue)


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(sampler, batch_size):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False
    )
    return batch_sampler


def batch_collator(batch):
    return torch.stack(batch, dim=0)


def make_data_loader(dataset, batch_sampler, collator):
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )
    return data_loader
