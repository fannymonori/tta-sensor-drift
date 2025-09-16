import numpy as np
from collections import OrderedDict

class ExperienceReplayMemory:
    replay_memory_X = None
    replay_memory_Y = None
    mem_size = None

    def __init__(self, mem_size):
        print("Initialize class-balanced ER class")

        self.mem_size = mem_size

    def addSample(self, X, Y):
        if self.replay_memory_X is None:
            self.replay_memory_X = X
            self.replay_memory_Y = Y
        else:
            self.replay_memory_X = np.concatenate((X, self.replay_memory_X), axis=0)
            self.replay_memory_Y = np.concatenate((Y, self.replay_memory_Y), axis=0)

        self.replay_memory_X = self.replay_memory_X[:self.mem_size]
        self.replay_memory_Y = self.replay_memory_Y[:self.mem_size]

        return self.replay_memory_X, self.replay_memory_Y

    def getMemoryAsArray(self):
        return self.replay_memory_X, self.replay_memory_Y

    def initMemory(self, memory_X, memory_Y):
        self.replay_memory_X = memory_X
        self.replay_memory_Y = memory_Y


class ClassBalancedExperienceReplayMemory:
    C_replay_memory = None
    mem_size = None

    def __init__(self, mem_size):
        print("Initialize class-balanced ER class")

        self.C_replay_memory = OrderedDict()
        self.mem_size = mem_size

    def addSample(self, X, Y):
        #print("ADD SAMPLE")
        if Y not in self.C_replay_memory:
            self.C_replay_memory[Y] = X
        else:
            c_rp_X = self.C_replay_memory[Y]
            rp_end = min(self.mem_size, c_rp_X.shape[0]) - 1
            self.C_replay_memory[Y] = np.concatenate((X, c_rp_X[:rp_end]), axis=0)

    def getMemoryAsArray(self):
        batch_X = None
        batch_Y = None
        for key, values in self.C_replay_memory.items():
            label = np.asarray([[key]])
            label = np.reshape(np.tile(label, values.shape[0]), (values.shape[0], label.shape[1]))
            if batch_X is None:
                batch_X = values
                batch_Y = label
            else:
                batch_X = np.concatenate((batch_X, values), axis=0)
                batch_Y = np.concatenate((batch_Y, label), axis=0)

        return batch_X, batch_Y

    def initMemory(self, memory):
        self.C_replay_memory = memory
