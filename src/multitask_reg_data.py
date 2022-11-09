import random

class MultiTaskLoader(object):

    def __init__(self, loaders, shuffle=True, sampling='roundrobin', n_batches=None, verbose=True):
        # loaders is a list contain different task loader
        self.loaders = loaders
        self.task2loader = {loader.task: loader for loader in self.loaders}
        self.shuffle = shuffle
        self.sampling = sampling
        self.n_batches = n_batches
        self.verbose = verbose
        self.epoch_tasks = None
        self.task2len = {loader.task: len(loader) for loader in self.loaders}
        if self.verbose:
            print('Task2len:', self.task2len)

        self.set_epoch(0)

    def __iter__(self):
        self.task2iter = {loader.task: iter(loader) for loader in self.loaders}
        return self

    # 在主函数的时候，每一个epoch一开始，都会有个set_epoch，所以把multitask的epoch_tasks在这里写好是一个很明智的idea;
    def set_epoch(self, epoch):
        for loader in self.loaders:
            loader.sampler.set_epoch(epoch)

        if self.sampling == 'roundrobin':
            epoch_tasks = []
            for task, loader in self.task2loader.items():
                n_batch = len(loader)
                epoch_tasks.extend([task]*n_batch)
        elif self.sampling == 'balanced':
            if self.n_batches is None:
                n_batches = sum(self.task2len.values()) // len(self.loaders)
            else:
                n_batches = self.n_batches
            if self.verbose:
                print('# batches:', n_batches)
            epoch_tasks = []
            for task, loader in self.task2loader.items():
                epoch_tasks.extend([task]*n_batches)

        if self.shuffle:
            random.Random(epoch).shuffle(epoch_tasks)
        self.epoch_tasks = epoch_tasks
        if self.verbose:
            print('# epoch_tasks:', len(self.epoch_tasks))

    def __next__(self):
        if len(self.epoch_tasks) > 0:
            task = self.epoch_tasks.pop()
            loader_iter = self.task2iter[task]
            return next(loader_iter)
        else:
            raise StopIteration

    def __len__(self):
        return len(self.epoch_tasks)
