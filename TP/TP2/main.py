from queue import Queue
from threading import Thread

from antcolony import AntColony

class Worker(Thread):
    """ Thread executing tasks from a given tasks queue """
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except Exception as e:
                # An exception happened in this thread
                print(e)
            finally:
                # Mark this task as done, whether an exception happened or not
                self.tasks.task_done()


class ThreadPool:
    """ Pool of threads consuming tasks from a queue """
    def __init__(self, num_threads):
        self.tasks = Queue(num_threads)
        for _ in range(num_threads):
            Worker(self.tasks)

    def add_task(self, func, *args, **kargs):
        """ Add a task to the queue """
        self.tasks.put((func, args, kargs))

    def map(self, func, args_list):
        """ Add a list of tasks to the queue """
        for args in args_list:
            self.add_task(func, args)

    def wait_completion(self):
        """ Wait for completion of all the tasks in the queue """
        self.tasks.join()

DATASET_NAMES=["graph1.txt", "graph2.txt", "graph3.txt"]
# DATASET_NAMES=["test.txt"]

def main():
    ANTS_NUM = [50, 100, 200, 1000]
    IT_NUM = [100]
    SIGMA = [0.1, 0.3, 0.5, 0.7, 0.9]
    ALPHA = [0.1, 0.5, 1.0, 2.0]
    BETA = [0.1, 0.5, 1.0, 2.0]
    K_ANTS = [0, 1, 5, 10, 15]

    #ANTS_NUM = [50]
    #IT_NUM = [100]
    #SIGMA = [0.9]
    #ALPHA = [1.0, 2.0]
    #BETA = [1.0, 2.0]
    #K_ANTS = [5]

    pool = ThreadPool(4)
    for ants_num in ANTS_NUM:
        for it_num in IT_NUM:
            for sigma in SIGMA:
                for alpha in ALPHA:
                    for beta in BETA:
                        for k_ants in K_ANTS:
                            for dname in DATASET_NAMES:
                                aco = AntColony(dname,
                                                ants_num=ants_num,
                                                it_num=it_num,
                                                sigma=sigma,
                                                alpha=alpha,
                                                beta=beta,
                                                k_ants=k_ants)
                                pool.add_task(aco.flux_colony)
                pool.wait_completion()
    
if __name__ == '__main__':
    main()
