import time

class Timer():

    def __init__(self):
        self.reset()

    def reset(self):
        self.reference_time = time.time()

    def get_duration(self):
        return time.time() - self.reference_time
