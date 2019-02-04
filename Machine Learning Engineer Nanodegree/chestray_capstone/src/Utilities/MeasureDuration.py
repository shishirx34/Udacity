import time
from colorama import Fore
from colorama import Style

class MeasureDuration:
    def __init__(self):
        self.start = None
        self.end = None
 
    def __enter__(self):
        self.start = time.time()
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print ("Total time taken: %s!" % self.duration())

    def duration(self):
        return str((self.end - self.start)) + ' seconds'