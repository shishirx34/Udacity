from colorama import Fore
from colorama import Style
import datetime

class MeasureDuration:
    def __init__(self):
        self.start = None
        self.end = None
 
    def __enter__(self):
        self.start = datetime.datetime.now()
        return self
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = datetime.datetime.now()
        print ("Total time taken: %s!" % self.duration())

    def duration(self):
        return str(self.end - self.start)