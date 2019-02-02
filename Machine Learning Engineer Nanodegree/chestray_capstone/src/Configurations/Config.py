# Configuration schema for config.json file

import json
import os 
from definitions import ROOT_DIR

configFilePath = os.path.join(ROOT_DIR, "Configurations", "config.json")

# Singleton config file for loading data
class Config:
    __instance = None

    @staticmethod 
    def GetConfig():
        if Config.__instance == None:
            with open(configFilePath) as file:
                configFileContent = json.load(file)
                Config(**configFileContent)
        return Config.__instance

    def __init__(self, DataFolderPath, *args, **kwargs):
        if Config.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            self.DataFolderPath = DataFolderPath
            Config.__instance = self