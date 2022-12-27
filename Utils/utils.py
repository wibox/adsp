import os
import sys
import glob
import json
import traceback

from typing import *

class Configurator():
    def __init__(self, filepath : str = None, filename : str = None):
        self.filepath = filepath
        self.filename = filename
        self.config = self._load_config()

    def _load_config(self) -> Tuple[bool, Dict[str, str]]:
        completed = False
        try:
            with open(os.path.join(self.filepath, self.filename), "r") as f:
                self.config = json.load(f)
        except Exception as e:
            print(e.format_exc())
            sys.exit()
        finally:
            return completed, self.config

    def get_config(self) -> dict:
        return self.config