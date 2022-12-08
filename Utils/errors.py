from typing import *

class UnknownError(Exception):
    def __init__(self, message : str = "Unknown error encountered."):
        self.message = message

    def __str__(self) -> str:
        return f'{self.message}'

class WrongArgument(Exception):
    def __init__(self, argument : str, message : str = "Wrong selection for {}, please use python3 interface.py --help for correct usage."):
        self.argument = argument
        self.message = message

    def __str__(self) -> str:
        return f"{self.message}".format(self.argument)

class EmptyListEncountered(Exception):
    def __init__(self, list_name : str = None, expected_content : str = None, message : str = "{} should contain {}, found empty."):
        self.message = message
        self.list_name = list_name
        self.expected_content = expected_content

    def __str__(self) -> str:
        return f"{self.message}".format(self.list_name, self.expected_content)

class NoneTypeEncountered(Exception):
    def __init__(self, correct_type : object, message : str ="NoneType Encountered where there shuould be {}"):
        self.message = message
        self.correct_type = correct_type

    def __str__(self) -> str:
        return f"{self.message}".format(self.correct_type)

class NotAnImage(Exception):
    def __init__(self, path : str = None, message : str = "{} doesn't correspond to any image."):
        self.path = path
        self.message = message

    def __str__(self) -> str:
        return f'{self.message}'.format(self.path)

class IncorrectDimensions(Exception):
    def __init__(self, height : int, width : int, expected_height : int = 512, expected_width : int = 512, message = "Incorrect dimensions found in current image. Excepted {}, found {}"):
        self.expected_height = expected_height
        self.expected_width = expected_width
        self.message = message
        self.height = height
        self.width = width

    def __str__(self) -> str:
        return f"{self.message}".format((self.expected_height, self.expected_width), (self.height, self.width))