import os
import json
from typing import *

class DatasetScanner():
    """
    Legge dalla master folder le cartelle all'interno delle quali ci sono pre, post e mask.
    Successivamente genera un csv con tutte le informazioni relative alla singola attivazione.
    In questo modo evitiamo di mantenere in memoria a runtime 520 giga di informazioni sul path assoluto
    E processiamo riga per riga il csv risultate. (generato da log_to_file()).
    """
    def __init__(self, master_folder_path=None, log_file_path=None, validation_file_path=None, valid_list=None, ignore_list=None, verbose=0):
        self.master_folder_path=master_folder_path
        self.log_file_path=log_file_path
        self.validation_file_path=validation_file_path
        self.verbose=verbose
        self.valid_list=valid_list
        self.ignore_list=ignore_list

    def scan_master_folder(self) -> List[str]:
        return os.listdir(self.master_folder_path)

    def trim_master_folder(self) -> Tuple(List[str], List[str]):
        try:
            with open(f"{self.validation_file_path}") as val_file:
                self.valid_list = json.load(val_file)
                for path in self.valid_list:
                    if os.isdir(path):
                        pass
                    else:
                        self.ignore_list.append(path)
        except:
            print(f"Error reading validation file: {self.validation_file_path}")
        finally:
            return self.valid_list, self.ignore_list

    def log_header_to_file(self, header="act_id") -> bool:
        """
        csv_header = act_id
        """
        try:
            with open(f"{self.log_file_path}", "w") as log_file:
                if self.verbose==1:
                    print(f"Writing header into log file: {self.log_file_path}")
                log_file.write(header + "\n")
        except:
            print(f"Error writing header into log file: {self.log_file_path}")
        finally:
            return True

    def log_to_file(self) -> bool:
        try:
            if not self.valid_list:
                with open(f"{self.log_file_path}", "a") as log_file:
                    for path in self.scan_master_folder().filter(lambda folder : os.isdir(folder)):
                        if self.verbose==1:
                            print(f"Writing act_id record into log file: {self.log_file_path}")
                    log_file.write(f"{path}")
            else:
                with open(f"{self.log_file_path}", "a") as log_file:
                    for path in self.valid_list:
                        if self.verbose==1:
                            print(f"Writing act_id record into log file: {self.log_file_path}")
                    log_file.write(f"{path}")
        except:
            print(f"Error writing act_id info into log file: {self.log_file_path}")
        finally:
            return True

class DatasetFormatter():
    def __init__(self):
        pass

    def check_dimensions(self):
        pass

    def tiling(self):
        pass
