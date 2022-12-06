import os
import json
import glob

from typing import *

from tqdm import tqdm

from datetime import datetime

class DatasetScanner():
    """
    Legge dalla master folder le cartelle all'interno delle quali ci sono pre, post e mask.
    Successivamente genera un csv con tutte le informazioni relative alla singola attivazione.
    In questo modo evitiamo di mantenere in memoria a runtime 520 giga di informazioni sul path assoluto
    E processiamo riga per riga il csv risultate. (generato da log_to_file()).
    """
    def __init__(self, master_folder_path=None, log_file_path=None, validation_file_path=None, valid_list=None, ignore_list=None, dataset=None, verbose=0):
        self.master_folder_path=master_folder_path
        self.log_file_path=log_file_path
        self.validation_file_path=validation_file_path
        self.valid_list=valid_list
        self.ignore_list=ignore_list
        self.dataset=dataset
        self.verbose=verbose

    def scan_master_folder(self) -> List[str]:
        # this function works with all three datasets, no modifications needed
        return os.listdir(self.master_folder_path)

    def trim_master_folder(self) -> Tuple[List[str], List[str]]:
        # to be called ONLY on complete effis with provided validated.json
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
        csv_header = act_id -> da arricchire
        """
        try:
            with open(f"{self.log_file_path}", "w") as log_file:
                if self.verbose==1:
                    print(f"Writing header into log file: {self.log_file_path}...")
                log_file.write(header + "\n")
        except:
            print(f"Error writing header into log file: {self.log_file_path}")
        finally:
            return True

    def log_to_file(self) -> bool:
        if self.dataset == "full-effis":
            try:
                if not self.valid_list:
                    with open(f"{self.log_file_path}", "a") as log_file:
                        #self.scan_master_folder().
                        for path in list(filter(lambda folder : os.isdir(folder), self.scan_master_folder())):
                            if self.verbose==1:
                                print(f"Writing {path} record into log file: {self.log_file_path}")
                        log_file.write(f"{path}")
                else:
                    with open(f"{self.log_file_path}", "a") as log_file:
                        for path in self.valid_list:
                            if self.verbose==1:
                                print(f"Writing {path} record into log file: {self.log_file_path}")
                        log_file.write(f"{path}")
            except:
                print(f"Error writing act_id info into log file: {self.log_file_path}")
            finally:
                return True
        elif self.dataset == "colombaset":
            # logghiamo act_id, path_pre, path_post, path_mask
            try:
                with open(f"{self.log_file_path}", "a") as log_file:
                    print(f"Logging info into log file {self.log_file_path}...")
                    for idx in tqdm(range(len(os.listdir(self.master_folder_path)))):
                        path = os.listdir(self.master_folder_path)[idx]
                        pre = None
                        post = None
                        mask = None
                        tmp = None
                        sorted_item_list = sorted(glob.glob(f"{self.master_folder_path}/{path}/*.tiff"), key = lambda x : x.split("/")[-1].split("_")[1].split(".")[0])
                        for item in sorted_item_list:
                            if item.split("/")[-1].startswith("EMS"):
                                mask = item
                            else:
                                if tmp == None:
                                    tmp = item.split("/")[-1].split("_")[1].split(".")[0]
                                if item.split("/")[-1].split("_")[1].split(".")[0] > tmp:
                                    post = item
                                    pre = "sentinel2_" + tmp
                        complete_pre = os.path.join(self.master_folder_path, path, pre)
                        log_file.write(f"{path},{complete_pre},{post},{mask}\n")             
            except:
                print(f"Error writing data info into log file: {self.log_file_path}")
            finally:
                return True
        elif self.dataset == "sub-effis":
            pass