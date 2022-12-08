import os
import json
import glob

from typing import *

from tqdm import tqdm

from Utils.errors import *

class DatasetScanner():
    """
    This class logs informations relative to the absolute position of each image into a log file.
    For each of the three datasets we are supposed to use there is a specific function suitable for handling each one of them.
    """
    def __init__(self, master_folder_path=None, log_file_path=None, validation_file_path=None, valid_list=None, ignore_list=None, dataset=None, verbose=0):
        self.master_folder_path=master_folder_path
        self.log_file_path=log_file_path
        self.validation_file_path=validation_file_path
        self.valid_list=valid_list
        self.ignore_list=ignore_list
        self.dataset=dataset
        self.verbose=verbose

    def scan_master_folder(self) -> List[Union[str, None]]:
        """
        Function used to scan the source folder for each datasets. Returns a list of folders -> activations.
        """
        paths = os.listdir(self.master_folder_path)
        try:
            if len(paths) == 0:
                raise EmptyListEncountered(list_name="master_folder_paths", expected_content="dataset absolute paths")
        except EmptyListEncountered as ele:
            print(ele)
        finally:
            return paths

    def sort_tmp_items(self, items : List[str]) -> List[str]:
        """
        Function used to sort subitems (pre, post) for Colomba Dataset according to activation's date. 
        Used to discrimintate between pre and post.
        PARAMETERS:
        items -> list of subitems per activations (pre, post)
        """
        return sorted(items, key = lambda x : x.split("/")[-1].split("_")[1].split(".")[0])

    def trim_master_folder(self) -> Tuple[Union[List[str], None], Union[List[str], None]]:
        """
        Function used to use a validated subset of Effis. Probably needs refactoring.
        """
        # to be called ONLY on complete effis with provided validated.json
        try:
            with open(f"{self.validation_file_path}") as val_file:
                self.valid_list = json.load(val_file)
                if len(self.valid_list) == 0:
                    raise EmptyListEncountered(list_name="validation_list", expected_content="paths")
                for path in self.valid_list:
                    if os.isdir(path):
                        pass
                    else:
                        self.ignore_list.append(path)
        except OSError as ose:
            print(ose)
        except EmptyListEncountered as ele:
            print(ele)
        finally:
            return self.valid_list, self.ignore_list

    def handle_full_effis(self):
        """
        THIS WHOLE FUNCTION WILL PROBABLY BE REFACTORED.
        """
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

    def handle_sub_effis(self):
        pass

    def handle_colombaset(self) -> bool:
        """
        Function used to handle Colomba dataset. Scans each folder and stores data concering the relative position
        of pre, post and mask.
        """
        completed = False
        try:
            with open(f"{self.log_file_path}", "a") as log_file:
                print(f"Logging info into log file {self.log_file_path}...")
                for idx in tqdm(range(len(os.listdir(self.master_folder_path)))):
                    path = os.listdir(self.master_folder_path)[idx]
                    pre = None
                    post = None
                    mask = None
                    tmp = None
                    sorted_item_list = self.sort_tmp_items(items = glob.glob(f"{self.master_folder_path}/{path}/*.tiff"))
                    if len(sorted_item_list) == 0:
                        raise EmptyListEncountered(list_name = "sorted_item_list")
                    for item in sorted_item_list:
                        if item.split("/")[-1].startswith("EMS"):
                            mask = item
                        else:
                            if tmp == None:
                                tmp = item.split("/")[-1].split("_")[1].split(".")[0]
                            if item.split("/")[-1].split("_")[1].split(".")[0] > tmp:
                                post = item
                                pre = "sentinel2_" + tmp + ".tiff"
                    complete_pre = os.path.join(self.master_folder_path, path, pre)
                    if (pre is None) or (post is None) or (mask is None):
                        raise NoneTypeEncountered(correct_type=str)
                    log_file.write(f"{path},{complete_pre},{post},{mask}\n")
                    completed = True         
        except OSError as ose:
            print(ose)
        except EmptyListEncountered as ele:
            print(ele)
        except NoneTypeEncountered as nte:
            print(nte)
        finally:
            return completed

    def log_header_to_file(self, header="act_id") -> bool:
        """
        Function used to log header into log file.
        """
        try:
            with open(f"{self.log_file_path}", "w") as log_file:
                if self.verbose==1:
                    print(f"Writing header into log file: {self.log_file_path}...")
                log_file.write(header + "\n")
        except OSError as ose:
            print(ose)
        finally:
            return True

    def log_to_file(self) -> bool:
        """
        Simple wrapper function.
        """
        if self.dataset == "full-effis":
            self.handle_full_effis()
        elif self.dataset == "sub-effis":
            self.handle_sub_effis()
        elif self.dataset == "colombaset":
            completed = self.handle_colombaset()
        return completed