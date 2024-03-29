import os
import json
import glob

from typing import *

from tqdm import tqdm

from Utils.errors import *

class DatasetScanner():
    """This class logs informations relative to the absolute position of each image into a log file.
    For each of the three datasets we are supposed to use there is a specific function suitable for handling each one of them.

    Args:
        - master_folder_path : str -> absolute path of data folder to scan
        - log_file_path : str -> relative fp to logfile
        - validation_file_path : str -> global validation file from which to extract self.valid_list
        - valid_list : List[int] -> list of activations suitable for training
        - ignore_list : List[int] -> list of activations to be ignored 
        - dataset : str -> dataset type
        - verbose : int -> verbosity level for logging purposes
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
        pass

    def handle_sub_effis(self):
        """Directly creates master_dict_effis.json with structure: {"processing_info":{<act_id>:{"tile_info_post":[], "tile_info_mask":[]}}}
        master_folder_path = /mnt/data1/adsp_data/effis
        """
        master_dict_train_effis = {}
        master_dict_val_effis = {}
        # popolo i dizionari con le relative chiavi (train/val)
        for train_act_id in glob.glob(f"{self.master_folder_path}/ann/train/*.tif"):
            train_activation_id = train_act_id.split("/")[-1].split("_")[0]
            master_dict_train_effis[f"{train_activation_id}"] = {"tile_info_post":list(), "tile_info_mask":list()}
        for val_act_id in glob.glob(f"{self.master_folder_path}/ann/val/*.tif"):
            val_activation_id = val_act_id.split("/")[-1].split("_")[0]
            master_dict_val_effis[f"{val_activation_id}"] = {"tile_info_post":list(), "tile_info_mask":list()}
        # qui costruisco le informazioni di post e mask per ogni attivazione
        for subdir in glob.glob(f"{self.master_folder_path}/*"):
            subdir = subdir.split("/")[-1]
            if subdir == "ann":
                for kind in glob.glob(f"{self.master_folder_path}/{subdir}/*"):
                    kind = kind.split("/")[-1]
                    if kind == "train":
                        print("Loading data for effis training masks")
                        for act_item in tqdm(os.listdir(f"{self.master_folder_path}/{subdir}/{kind}/")):
                            act_id = act_item.split("_")[0]
                            info = f"{self.master_folder_path}/{subdir}/{kind}/{act_item}"
                            master_dict_train_effis[act_id]["tile_info_mask"].append(info)
                    elif kind == "val":
                        print("Loading data for effis validation masks")
                        for act_item in tqdm(os.listdir(f"{self.master_folder_path}/{subdir}/{kind}/")):
                            act_id = act_item.split("_")[0]
                            info = f"{self.master_folder_path}/{subdir}/{kind}/{act_item}"
                            master_dict_val_effis[act_id]["tile_info_mask"].append(info)
            elif subdir == "img":
                for kind in glob.glob(f"{self.master_folder_path}/{subdir}/*"):
                    kind = kind.split("/")[-1]
                    if kind == "train":
                        print("Loading data for effis training tiles")
                        for act_item in tqdm(os.listdir(f"{self.master_folder_path}/{subdir}/{kind}/")):
                            act_id = act_item.split("_")[0]
                            info = f"{self.master_folder_path}/{subdir}/{kind}/{act_item}"
                            master_dict_train_effis[act_id]["tile_info_post"].append(info)
                    elif kind == "val":
                        print("Loading data for effis validation tiles")
                        for act_item in tqdm(os.listdir(f"{self.master_folder_path}/{subdir}/{kind}/")):
                            act_id = act_item.split("_")[0]
                            info = f"{self.master_folder_path}/{subdir}/{kind}/{act_item}"
                            master_dict_val_effis[act_id]["tile_info_post"].append(info)

        with open("Log/master_dict_train_effis.json", "w") as train_out_md:
            json.dump(master_dict_train_effis, train_out_md, indent=4)

        with open("Log/master_dict_val_effis.json", "w") as val_out_md:
            json.dump(master_dict_val_effis, val_out_md, indent=4)

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
        completed = False
        if self.dataset == "full-effis":
            self.handle_full_effis()
            completed = True
        elif self.dataset == "sub-effis":
            self.handle_sub_effis()
            completed = True
        elif self.dataset == "colombaset":
            completed = self.handle_colombaset()
            completed = True
        return completed