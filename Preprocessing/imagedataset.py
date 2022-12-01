import json

from typing import *

from torch.utils.data.dataset import Dataset

class ImageDataset(Dataset):
    def __init__(self, processed_master_folder_path=None, master_dict_path=None, tile_width=512, tile_height=512, transform=None, use_pre=False, verbose=0):
        self.processed_master_folder_path=processed_master_folder_path
        self.master_dict_path=master_dict_path
        self.tile_width=tile_width
        self.tile_heigth=tile_height
        self.transform=transform
        self.use_pre=use_pre
        self.verbose=verbose

        self.master_dict = json.loads(self.master_dict_path)

    def __len__(self) -> int:
        try:
            with open(f"{self.master_dict_path}", "r") as jsonfp:
                info_dict = json.load(jsonfp)
        except:
            print(f"Error dealing with log json file at: {self.master_dict_path}")
        finally:
            return len(info_dict)

    def __getitem__(self, idx):
        """
        Questa funzione prende in input un indice relativo all'immagine e alla maschera che vogliamo caricare sull'algoritmo da trainare
        e, tramite l'indice, carica la relativa immagine e maschera e le ritorna in un dizionario.
        """
        local_dict = dict()

        local_dict['post_image_tile'] = self.master_dict[f"act_id_{idx}"]["tile_info_post"]
        local_dict['mask_tile'] = self.master_dict[f"act_id_{idx}"]["tile_info_mask"]
        if self.use_pre:
            local_dict['pre_image_tile'] = self.master_dict[f"act_id_{idx}"]["tile_info_pre"]

        return local_dict