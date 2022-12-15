import json
import os
import traceback

from typing import *

from torch.utils.data.dataset import Dataset

class ImageDataset(Dataset):
    def __init__(self, formatted_folder_path = None, log_folder=None, master_dict=None, transformations=None, use_pre=False, verbose=0):
        self.formatted_folder_path=formatted_folder_path
        self.log_folder=log_folder
        self.master_dict=master_dict
        self.transformations=transformations
        self.use_pre=use_pre
        self.verbose=verbose
        
        self.post_tiles = list()
        self.mask_tiles = list()
        self.activations = os.listdir(self.formatted_folder_path)

        try:
            loading_completed = False
            with open(os.path.join(self.log_folder, self.master_dict), "r") as md: #master_dict
                self.loaded_tile_data = json.load(md) # questo è il dizionario completo
            loading_completed = True
        except OSError as ose:
            print(ose)
        finally:
            return None

    def _log_info(self) -> bool:
        with open("Log/act_id_info.csv", "w") as log_prova:
            for idx in range(len(self.activations)):
                dir_act = self.activations[idx]
                key_act = list(self.loaded_tile_data['processing_info'][idx].keys())[0]
                log_prova.write(f"{dir_act},{key_act}\n")

    def _load_tiles(self) -> bool:
        """
        Questa funzione deve caricare nell'istanza della classe le liste di tutti i path di post e mask per ogni attivazione.
        self.post_tiles = [[tiles_post_act_1] + [tiles_post_act_2] + ...]
        self.mask_tiles = [[tiles_mask_act_1] + [tiles_mask_act_2] + ...]
        """
        completed = False
        try:
            for activation_idx in range(len(self.loaded_tile_data['processing_info'])):
                current_key = list(self.loaded_tile_data['processing_info'][activation_idx].keys())[0]
                self.post_tiles.extend(self.loaded_tile_data['processing_info'][activation_idx][current_key]["tile_info_post"][0])
                self.mask_tiles.extend(self.loaded_tile_data['processing_info'][activation_idx][current_key]["tile_info_mask"][0])
            completed = True
        except Exception as e:
            print(e.format_exc())
        finally:
            return completed

    def __len__(self) -> int:
        # m_dict = None
        # try:
        #     with open(f"{os.path.join(self.log_folder, self.master_dict)}", "r") as jsonfp:
        #         m_dict = json.load(jsonfp)
        # except:
        #     print(f"Error dealing with log json file at: {self.master_dict_path}")
        # finally:
        #     return len(m_dict)
        if len(self.post_tiles) == len(self.mask_tiles):
            return len(self.post_tiles)
        else:
            raise Exception("Number of tiles different from number of masks.")

    def __getitem__(self, idx) -> Tuple[str, str]:
        """
        Questa funzione prende in input un indice relativo all'immagine e alla maschera che vogliamo caricare sull'algoritmo da trainare
        e, tramite l'indice, carica la relativa immagine e maschera e le ritorna in un dizionario.
        """
        item_dict = dict()
        my_image = None
        my_mask = None

        if self.transformations is not None:
            my_image = self.transformations(self.post_tiles[idx])
            my_mask = self.transformations(self.mask_tiles[idx])
        
        if my_image is not None or my_mask is not None:
            item_dict["image"] = my_image
            item_dict["mask"] = my_mask
            return item_dict
        else:
            raise Exception("Error when loading mask or image.")