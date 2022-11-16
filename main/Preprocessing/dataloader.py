"""
QUESTO FILE CONTIENE LA CLASSE NECESSARIA A SCANNERIZZARE TUTTE LE INFORMAZIONI RELATIVE AL DATASET
A NOSTRA DISPOSIZIONE E SALVA TUTTI I RECORD IN UN FILE .CSV APPOSITAMENTE FORMATTATO(cocks)

IL WORKFLOW E' IL SEGUENTE:
    - PRENDO IMMAGINE DA /pre (RICOSTRUIRE TILES)
    - PRENDO IMMAGINE DA /post (RICOSTRUIRE TILES)
    - ASSEMBLO PRE E POST E LI METTO IN /complete (sono già tile-assembled)
    - APPLICO LA MASCHERA A QUELLA COMPLETA

"""

import os
import sys
import glob


class DatasetScanner():
    """
    Legge dalla master folder le cartelle all'interno delle quali ci sono pre, post e mask.
    Successivamente genera un csv con tutte le informazioni relative alla singola attivazione.
    In questo modo evitiamo di mantenere in memoria a runtime 520 giga di informazioni sul path assoluto
    E processiamo riga per riga il csv risultate. (generato da log_to_file()).
    """
    def __init__(self, master_folder_path=None, log_file_path=None, verbose=0, valid_list=None, ignore_list=None):
        self.master_folder_path=master_folder_path
        self.log_file_path=log_file_path
        self.verbose=verbose
        self.valid_list=valid_list
        self.ignore_list=ignore_list

    def scan_master_folder(self):
            return os.listdir(self.master_folder_path)

    def log_header_to_file(self, header="act_id"):
        """
        csv_header = act_id
        """
        with open(f"{self.log_file_path}", "w") as log_file:
            log_file.write(header + "\n")
    
    def log_to_file(self):
        with open(f"{self.log_file_path}", "a") as log_file:
            for path in self.scan_master_folder().filter(lambda folder : os.isdir(folder)):
                log_file.write(f"{path}")

class DatasetFormatter():
    def __init__(self):
        pass

    def check_dimensions(self):
        pass

    def tiling(self):
        pass
