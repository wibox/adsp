from Preprocessing.dataloader import DatasetScanner

import os

# Mettere un parser.
# Inserire la logica di inizializzazione degli oggetti
### QUI INIZIA IL MAIN ###
# Inizializzare i vari dataset con i vari split (per il dataset che abbiamo scelto dal parser)
# Inizializzare la rete scelta (dal parser)
# Trainare la rete scelta 
# Computare le metriche in output



ds = DatasetScanner(
    master_folder_path="/hdd/effis-wildfire",
    log_file_path=os.join.path("/logs", "act_id.csv")
)

ds.scan_master_folder()
ds.log_header_to_file()
ds.log_to_file()