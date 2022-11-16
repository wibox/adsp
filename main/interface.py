from Preprocessing.dataloader import DatasetScanner

import os

ds = DatasetScanner(
    master_folder_path="/hdd/effis-wildfire",
    log_file_path=os.join.path("/logs", "act_id.csv")
)

ds.scan_master_folder()
ds.log_header_to_file()
ds.log_to_file()