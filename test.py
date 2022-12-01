from Preprocessing import dataloader as dl

ds = dl.DatasetScanner(
        master_folder_path="/mnt/c/Users/franc/Desktop/colomba_dataset",
        log_file_path="log/master_folder_log_colombaset.csv",
        validation_file_path=None,
        dataset="colombaset",
        verbose=1
    )

ds.scan_master_folder()
ds.log_header_to_file(header="act_id,pre,post,mask")
ds.log_to_file()