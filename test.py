from Preprocessing import datasetformatter as df
from Preprocessing import datasetscanner as ds
# from Preprocessing import imagedataset as id

datascanner = ds.DatasetScanner(
        master_folder_path="/mnt/c/Users/franc/Desktop/colomba_dataset",
        log_file_path="Log/master_folder_log_colombaset.csv",
        validation_file_path=None,
        dataset="colombaset",
        verbose=1
    )

datascanner.scan_master_folder()
datascanner.log_header_to_file(header="act_id,pre,post,mask")
datascanner.log_to_file()

dataformatter = df.DatasetFormatter(
        master_folder_path="/mnt/c/Users/franc/Desktop/formatted_colomba_dataset",
        log_file_path="Log/",
        log_filename="master_folder_log_colombaset.csv",
        master_dict_path="Log/",
        tile_height=512,
        tile_width=512,
        thr_pixels=112,
        use_pre=False,
        dataset="colombaset",
        verbose=1
)

dataformatter.tiling()

