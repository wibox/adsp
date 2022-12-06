from Preprocessing import datasetformatter as df
from Preprocessing import datasetscanner as ds
# from Preprocessing import imagedataset as id

INITIAL_DATASET_PATH = "/home/francesco/Desktop/colomba_dataset"
FORMATTED_DATASET_PATH = "/home/francesco/Desktop/formatted_colombaset"

datascanner = ds.DatasetScanner(
        master_folder_path=INITIAL_DATASET_PATH,
        log_file_path="Log/master_folder_log_colombaset.csv",
        validation_file_path=None,
        dataset="colombaset",
        verbose=1
    )

datascanner.scan_master_folder()
datascanner.log_header_to_file(header="act_id,pre,post,mask")
datascanner.log_to_file()

dataformatter = df.DatasetFormatter(
        master_folder_path=FORMATTED_DATASET_PATH,
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

