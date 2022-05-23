import os
from pathlib import Path

# Configuration of the parameters for the 1-Preprocessing.ipynb notebook
class Configuration:
    """
    Configuration for the first notebook.
    Copy the configTemplate folder and define the paths to input and output data. Variables such as raw_ndvi_image_prefix may also need to be corrected if you are use a different source.
    """
    def __init__(self):
        # For reading the training areas and polygons
        self.training_base_dir = Path(
            "/home/dguerrero/1_modules/3_WADL/notebooks/training_data/0_raw/annotations"
        )
        self.training_area_fn = "training_area_proj.shp"
        self.training_polygon_fn = "training_polygons2.shp"

        # For reading the VHR images
        self.bands = [0,1,2,3]
        self.raw_image_base_dir = Path(
            "/home/dguerrero/1_modules/3_WADL/notebooks/training_data/0_raw/images"
        )
        self.raw_image_file_type = ".tif"
        self.raw_image_suffix = "harmonized_clip"

        # For writing the extracted images and their corresponding
        # annotations and boundary file
        self.path_to_write = Path(
            "/home/dguerrero/1_modules/3_WADL/notebooks/training_data/0_raw/1_processed"
        )
        self.show_boundaries_during_processing = False
        self.extracted_file_type = ".png"
        self.extracted_bands_folder = "bands"
        self.extracted_annotation_folder = "annotation"
        self.extracted_boundary_folder = "boundary"
        # Path to write should be a valid directory
        

        bands_folder = self.path_to_write / self.extracted_bands_folder
        bands_folder.mkdir(exist_ok=True, parents=True)

        annotation_folder = self.path_to_write / self.extracted_annotation_folder
        annotation_folder.mkdir(exist_ok=True, parents=True)

        boundary_folder = self.path_to_write / self.extracted_boundary_folder
        boundary_folder.mkdir(exist_ok=True, parents=True)
        
        if not len(os.listdir(self.path_to_write)) == 0:
            print(
                "Warning: path_to_write is not empty! The old files in the directory"
                " may not be overwritten!!"
            )
            
        assert os.path.exists(self.path_to_write)            