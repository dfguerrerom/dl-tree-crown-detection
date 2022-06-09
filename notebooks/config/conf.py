from pathlib import Path
import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        
        # For reading the training areas and polygons
        self.base_dir = Path("/home/dguerrero/1_modules/3_WADL/notebooks/")
        self.training_dir = self.base_dir/"training_afk/"
        
        self.model_dir = self.training_dir/"saved_models/UNet"
        
        self.image_dir = self.training_dir/"0_raw/images"
        self.annotation_dir = self.training_dir/"0_raw/annotation"
        self.boundary_dir = self.training_dir/"0_raw/boundary"
        
        self.path_to_write = self.training_dir/"1_processed"
        self.output_dir = self.training_dir/"2_prediction"
        self.patch_dir = self.training_dir/f"3_patches"
        
        out_image_dir = self.path_to_write / "image"
        out_annot_dir = self.path_to_write / "annotation"
        out_bound_dir = self.path_to_write / "boundary"

        # The split of training areas into training, validation and testing set, is
        # cached in patch_dir.
        
        self.frames_json = self.training_dir/"frames_list.json"
        
        folders = [
            self.model_dir,
            self.image_dir,
            self.image_dir,
            self.annotation_dir,
            self.boundary_dir,
            self.path_to_write,
            self.output_dir,
            self.patch_dir,
            out_image_dir,
            out_annot_dir,
            out_bound_dir,
        ]
        
        # Initialize folders
        [f.mkdir(exist_ok=True, parents=True) for f in folders]
        
        
        self.training_area_fn = self.training_dir/"0_raw/annotations/training_area_proj.shp"
        self.training_polygon_fn = self.training_dir/"0_raw/annotations/training_polygons2.shp"

        # For reading the VHR images
        self.bands = [0,1,2,3]
        
        self.raw_image_file_type = ".tif"
        self.raw_image_suffix = "harmonized_clip"

        self.show_boundaries_during_processing = False
        self.extracted_file_type = ".png"
        self.extracted_bands_folder = "bands"
        self.extracted_annotation_folder = "annotation"
        self.extracted_boundary_folder = "boundary"
        # Path to write should be a valid directory
                
        if not len(os.listdir(self.path_to_write)) == 0:
            print(
                "Warning: path_to_write is not empty! The old files in the directory"
                " may not be overwritten!!"
            )
                    
        self.image_type = ".png"
        self.annotation_fn = "annotation"
        self.weight_fn = "boundary"

        # Patch generation; from the training areas (extracted in the last notebook), 
        # we generate fixed size patches.
        
        # random: a random training area is selected and a patch in extracted from a 
        # random location inside that training area. Uses a lazy stratergy i.e. batch 
        # of patches are extracted on demand.
        
        # sequential: training areas are selected in the given order and patches 
        # extracted from these areas sequential with a given step size. All the 
        # possible
        # patches are returned in one call.
        
        self.patch_generation_stratergy = "random"  # 'random' or 'sequential'
        self.patch_size = (256, 256, 6)  # Height * Width * (Input + Output) channels
        # # When stratergy == sequential, then you need the step_size as well
        # step_size = (128,128)

        # The training areas are divided into training, validation and testing set. 
        # Note that training area can have different sizes, so it doesn't guarantee 
        # that the final generated patches (when using sequential stratergy) will be in 
        # the same ratio.
        self.test_ratio = 0.2
        self.val_ratio = 0.2

        # Probability with which the generated patches should be normalized 0 -> don't 
        # normalize, 1 -> normalize all
        self.normalize = 0.4



        # Shape of the input data, height*width*channel; Here channels are NVDI and Pan
        self.input_shape = (256, 256, 4)
        self.input_image_channel = [0, 1, 2, 3]
        self.input_label_channel = [4]
        self.input_weight_channel = [5]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 8
        self.NB_EPOCHS = 200

        # number of validation images to use
        self.VALID_IMG_COUNT = 200
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 1000

        
        # Input related variables
        self.input_image_type = ".tif"

        # Output related variables
        self.output_image_type = ".tif"
        self.output_prefix = "det_"
        self.output_shapefile_type = ".shp"
        self.overwrite_analysed_files = False
        self.output_dtype = "uint8"

        # Variables related to batches and model
        self.BATCH_SIZE = 200  # Depends upon GPU memory and WIDTH and HEIGHT 
        # (Note: Batch_size for prediction can be different then for training.
        self.WIDTH = 256  # Should be same as the WIDTH used for training the model
        self.HEIGHT = 256  # Should be same as the HEIGHT used for training the model
        self.STRIDE = 224  # 224 or 196   # STRIDE = WIDTH means no overlap, 
        # STRIDE = WIDTH/2 means 50 % overlap in prediction        
        
conf = Configuration()