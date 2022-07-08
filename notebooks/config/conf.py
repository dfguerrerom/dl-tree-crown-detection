from pathlib import Path
import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        
        # For reading the training areas and polygons
        self.base_dir = Path.home()/"1_modules/3_WADL/notebooks/"
        self.training_dir = self.base_dir/"training_afk/"
        
        self.train_image_dir = self.training_dir/"0_raw/images"
        self.annotation_dir = self.training_dir/"1_processed/annotation"
        self.boundary_dir = self.training_dir/"1_processed/boundary"
        
        self.image_dir = self.training_dir/"1_processed/image"
        
        self.training_area = self.training_dir/"0_raw/1_input_data/4th_batch/areas.shp"
        self.training_polygon = self.training_dir/"0_raw/1_input_data/4th_batch/trees.shp"

        self.model_dir = self.training_dir/"saved_models/UNet"
        
        self.path_to_write = self.training_dir/"1_processed"
        self.output_dir = self.training_dir/"2_prediction"
        self.patch_dir = self.training_dir/f"3_patches"
        
        self.out_image_dir = self.path_to_write / "image"
        self.out_annot_dir = self.path_to_write / "annotation"
        self.out_bound_dir = self.path_to_write / "boundary"

        # The split of training areas into training, validation and testing set, is
        # cached in patch_dir.
        
        self.frames_json = self.training_dir/"frames_list.json"

        # For reading the VHR images
        self.bands = [0,1,2,3,4,5,6,7]
        
        self.raw_image_file_type = ".tif"
        self.raw_image_suffix = "composite"

        self.show_boundaries_during_processing = False
        self.extracted_file_type = ".png"
        self.extracted_bands_folder = "bands"
        self.extracted_annotation_folder = "annotation"
        self.extracted_boundary_folder = "boundary"
        # Path to write should be a valid directory
                
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
        
        # Height * Width * (Input + Output) channels
        self.patch_size = (64, 64, len(self.bands)+2)  
        
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
        self.input_shape = (64, 64, len(self.bands))
        self.input_image_channel = self.bands
        self.input_label_channel = [max(self.bands)+1]
        self.input_weight_channel = [max(self.bands)+2]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 10
        self.NB_EPOCHS = 200

        # number of validation images to use
        self.VALID_IMG_COUNT = 200
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 96

        
        # Input related variables
        self.input_image_type = ".tif"

        # Output related variables
        self.output_image_type = ".tif"
        self.output_prefix = "det_"
        self.output_shapefile_type = ".shp"
        self.overwrite_analysed_files = False
        self.output_dtype = "uint8"

        # Variables related to batches and model
        self.BATCH_SIZE_PREDICT = 200  # Depends upon GPU memory and WIDTH and HEIGHT 
        # (Note: Batch_size for prediction can be different then for training.
        self.WIDTH = 64  # Should be same as the WIDTH used for training the model
        self.HEIGHT = 64  # Should be same as the HEIGHT used for training the model
        self.STRIDE = 64  # 224 or 196   # STRIDE = WIDTH means no overlap, 
        # STRIDE = WIDTH/2 means 50 % overlap in prediction
        
        paths = [item for item in self.__dict__.values() if isinstance(item, Path)]
        [path.mkdir(exist_ok=True, parents=True) for path in paths if not path.suffix]
        
config = Configuration()
