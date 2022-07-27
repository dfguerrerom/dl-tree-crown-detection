from pathlib import Path
import os

# Configuration of the parameters for the 2-UNetTraining.ipynb notebook
class Configuration:
    def __init__(self):
        
        # For reading the training areas and polygons
        self.base_dir = Path.home()/"1_modules/3_WADL/notebooks/"
        self.training_dir = self.base_dir/"train-jpg-kenya/"
        
        self.annotation_dir = self.training_dir/"mask"
        self.image_dir = self.training_dir/"img"
        
        self.model_dir = self.training_dir/"saved_models/UNet"
        
        self.path_to_write = self.training_dir/"1_processed"
        self.output_dir = self.training_dir/"2_prediction"
        self.img_to_predict = self.training_dir/"src"
        
        # For reading the VHR images
        self.bands = [0,1,2,3]
        
        self.image_type = ".tif"

        
        # Probability with which the generated patches should be normalized 0 -> don't 
        # normalize, 1 -> normalize all
        self.normalize = 0.4

        # Shape of the input data, height*width*channel; Here channels are NVDI and Pan
        self.input_shape = (128, 128, len(self.bands))
        self.input_image_channel = self.bands
        self.input_label_channel = [max(self.bands)+1]

        # CNN model related variables used in the notebook
        self.BATCH_SIZE = 32
        self.EPOCHS = 200

        # number of validation images to use
        self.VALID_IMG_COUNT = 200
        # maximum number of steps_per_epoch in training
        self.MAX_TRAIN_STEPS = 200


        # Output related variables
        self.output_image_type = ".tif"
        self.output_prefix = "det_"
        self.output_shapefile_type = ".shp"
        self.overwrite_analysed_files = False
        self.output_dtype = "uint8"

        # Variables related to batches and model
        self.BATCH_SIZE_PREDICT = 200  # Depends upon GPU memory and WIDTH and HEIGHT 
        # (Note: Batch_size for prediction can be different then for training.
        self.WIDTH = 128  # Should be same as the WIDTH used for training the model
        self.HEIGHT = 128  # Should be same as the HEIGHT used for training the model
        self.STRIDE = 64  # 224 or 196   # STRIDE = WIDTH means no overlap, 
        # STRIDE = WIDTH/2 means 50 % overlap in prediction
        
        paths = [item for item in self.__dict__.values() if isinstance(item, Path)]
        [path.mkdir(exist_ok=True, parents=True) for path in paths if not path.suffix]
        
config = Configuration()
