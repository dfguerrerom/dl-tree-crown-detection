from collections import defaultdict
import cv2
from importlib import reload
from itertools import product
from tqdm import tqdm
import numpy as np
import rasterio as rio
import fiona
from shapely.geometry import Polygon, mapping
import PIL.Image, PIL.ImageDraw
from matplotlib.patches import Polygon

import config.conf as conf
from core.frame_info import  image_normalize

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

conf = reload(conf)
config = conf.Configuration()

# Methods to add results of a patch to the total results of a larger area. 
# The operator could be min (useful if there are too many false positives), 
# max (useful for tackle false negatives)
def addTOResult(res, prediction, row, col, he, wi, operator="MAX"):
    currValue = res[row : row + he, col : col + wi]
    newPredictions = prediction[:he, :wi]
    # IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! 
    # If you want to use MIN initial the mask with -1 and handle the case of default 
    # value(-1) separately.
    if operator == "MIN":  
        # Takes the min of current prediction and new prediction for each pixel
        currValue[currValue == -1] = 1  # Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, newPredictions)
    elif operator == "MAX":
        resultant = np.maximum(currValue, newPredictions)
    else:  # operator == 'REPLACE':
        resultant = newPredictions
    # Alternative approach; Lets assume that quality of prediction is better in
    # the centre of the image than on the edges
    # We use numbers from 1-5 to denote the quality, where 5 is the best and 1
    # is the worst.In that case, the best result would be to take into quality 
    # of prediction based upon position in account
    # So for merge with stride of 0.5, for eg. [12345432100000] AND [00000123454321], 
    # should be [1234543454321] instead of [1234543214321] that you will currently get.
    # However, in case the values are strecthed before hand this problem will
    # be minimized
    res[row : row + he, col : col + wi] = resultant
    return res

# Methods that actually makes the predictions
def predict_using_model(model, batch, batch_pos, mask, operator):
    tm = np.stack(batch, axis=0)
    prediction = model.predict(tm)
    for i in range(len(batch_pos)):
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(prediction[i], axis=-1)
        # Instead of replacing the current values with new values, 
        # use the user specified operator (MIN,MAX,REPLACE)
        mask = addTOResult(mask, p, row, col, he, wi, operator)
    return mask


def detect_tree(model, img, width=256, height=256, stride=128, normalize=True):
    
    nols, nrows = img.meta["width"], img.meta["height"]
    meta = img.meta.copy()
    if "float" not in meta["dtype"]:  
        # The prediction is a float so we keep it as float 
        # to be consistent with the prediction.
        meta["dtype"] = np.float32

    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = rio.windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    mask = np.zeros((nrows, nols), dtype=meta["dtype"])
    #     mask = mask -1 # Note: The initial mask is initialized with -1 
    # instead of zero to handle the MIN case (see addToResult)

    batch = []
    batch_pos = []
    for col_off, row_off in tqdm(offsets):

        window = rio.windows.Window(
            col_off=col_off, row_off=row_off, width=width, height=height
        ).intersection(big_window)

        transform = rio.windows.transform(window, img.transform)
        # Add zero padding in case of corner images
        patch = np.zeros((height, width, 8))  
        img_sm = img.read(window=window)
        img_sm = np.transpose(img_sm, axes=(1,2,0))

        temp_im = np.squeeze(img_sm)

        if normalize:
            temp_im = image_normalize(temp_im, axis=(0, 1))  
            # Normalize the image along the width and height i.e. independently 
            # per channel

        patch[: window.height, : window.width] = temp_im
        batch.append(patch)
        batch_pos.append((window.col_off, window.row_off, window.width, window.height))

        if len(batch) == config.BATCH_SIZE:
            mask = predict_using_model(model, batch, batch_pos, mask, "MAX")
            batch = []
            batch_pos = []

    # To handle the edge of images as the image size may not be divisible by n complete
    # batches and few frames on the edge may be left.
    if batch:
        mask = predict_using_model(model, batch, batch_pos, mask, "MAX")
        batch = []
        batch_pos = []

    return mask, meta

def drawPolygons(polygons, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    for polygon in polygons:
        xy = [(point[1], point[0]) for point in polygon]
        draw.polygon(xy=xy, outline=255, fill=255)
    mask = np.array(mask)  # , dtype=bool)
    return mask


def transformToXY(polygons, transform):
    tp = []
    for polygon in polygons:
        rows, cols = zip(*polygon)
        x, y = rio.transform.xy(transform, rows, cols)
        tp.append(list(zip(x, y)))
    return tp


def createShapefileObject(polygons, meta, wfile):

    schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "str",
            "canopy": "float:15.2",
        },
    }

    with fiona.open(
        wfile,
        "w",
        crs=meta.get("crs").to_dict(),
        driver="ESRI Shapefile",
        schema=schema,
    ) as sink:
        for idx, mp in enumerate(polygons):
            try:
                #                 poly = Polygon(poly)
                #             assert mp.is_valid
                #             assert mp.geom_type == 'Polygon'
                sink.write(
                    {
                        "geometry": mapping(mp),
                        "properties": {"id": str(idx), "canopy": mp.area},
                    }
                )
            except:
                print(
                    "An exception occurred in createShapefileObject; Polygon"
                    " must have more than 2 points"
                )




def mask_to_polygons(maskF, transform):
    # first, find contours with cv2: it's much faster than shapely
    th = 0.5
    mask = maskF.copy()
    mask[mask < th] = 0
    mask[mask >= th] = 1
    mask = ((mask) * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Convert contours from image coordinate to xy coordinate
    contours = transformContoursToXY(contours, transform)
    if not contours:  # TODO: Raise an error maybe
        print("Warning: No contours/polygons detected!!")
        return [Polygon()]
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []

    for idx, cnt in enumerate(contours):
        if idx not in child_contours:  
            
            # and cv2.contourArea(cnt) >= min_area: #Do we need to check for min_area??
            try:
                poly = Polygon(shell=cnt, holes=[c for c in cnt_children.get(idx, [])])
                # if cv2.contourArea(c) >= min_area]) #Do we need to check for min_area??
                all_polygons.append(poly)
            except:
                pass
    #                 print("An exception occurred in createShapefileObject; 
    # Polygon must have more than 2 points")
    print(len(all_polygons))
    return all_polygons


def create_contours_shapefile(mask, meta, out_fn):
    res = mask_to_polygons(mask, meta["transform"])
    # res = transformToXY(contours, meta['transform'])
    createShapefileObject(res, meta, out_fn)


def writeMaskToDisk(
    detected_mask,
    detected_meta,
    wp,
    write_as_type="uint8",
    th=0.5,
    create_countors=False,
):
    # Convert to correct required before writing
    if "float" in str(detected_meta["dtype"]) and "int" in write_as_type:
        print(
            f"Converting prediction from {detected_meta['dtype']} to {write_as_type},"
            f"using threshold of {th}"
        )
        detected_mask[detected_mask < th] = 0
        detected_mask[detected_mask >= th] = 1
        detected_mask = detected_mask.astype(write_as_type)
        detected_meta["dtype"] = write_as_type
        detected_meta["count"] = 1

    with rio.open(wp, "w", **detected_meta) as outds:
        outds.write(detected_mask, 1)
        
    if create_countors:
        wp = wp.with_suffix(".shp")
        create_contours_shapefile(detected_mask, detected_meta, wp)