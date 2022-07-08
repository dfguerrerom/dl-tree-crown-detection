from importlib import reload
import itertools
from pathlib import Path
import matplotlib.pyplot as plt  # plotting tools
import PIL.ImageDraw
import numpy as np
import rasterio as rio  # I/O raster data (netcdf, height, geotiff, ...)
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import json
from IPython.core.interactiveshell import InteractiveShell
from tqdm import tqdm_notebook as tqdm
import warnings
import config.conf as conf

from core.frame_info import image_normalize
conf = reload(conf)
config = conf.Configuration()

warnings.filterwarnings("ignore")  # ignore annoying warnings
InteractiveShell.ast_node_interactivity = "all"

# Create boundary from polygon file
def calculateBoundaryWeight(polygonsInArea, scale_polygon=1.5, output_plot=True):
    """For each polygon, create a weighted boundary where the weights of shared/close 
    boundaries is higher than weights of solitary boundaries."""
    
    # If there are not polygons in a area, the boundary polygons return
    # an empty geo dataframe
    if not polygonsInArea:
        return gpd.GeoDataFrame({})

    pol_df = gpd.GeoDataFrame(pd.DataFrame(polygonsInArea))
    pol_df.reset_index(drop=True, inplace=True)
    pol_df["scaled"] = pol_df[["geometry"]].scale(*[scale_polygon]*3, origin="center")  

    intersections = []
    
    # for each polygon in area scale, compare with other polygons:
    for pol_idx, pol_row in pol_df.iterrows(), total=len(pol_df):

        # Intersect all the dataframe with current pol_row.scaled geometry
        intersect = pol_df[
            ~pol_df.index.isin([pol_idx])
        ]["scaled"].intersection(pol_row.scaled)

        intersect = intersect[~intersect.is_empty]
        intersections.append(intersect)

    intersections_gdf = gpd.GeoDataFrame(
        pd.concat([gpd.GeoDataFrame(series) for series in intersections])
    )
    intersections_gdf.columns = ["geometry"]
    
    if not len(intersections_gdf):
        return gpd.GeoDataFrame({})
    
    # Remove from the intersection the original tree shape
    intersections_gdf = intersections_gdf[intersections_gdf.geom_type == 'Polygon']

    bounda = gpd.overlay(intersections_gdf, pol_df, how="difference")
    bounda = bounda.explode()
    bounda.reset_index(drop=True, inplace=True)

    if output_plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        bounda.plot(ax=ax, color="red")
        plt.show()

    return bounda


# As input we received two shapefile, first one contains the training areas/rectangles
# and other contains the polygon of trees/objects in those training areas
# The first task is to determine the parent training area for each polygon and
# generate a weight map based upon the distance of a polygon boundary to other objects.
# Weight map will be used by the weighted loss during the U-Net training


def dividePolygonsInTrainingAreas(polygons_df, areas_df):
    """
    Assign annotated ploygons in to the training areas.
    """
    
    polygons_tmp = polygons_df.copy()
    polygons_by_area = {}
    
    for area_idx, area_row in tqdm(areas_df.iterrows(), total=len(areas_df)):
        
        spTemp = []
        allocated = []
        
        for pol_idx, pol_row in polygons_tmp.iterrows():
            if area_row.geometry.intersects(pol_row.geometry):
                spTemp.append(pol_row)
                allocated.append(pol_idx)

        # Order of bounds: minx miny maxx maxy
        boundary = calculateBoundaryWeight(
            spTemp,
            scale_polygon=1.5,
            output_plot=config.show_boundaries_during_processing,
        )
        
        polygons_by_area[area_idx] = {
            "polygons": spTemp,
            "boundaryWeight": boundary,
            "bounds": list(area_row.geometry.bounds),
        }
        
        # Remove those trees that were already assigned to an area
        polygons_tmp = polygons_tmp.drop(allocated)
        
    return polygons_by_area


def read_input_images(raw_image_base_dir, raw_image_file_type, raw_image_suffix):
    """Reads all images  in the image_base_dir directory."""

    return [
        path
        for path in raw_image_base_dir.rglob(f"*{raw_image_file_type}")
        if raw_image_suffix in path.stem
    ]


def drawPolygons(polygons, shape, outline, fill):
    """
    From the polygons, create a numpy mask with fill value in the
    foreground and 0 value in the background.
    Outline (i.e the edge of the polygon) can be assigned a separate value.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    # Syntax: PIL.ImageDraw.Draw.polygon(xy, fill=None, outline=None)
    # Parameters:
    # xy – Sequence of either 2-tuples like [(x, y), (x, y), …] or numeric values like [x, y, x, y, …].
    # outline – Color to use for the outline.
    # fill – Color to use for the fill.
    # Returns: An Image object.
    for polygon in polygons:
        xy = [(point[1], point[0]) for point in polygon]
        draw.polygon(xy=xy, outline=outline, fill=fill)
    mask = np.array(mask)  # , dtype=bool)
    return mask


# For each raw satellite image, determine if it overlaps with a training area.
# If a overlap if found, then extract + write the overlapping part of the raw
# image, create + write an image from training polygons and create + write an
# image from boundary weights in the that overlapping region.


def rowColPolygons(area_df, areaShape, profile, filename, outline, fill):
    """
    Convert polygons coordinates to image pixel coordinates,
    create annotation image using drawPolygons() and write the
    results into an image file.
    """
    transform = profile["transform"]
    polygons = []

    for i, row in area_df.iterrows():

        gm = row["geometry"]

        # Fix those polygons whose are multipol
        if gm.type != "Polygon":
            gm = gm.convex_hull

        a, b = zip(*list(gm.exterior.coords))
        row, col = rio.transform.rowcol(transform, a, b)
        zipped = list(zip(row, col))  # [list(rc) for rc in list(zip(row,col))]
        polygons.append(zipped)

    with open(filename, "w") as outfile:
        json.dump({"Trees": polygons}, outfile)

    mask = drawPolygons(polygons, areaShape, outline=outline, fill=fill)
    profile["dtype"] = rio.int16
    profile["count"] = 1

    with rio.open(filename.with_suffix(".png"), "w", **profile) as dst:
        dst.write(mask.astype(rio.int16), 1)


def writeExtractedImageAndAnnotation(
    name,
    masked,
    profile,
    polygonsInAreaDf,
    boundariesInAreaDf,
    area_id,
    normalize=True,
):
    """Write the part of raw image that overlaps with a training area into a separate
    image file. Use rowColPolygons to create and write annotation and boundary image
    from polygons in the training area.

    area_id (int): polyon area unique identificator

    """
    
    # Here we are going to create the images that we will use to train 
    with rio.open(config.out_image_dir / f"{name}_id{area_id}.png", "w", **profile) as dst:
        for band in range(len(config.bands)):
            norm_band = image_normalize(masked[band]).astype(profile["dtype"])
            # Remove zeros (if are)
            norm_band = np.nan_to_num(norm_band)
            dst.write(norm_band, band + 1)

    # The object is given a value of 1, the outline or the border of the
    # object is given a value of 0 and rest of the image/background is
    # given a value of 0
    
    
    # The boundaries are given a value of 1, the outline or the border of the
    # boundaries is also given a value of 1 and rest is given a value of 0

    annotation_json_filepath = config.annotation_dir / f"{name}_id{area_id}.json"
    boundary_json_filepath = config.boundary_dir / f"{name}_id{area_id}.json"
    
    paths = [annotation_json_filepath, boundary_json_filepath,]
    
    for i, df in enumerate([polygonsInAreaDf, boundariesInAreaDf]):
        rowColPolygons(
            df,
            (masked.shape[1], masked.shape[2]),
            profile,
            paths[i],
            outline=i,
            fill=1,
        )
        
def create_dataset(data, crs, transform, name):
    """Receives an image (ch, height, width)"""
    
    memfile = rio.io.MemoryFile()
    memfile.name = name
    dataset = memfile.open(
        driver='GTiff', 
        height=data.shape[1], 
        width=data.shape[2], 
        count=data.shape[0], 
        crs=crs, 
        transform=transform, 
        dtype=data.dtype
    )

    for band in range(data.shape[0]):
        dataset.write(data[band], band+1)
        
    return dataset


def findOverlap(input_images, areas_with_polygons,):
    """
    Finds overlap of image with a training area.
    Use writeExtractedImageAndAnnotation() to write the overlapping training area
    and corresponding polygons in separate image files.
    """

    overlaps = {area_id: [] for area_id in areas_with_polygons}
    
    # Some areas migth contain different images. Let's group them and create
    # the composite before extract their annotations.
    images_by_area = {}
    
    for img_path, area_id in itertools.product(input_images, areas_with_polygons):

        img = rio.open(img_path)
        areaInfo = areas_with_polygons[area_id]
        area_box = box(*areaInfo["bounds"])
        image_box = box(*img.bounds)
        
        if area_box.intersects(image_box): overlaps[area_id].append(img_path)
            
    return overlaps

def create_annotations(overlap_dict, areas_with_polygons):
    
    for area_id, images in tqdm(overlap_dict.items()):

        area_info = areas_with_polygons[area_id]
        trees_by_area_df = gpd.GeoDataFrame(area_info["polygons"])
        boundaries_by_area_df = gpd.GeoDataFrame(area_info["boundaryWeight"])
        area_box = box(*area_info["bounds"])

        if len(images) == 1:
            image = rio.open(images[0])

        else:
            rio_imgs = [rio.open(img) for img in images]
            merged, transform = rio.merge.merge(rio_imgs)
            crs = rio.open(images[0]).profile["crs"]
            name = "_".join([Path(img.name).stem for img in rio_imgs])
            image = create_dataset(merged, crs, transform, name)

        masked, mask_transform = rio.mask.mask(
            image, [area_box], all_touched=True, crop=True
        )

        profile = image.profile
        profile["height"] = masked.shape[1]
        profile["width"] = masked.shape[2]
        profile["transform"] = mask_transform
        profile["dtype"] = rio.float32

        name = Path(image.name).stem
        writeExtractedImageAndAnnotation(
            name,
            masked,
            profile,
            trees_by_area_df,
            boundaries_by_area_df,
            area_id,
        )

