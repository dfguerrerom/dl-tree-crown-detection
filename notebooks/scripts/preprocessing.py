from importlib import reload
import itertools
from pathlib import Path
import matplotlib.pyplot as plt  # plotting tools
import PIL.ImageDraw
import numpy as np
import rasterio as rio  # I/O raster data (netcdf, height, geotiff, ...)
import rasterio.mask
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
    """For each polygon, create a weighted boundary
    where the weights of shared/close boundaries
    is higher than weights of solitary boundaries.
    """
    # If there are not polygons in a area, the boundary polygons return
    # an empty geo dataframe

    if not polygonsInArea:
        return gpd.GeoDataFrame({})
    tempPolygonDf = pd.DataFrame(polygonsInArea)
    tempPolygonDf.reset_index(drop=True, inplace=True)
    tempPolygonDf = gpd.GeoDataFrame(tempPolygonDf.drop(columns=["id"]))
    new_c = []
    # for each polygon in area scale, compare with other polygons:
    for i in tqdm(range(len(tempPolygonDf))):
        pol1 = gpd.GeoSeries(tempPolygonDf.iloc[i]["geometry"])
        sc = pol1.scale(
            xfact=scale_polygon,
            yfact=scale_polygon,
            zfact=scale_polygon,
            origin="center",
        )
        scc = pd.DataFrame(columns=["id", "geometry"])
        scc = scc.append({"id": None, "geometry": sc[0]}, ignore_index=True)
        scc = gpd.GeoDataFrame(pd.concat([scc] * len(tempPolygonDf), ignore_index=True))

        pol2 = gpd.GeoDataFrame(tempPolygonDf[~tempPolygonDf.index.isin([i])])
        # scale pol2 also and then intersect, so in the end no need for scale
        pol2 = gpd.GeoDataFrame(
            pol2.scale(
                xfact=scale_polygon,
                yfact=scale_polygon,
                zfact=scale_polygon,
                origin="center",
            )
        )
        pol2.columns = ["geometry"]

        ints = scc.intersection(pol2)
        for k in range(len(ints)):
            if ints.iloc[k] is not None:
                if ints.iloc[k].is_empty != 1:
                    new_c.append(ints.iloc[k])

    new_c = gpd.GeoSeries(new_c)
    new_cc = gpd.GeoDataFrame({"geometry": new_c})
    new_cc.columns = ["geometry"]
    new_cc = new_cc[new_cc.geom_type == 'Polygon']
    bounda = gpd.overlay(new_cc, tempPolygonDf, how="difference")
    if output_plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        bounda.plot(ax=ax, color="red")
        plt.show()
    # change multipolygon to polygon
    bounda = bounda.explode()
    bounda.reset_index(drop=True, inplace=True)
    # bounda.to_file('boundary_ready_to_use.shp')
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
    # For efficiency, assigned polygons are removed from the list, we
    # make a copy here.

    polygons_tmp = polygons_df.copy()
    polygons_by_area = {}
    
    for idx, area_row in tqdm(areas_df.iterrows(), total=len(areas_df)):
        
        spTemp = []
        allocated = []
        
        for pol_idx, pol_row in polygons_tmp.iterrows():
            
            if area_row.geometry.intersects(pol_row.geometry]):
                
                spTemp.append(pol_row)
                allocated.append(j)

            # Order of bounds: minx miny maxx maxy
        boundary = calculateBoundaryWeight(
            spTemp,
            scale_polygon=1.5,
            output_plot=config.show_boundaries_during_processing,
        )
        
        splitPolygons[area_idx] = {
            "polygons": spTemp,
            "boundaryWeight": boundary,
            "bounds": list(area_row.geometry.bounds),
        }
        
        cpTrainingPolygon = cpTrainingPolygon.drop(allocated)
        
    return splitPolygons


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
    sm,
    profile,
    polygonsInAreaDf,
    boundariesInAreaDf,
    writePath,
    bands_folder,
    annotation_folder,
    boundary_folder,
    bands,
    area_id,
    normalize=True,
):
    """Write the part of raw image that overlaps with a training area into a separate
    image file. Use rowColPolygons to create and write annotation and boundary image
    from polygons in the training area.

    area_id (int): polyon area unique identificator

    """
    
    with rio.open(conf.out_image_dir / f"{name}_id{area_id}.png", "w", **profile) as dst:
        for band in range(len(bands)):
            norm_band = image_normalize(sm[0][band]).astype(profile["dtype"])
            dst.write(norm_band, band + 1)

    if annotation_folder:
        annotation_json_filepath = conf.annotation_folder / f"{name}_id{area_id}.json"
        # The object is given a value of 1, the outline or the border of the
        # object is given a value of 0 and rest of the image/background is
        # given a value of 0
        rowColPolygons(
            polygonsInAreaDf,
            (sm[0].shape[1], sm[0].shape[2]),
            profile,
            annotation_json_filepath,
            outline=0,
            fill=1,
        )
    if boundary_folder:
        boundary_json_filepath = conf.boundary_folder / f"{name}_id{area_id}.json"
        # The boundaries are given a value of 1, the outline or the border of the
        # boundaries is also given a value of 1 and rest is given a value of 0
        rowColPolygons(
            boundariesInAreaDf,
            (sm[0].shape[1], sm[0].shape[2]),
            profile,
            boundary_json_filepath,
            outline=1,
            fill=1,
        )


def findOverlap(
    input_images,
    areas_with_polygons,
    writePath,
    bands_folder,
    annotation_folder,
    boundary_folder,
    bands,
):
    """
    Finds overlap of image with a training area.
    Use writeExtractedImageAndAnnotation() to write the overlapping training area
    and corresponding polygons in separate image files.
    """

    if not writePath.exists():
        writePath.mkdir(exist_ok=True, parents=True)

    overlaps = {image.name: [] for image in input_images}

    for img_path, area_id in itertools.product(input_images, areas_with_polygons):

        name = img_path.stem
        img = rio.open(img_path)

        areaInfo = areas_with_polygons[area_id]

        # Convert the polygons in the area in a dataframe and get the bounds of
        # the area.
        polygonsInAreaDf = gpd.GeoDataFrame(areaInfo["polygons"])
        boundariesInAreaDf = gpd.GeoDataFrame(areaInfo["boundaryWeight"])
        bboxArea = box(*areaInfo["bounds"])
        bboxImg = box(*img.bounds)
        # Extract the window if area is in the image

        if bboxArea.intersects(bboxImg):
            profile = img.profile
            sm = rio.mask.mask(img, [bboxArea], all_touched=True, crop=True)
            profile["height"] = sm[0].shape[1]
            profile["width"] = sm[0].shape[2]
            profile["transform"] = sm[1]
            # That's a problem with rio, if the height and the width are less
            # then 256 it throws: ValueError: blockysize exceeds raster height
            # So I set the blockxsize and blockysize to prevent this problem
            profile["dtype"] = rio.float32
            # writeExtractedImageAndAnnotation writes the image, annotation and
            # boundaries and returns the counter of the next file to write.

            writeExtractedImageAndAnnotation(
                name,
                sm,
                profile,
                polygonsInAreaDf,
                boundariesInAreaDf,
                writePath,
                bands_folder,
                annotation_folder,
                boundary_folder,
                bands,
                area_id,
            )

            overlaps[img_path.name].append(area_id)

    return overlaps
