import os
import numpy as np
import nrrd  # biblioteka pynrrd, pip install pynrrd

def read_nrrds(file_seg: str, file_org: str):
    """
    Reads two NRRD files: one segmented and one original.

    Parameters:
        file_seg (str): Path to the segmented NRRD file.
        file_org (str): Path to the original NRRD file.

    Returns:
        tuple: data_seg, header_seg, data_org, header_org
    """
    data_seg, header_seg = nrrd.read(file_seg)
    data_org, header_org = nrrd.read(file_org)
    return data_seg, header_seg, data_org, header_org

def find_indexes(data_seg, label_nr):
    """
    Finds the bounding box indices (min and max) of a given label in a segmented volume.

    Parameters:
        data_seg (ndarray): Segmented data array.
        label_nr (int): Label value to locate in the volume.

    Returns:
        tuple: mins (array), maxs (array) of bounding box coordinates.
    """
    indexes = np.argwhere(data_seg == label_nr)
    mins = np.min(indexes, axis=0)
    maxs = np.max(indexes, axis=0)
    return mins, maxs

def extract_bbox(key, label_value, segmentID, data_seg, header_seg, data_org, header_org, new_dir, padding=2):
    """
    Extracts a bounding box volume from the original image based on the labeled region in the segmented image and saves the original values in file.

    Parameters:
        key (str): Header key for the label value.
        label_value (int): The value representing the label in the segmented data.
        data_seg (ndarray): Segmented data array.
        header_seg (dict): Header of the segmented NRRD file.
        data_org (ndarray): Original data array.
        header_org (dict): Header of the original NRRD file.
        new_dir (str): Directory to save the extracted volume.
        padding (int, optional): Padding added to the bounding box in each dimension.

    Returns:
        tuple: new_filename (str), indexes (str) representing the bounding box ranges.
    """
    if key in header_seg:
        label_value = int(header_seg[key])
    mins, maxs = find_indexes(data_seg, label_value)
    mins = mins - padding
    maxs = maxs + padding
    mins[mins < 0] = 0
    maxs = [min(maxs[i], data_org.shape[i]) for i in range(3)]
    new_volume = data_org[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]
    new_filename = os.path.join(new_dir, f"{segmentID}.nrrd")
    nrrd.write(new_filename, new_volume, header_org)
    indexes = f"{mins[0]}:{maxs[0]}, {mins[1]}:{maxs[1]}, {mins[2]}:{maxs[2]}"
    return new_filename, indexes


def extract_bboxes(data_seg, header_seg, data_org, header_org, new_dir, textfile, new_dir_exists=False, padding=2):
    """
    Iterates over potential segments and extracts bounding boxes for labeled regions.
    Saves extracted volumes and writes metadata to a text file.

    Parameters:
        data_seg (ndarray): Segmented data array.
        header_seg (dict): Header of the segmented NRRD file.
        data_org (ndarray): Original data array.
        header_org (dict): Header of the original NRRD file.
        new_dir (str): Directory to save extracted volumes.
        textfile (str): File path to write metadata about extractions.
        new_dir_exists (bool, optional): Whether to allow using an existing directory. Defaults to False.
        padding (int, optional): Padding added to the bounding box in each dimension.
    """
    os.makedirs(new_dir, exist_ok=new_dir_exists)
    for x in range(26):
        key = f"Segment{x}_LabelValue"
        if key in header_seg:
            label_value = int(header_seg[key])
            raw_id = header_seg[f"Segment{x}_ID"]
            if raw_id.startswith("vertebrae_"):
                raw_id = raw_id.replace("vertebrae_", "")
                if raw_id[0] in ("L", "T") and raw_id[1:].isdigit():
                    number = int(raw_id[1:])
                    segmentID = f"{raw_id[0]}{number:02d}"
                    new_filename, indexes = extract_bbox(key, label_value, segmentID, data_seg, header_seg,
                                                         data_org, header_org,
                                                         new_dir, padding)
                    with open(textfile, 'a') as file:
                        file.write(new_filename + '\n')
                        file.write(str(indexes) + '\n')
                        file.write(str(segmentID) + '\n')

        else:
            break  # Exit when the next Segment{x}_LabelValue is not found


def read_and_extract(file_seg: str, file_org: str, new_dir: str, textfile: str = None, new_dir_exists=False,
                     padding: int = 2):
    """
    Function to read NRRD files and extract bounding boxes for labeled regions.

    Parameters:
        file_seg (str): Path to the segmented NRRD file.
        file_org (str): Path to the original NRRD file.
        new_dir (str): Directory to save extracted volumes.
        textfile (str, optional): File path to write metadata. Defaults to '<new_dir>/segmentation_results.txt'.
        new_dir_exists (bool, optional): Whether to allow using an existing directory. Defaults to False.
        padding (int, optional): Padding added to bounding boxes. Defaults to 2.
    """
    data_seg, header_seg, data_org, header_org = read_nrrds(file_seg, file_org)

    if textfile is None:
        textfile = new_dir + "\\segmentation_results.txt"
    extract_bboxes(data_seg, header_seg, data_org, header_org, new_dir, textfile, new_dir_exists, padding)
