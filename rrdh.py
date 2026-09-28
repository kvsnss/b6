import cv2
import numpy as np

from dwt import integer_dwt2, integer_idwt2
from roi import detect_roi
from rdh import rdh_extract

# EXTRACT PAYLOAD FROM STEGO IMAGE

def extract_payload(
    stego_path,
    metadata,
    recovered_image_path=None
):

    # READ STEGO IMAGE
    
    image = cv2.imread(
        stego_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise ValueError(
            "Unable to read received stego image."
        )

    # CHECK DIMENSIONS
 
    height, width = image.shape

    if height % 2 != 0:

        raise ValueError(
            "Stego image height must be even."
        )

    if width % 2 != 0:

        raise ValueError(
            "Stego image width must be even."
        )

    # RECREATE ROI
   
    _, roi_mask, non_roi_mask = detect_roi(
        stego_path
    )

    # INTEGER DWT OF STEGO IMAGE

    LL, LH, HL, HH_embedded = integer_dwt2(
        image
    )

    # RDH EXTRACTION

    recovered_data, recovered_HH = rdh_extract(
        HH_embedded,
        non_roi_mask,
        metadata
    )

    # IDWT
    # -----------------------------------------------------
    #
    # Use recovered HH.
    #
    # This should reconstruct the
    # original medical image.
    #
    # -----------------------------------------------------

    recovered_image = integer_idwt2(
        LL,
        LH,
        HL,
        recovered_HH
    )

    # EXACT RANGE CHECK
    
    if (
        recovered_image.min() < 0
        or
        recovered_image.max() > 255
    ):

        raise ValueError(
            "Recovered image contains invalid pixel values."
        )

    recovered_image = recovered_image.astype(
        np.uint8
    )

    # SAVE RECOVERED IMAGE

    if recovered_image_path:

        cv2.imwrite(
            recovered_image_path,
            recovered_image
        )

    return (
        recovered_data,
        recovered_image
    )