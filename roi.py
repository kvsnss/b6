import cv2
import numpy as np
import os


# =========================================================
# ROI DETECTION
# =========================================================
#
# A central rectangular region is used as the ROI.
# The ROI is protected from data embedding.
#
# Everything outside the ROI is treated as Non-ROI.
# The Non-ROI region is available for RDH embedding.
#
# ROI and Non-ROI images are saved in the output folder,
# but they are NOT displayed on the website.
#
# =========================================================

def detect_roi(image_path):

    # -----------------------------------------------------
    # READ MEDICAL IMAGE
    # -----------------------------------------------------

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise ValueError(
            "Medical image could not be loaded."
        )

    height, width = image.shape

    # -----------------------------------------------------
    # CREATE ROI MASK
    # -----------------------------------------------------

    roi_mask = np.zeros(
        image.shape,
        dtype=np.uint8
    )

    # -----------------------------------------------------
    # DEFINE CENTRAL ROI
    # -----------------------------------------------------

    y1 = height // 4
    y2 = 3 * height // 4

    x1 = width // 4
    x2 = 3 * width // 4

    roi_mask[
        y1:y2,
        x1:x2
    ] = 255

    # -----------------------------------------------------
    # CREATE NON-ROI MASK
    # -----------------------------------------------------

    non_roi_mask = cv2.bitwise_not(
        roi_mask
    )

    # -----------------------------------------------------
    # CREATE ROI IMAGE
    # -----------------------------------------------------

    roi_image = cv2.bitwise_and(
        image,
        image,
        mask=roi_mask
    )

    # -----------------------------------------------------
    # CREATE NON-ROI IMAGE
    # -----------------------------------------------------

    non_roi_image = cv2.bitwise_and(
        image,
        image,
        mask=non_roi_mask
    )

    # -----------------------------------------------------
    # RETURN ALL REQUIRED DATA
    # -----------------------------------------------------

    return (
        image,
        roi_mask,
        non_roi_mask,
        roi_image,
        non_roi_image
    )


# =========================================================
# SAVE ROI AND NON-ROI IMAGES
# =========================================================

def save_roi_images(
    roi_image,
    non_roi_image,
    output_dir="output"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # -----------------------------------------------------
    # SAVE ROI IMAGE
    # -----------------------------------------------------

    roi_path = os.path.join(
        output_dir,
        "roi.png"
    )

    cv2.imwrite(
        roi_path,
        roi_image
    )

    # -----------------------------------------------------
    # SAVE NON-ROI IMAGE
    # -----------------------------------------------------

    non_roi_path = os.path.join(
        output_dir,
        "non_roi.png"
    )

    cv2.imwrite(
        non_roi_path,
        non_roi_image
    )

    return {
        "roi": "roi.png",
        "non_roi": "non_roi.png"
    }