import cv2
import numpy as np
import os


# =========================================================
# INTEGER HAAR DWT - 1D FORWARD
# =========================================================

def haar_forward_1d(x):

    x = x.astype(np.int64)

    even = x[::2]
    odd = x[1::2]

    # Prediction step
    d = odd - even

    # Update step
    s = even + np.floor_divide(d, 2)

    return s, d


# =========================================================
# INTEGER HAAR DWT - 1D INVERSE
# =========================================================

def haar_inverse_1d(s, d):

    s = s.astype(np.int64)
    d = d.astype(np.int64)

    # Inverse update
    even = s - np.floor_divide(d, 2)

    # Inverse prediction
    odd = d + even

    result = np.empty(
        even.size * 2,
        dtype=np.int64
    )

    result[::2] = even
    result[1::2] = odd

    return result


# =========================================================
# INTEGER HAAR DWT - 2D
# =========================================================

def integer_dwt2(image):

    image = image.astype(np.int64)

    height, width = image.shape

    # Reversible 2D Haar implementation requires
    # even image dimensions.
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            "Image width and height must be even. "
            f"Current size: {width} x {height}"
        )

    # -----------------------------------------------------
    # ROW TRANSFORM
    # -----------------------------------------------------

    low_rows = np.zeros(
        (height, width // 2),
        dtype=np.int64
    )

    high_rows = np.zeros(
        (height, width // 2),
        dtype=np.int64
    )

    for r in range(height):

        s, d = haar_forward_1d(
            image[r, :]
        )

        low_rows[r, :] = s
        high_rows[r, :] = d

    # -----------------------------------------------------
    # COLUMN TRANSFORM
    # -----------------------------------------------------

    LL = np.zeros(
        (height // 2, width // 2),
        dtype=np.int64
    )

    LH = np.zeros(
        (height // 2, width // 2),
        dtype=np.int64
    )

    HL = np.zeros(
        (height // 2, width // 2),
        dtype=np.int64
    )

    HH = np.zeros(
        (height // 2, width // 2),
        dtype=np.int64
    )

    for c in range(width // 2):

        # Low-row component
        s, d = haar_forward_1d(
            low_rows[:, c]
        )

        LL[:, c] = s
        HL[:, c] = d

        # High-row component
        s, d = haar_forward_1d(
            high_rows[:, c]
        )

        LH[:, c] = s
        HH[:, c] = d

    return LL, LH, HL, HH


# =========================================================
# INTEGER HAAR IDWT - 2D
# =========================================================

def integer_idwt2(
    LL,
    LH,
    HL,
    HH
):

    half_height, half_width = LL.shape

    height = half_height * 2
    width = half_width * 2

    # -----------------------------------------------------
    # INVERSE COLUMN TRANSFORM
    # -----------------------------------------------------

    low_rows = np.zeros(
        (height, half_width),
        dtype=np.int64
    )

    high_rows = np.zeros(
        (height, half_width),
        dtype=np.int64
    )

    for c in range(half_width):

        # Reconstruct low-row component
        low_rows[:, c] = haar_inverse_1d(
            LL[:, c],
            HL[:, c]
        )

        # Reconstruct high-row component
        high_rows[:, c] = haar_inverse_1d(
            LH[:, c],
            HH[:, c]
        )

    # -----------------------------------------------------
    # INVERSE ROW TRANSFORM
    # -----------------------------------------------------

    image = np.zeros(
        (height, width),
        dtype=np.int64
    )

    for r in range(height):

        image[r, :] = haar_inverse_1d(
            low_rows[r, :],
            high_rows[r, :]
        )

    return image


# =========================================================
# NORMALIZE DWT BAND FOR DISPLAY ONLY
# =========================================================

def normalize_band(data):

    data = np.asarray(data)

    minimum = data.min()
    maximum = data.max()

    if maximum == minimum:

        return np.zeros(
            data.shape,
            dtype=np.uint8
        )

    normalized = (
        (data - minimum)
        /
        (maximum - minimum)
        *
        255
    )

    return normalized.astype(
        np.uint8
    )


# =========================================================
# SAVE DWT BANDS
# =========================================================

def save_dwt_bands(
    LL,
    LH,
    HL,
    HH,
    output_dir="output"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # LL
    cv2.imwrite(
        os.path.join(
            output_dir,
            "dwt_ll.png"
        ),
        normalize_band(LL)
    )

    # LH
    cv2.imwrite(
        os.path.join(
            output_dir,
            "dwt_lh.png"
        ),
        normalize_band(LH)
    )

    # HL
    cv2.imwrite(
        os.path.join(
            output_dir,
            "dwt_hl.png"
        ),
        normalize_band(HL)
    )

    # HH
    cv2.imwrite(
        os.path.join(
            output_dir,
            "dwt_hh.png"
        ),
        normalize_band(HH)
    )


# =========================================================
# SAVE ROI AND NON-ROI IMAGES
# =========================================================

def save_roi_images(
    image,
    roi_mask,
    non_roi_mask,
    output_dir="output"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Make sure masks have same dimensions
    if roi_mask.shape != image.shape:

        roi_mask = cv2.resize(
            roi_mask,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    if non_roi_mask.shape != image.shape:

        non_roi_mask = cv2.resize(
            non_roi_mask,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )

    # Ensure uint8
    roi_mask = roi_mask.astype(
        np.uint8
    )

    non_roi_mask = non_roi_mask.astype(
        np.uint8
    )

    # -----------------------------------------------------
    # ROI IMAGE
    # -----------------------------------------------------

    roi_image = cv2.bitwise_and(
        image,
        image,
        mask=roi_mask
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "roi.png"
        ),
        roi_image
    )

    # -----------------------------------------------------
    # NON-ROI IMAGE
    # -----------------------------------------------------

    non_roi_image = cv2.bitwise_and(
        image,
        image,
        mask=non_roi_mask
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "non_roi.png"
        ),
        non_roi_image
    )

    return roi_image, non_roi_image


# =========================================================
# CREATE STEGO IMAGE
# =========================================================

def create_stego_image(
    LL,
    LH,
    HL,
    HH_embedded,
    stego_path
):

    # -----------------------------------------------------
    # INTEGER IDWT
    # -----------------------------------------------------

    idwt_image = integer_idwt2(
        LL,
        LH,
        HL,
        HH_embedded
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # DO NOT USE np.clip()
    # -----------------------------------------------------

    minimum = int(
        idwt_image.min()
    )

    maximum = int(
        idwt_image.max()
    )

    if minimum < 0 or maximum > 255:

        raise ValueError(
            "Overflow/underflow detected after IDWT. "
            f"Image range is {minimum} to {maximum}."
        )

    # Safe conversion because values are already
    # guaranteed to be within [0,255]
    stego_image = idwt_image.astype(
        np.uint8
    )

    # -----------------------------------------------------
    # SAVE STEGO IMAGE
    # -----------------------------------------------------

    os.makedirs(
        os.path.dirname(stego_path)
        if os.path.dirname(stego_path)
        else ".",
        exist_ok=True
    )

    success = cv2.imwrite(
        stego_path,
        stego_image
    )

    if not success:

        raise ValueError(
            "Unable to save stego image."
        )

    return stego_image


# =========================================================
# COMPLETE TRANSMITTER IMAGE PROCESS
# =========================================================

def create_transmitter_images(
    image_path,
    roi_mask,
    non_roi_mask,
    LL,
    LH,
    HL,
    HH,
    HH_embedded,
    stego_path,
    output_dir="output"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # =====================================================
    # READ ORIGINAL MEDICAL IMAGE
    # =====================================================

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise ValueError(
            "Unable to read medical image."
        )

    image = image.astype(
        np.uint8
    )

    # =====================================================
    # SAVE ORIGINAL IMAGE
    # =====================================================

    original_path = os.path.join(
        output_dir,
        "original_scan.png"
    )

    cv2.imwrite(
        original_path,
        image
    )

    # =====================================================
    # VERIFY ORIGINAL DWT → IDWT
    # =====================================================

    reconstructed_original = integer_idwt2(
        LL,
        LH,
        HL,
        HH
    )

    if not np.array_equal(
        image.astype(np.int64),
        reconstructed_original
    ):

        raise ValueError(
            "Integer Haar DWT/IDWT is not exactly reversible."
        )

    # =====================================================
    # SAVE ALL FOUR DWT BANDS
    # =====================================================

    save_dwt_bands(
        LL,
        LH,
        HL,
        HH,
        output_dir
    )

    # =====================================================
    # SAVE ROI + NON-ROI
    # =====================================================

    save_roi_images(
        image,
        roi_mask,
        non_roi_mask,
        output_dir
    )

    # =====================================================
    # CREATE STEGO IMAGE
    # =====================================================

    stego_image = create_stego_image(
        LL,
        LH,
        HL,
        HH_embedded,
        stego_path
    )

    # =====================================================
    # SAVE IDWT IMAGE
    # =====================================================
    #
    # In this implementation, the IDWT image after
    # embedding is the stego image itself.
    #
    # We therefore create a separate copy for analysis.
    #

    idwt_path = os.path.join(
        output_dir,
        "idwt_image.png"
    )

    cv2.imwrite(
        idwt_path,
        stego_image
    )

    # =====================================================
    # RETURN OUTPUT INFORMATION
    # =====================================================

    return {

        "original":
            "original_scan.png",

        "dwt_ll":
            "dwt_ll.png",

        "dwt_lh":
            "dwt_lh.png",

        "dwt_hl":
            "dwt_hl.png",

        "dwt_hh":
            "dwt_hh.png",

        "roi":
            "roi.png",

        "non_roi":
            "non_roi.png",

        "idwt":
            "idwt_image.png",

        "stego":
            os.path.basename(
                stego_path
            )
    }