import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# =========================================================
# IMAGE QUALITY METRICS
# Original Image vs Stego Image
# =========================================================

def calculate_metrics(
    original_path,
    stego_path
):

    # -----------------------------------------------------
    # READ ORIGINAL IMAGE
    # -----------------------------------------------------

    original = cv2.imread(
        original_path,
        cv2.IMREAD_GRAYSCALE
    )

    # -----------------------------------------------------
    # READ STEGO IMAGE
    # -----------------------------------------------------

    stego = cv2.imread(
        stego_path,
        cv2.IMREAD_GRAYSCALE
    )

    # -----------------------------------------------------
    # CHECK IMAGES
    # -----------------------------------------------------

    if original is None:

        raise ValueError(
            "Original image not found."
        )

    if stego is None:

        raise ValueError(
            "Stego image not found."
        )

    # -----------------------------------------------------
    # CHECK IMAGE SIZE
    # -----------------------------------------------------

    if original.shape != stego.shape:

        raise ValueError(
            "Original and stego images "
            "must have the same dimensions."
        )

    # -----------------------------------------------------
    # CONVERT TO FLOAT
    # -----------------------------------------------------

    original_float = (
        original.astype(np.float64)
    )

    stego_float = (
        stego.astype(np.float64)
    )

    # =====================================================
    # MSE
    # =====================================================

    mse = np.mean(
        (
            original_float -
            stego_float
        ) ** 2
    )

    # =====================================================
    # PSNR
    # =====================================================

    if mse == 0:

        psnr = float("inf")

    else:

        psnr = 10 * np.log10(
            (255 ** 2) / mse
        )

    # =====================================================
    # SSIM
    # =====================================================

    ssim_value = ssim(
        original,
        stego,
        data_range=255
    )

    return (
        mse,
        psnr,
        ssim_value
    )


# =========================================================
# REVERSIBILITY VERIFICATION
# Original Image vs Recovered Image
# =========================================================

def verify_recovery(
    original_path,
    recovered_path
):

    # Read images

    original = cv2.imread(
        original_path,
        cv2.IMREAD_GRAYSCALE
    )

    recovered = cv2.imread(
        recovered_path,
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:

        raise ValueError(
            "Original image not found."
        )

    if recovered is None:

        raise ValueError(
            "Recovered image not found."
        )

    if original.shape != recovered.shape:

        raise ValueError(
            "Original and recovered images "
            "must have the same dimensions."
        )

    # Difference

    difference = (
        original.astype(np.int64)
        -
        recovered.astype(np.int64)
    )

    # Maximum error

    maximum_error = np.max(
        np.abs(difference)
    )

    # MSE

    recovery_mse = np.mean(
        difference ** 2
    )

    # Exact equality

    exact_recovery = np.array_equal(
        original,
        recovered
    )

    return (
        maximum_error,
        recovery_mse,
        exact_recovery
    )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    original_path = (
        "output/original_scan.png"
    )

    stego_path = (
        "output/stego_handfracture.png"
    )

    # -----------------------------------------------------
    # TRANSMITTER QUALITY EVALUATION
    # -----------------------------------------------------

    mse, psnr, ssim_value = calculate_metrics(
        original_path,
        stego_path
    )

    print()
    print(
        "========================================"
    )
    print(
        "TRANSMITTER IMAGE QUALITY"
    )
    print(
        "========================================"
    )

    print(
        "MSE  :",
        mse
    )

    print(
        "PSNR :",
        psnr,
        "dB"
    )

    print(
        "SSIM :",
        ssim_value
    )

    print(
        "========================================"
    )