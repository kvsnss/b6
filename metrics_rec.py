import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ============================================================
# RECEIVER-SIDE IMAGE RECOVERY METRICS
# ============================================================
# These metrics compare:
#
#       ORIGINAL IMAGE  vs  RECOVERED IMAGE
#
# They are used to verify reversible image recovery.
# ============================================================

def calculate_receiver_metrics(
    original_path,
    recovered_path,
    stego_path=None
):
    """
    Calculate receiver-side reversibility metrics.

    Parameters
    ----------
    original_path : str
        Path to the original medical image.

    recovered_path : str
        Path to the image recovered after RDH extraction
        and inverse DWT.

    stego_path : str, optional
        Path to the stego image.
        This is accepted for reference but is NOT used
        for the recovery metrics.

    Returns
    -------
    dict
        MSE, PSNR, SSIM, maximum error, exact recovery,
        different pixels, total pixels and pixel error rate.
    """

    # ========================================================
    # LOAD IMAGES
    # ========================================================

    original = cv2.imread(
        original_path,
        cv2.IMREAD_GRAYSCALE
    )

    recovered = cv2.imread(
        recovered_path,
        cv2.IMREAD_GRAYSCALE
    )


    # ========================================================
    # VALIDATE IMAGES
    # ========================================================

    if original is None:
        raise ValueError(
            f"Original image not found: {original_path}"
        )

    if recovered is None:
        raise ValueError(
            f"Recovered image not found: {recovered_path}"
        )

    if original.shape != recovered.shape:
        raise ValueError(
            "Original and recovered images must have "
            "the same dimensions."
        )


    # ========================================================
    # PIXEL DIFFERENCE
    # ========================================================

    # int16 prevents unsigned integer subtraction problems.
    diff = (
        original.astype(np.int16)
        - recovered.astype(np.int16)
    )


    # ========================================================
    # MSE
    # ========================================================

    recovery_mse = float(
        np.mean(
            diff.astype(np.float64) ** 2
        )
    )


    # ========================================================
    # MAXIMUM PIXEL ERROR
    # ========================================================

    max_error = int(
        np.max(
            np.abs(diff)
        )
    )


    # ========================================================
    # EXACT RECOVERY
    # ========================================================

    exact_recovery = bool(
        np.array_equal(
            original,
            recovered
        )
    )


    # ========================================================
    # DIFFERENT PIXELS
    # ========================================================

    different_pixels = int(
        np.count_nonzero(
            original != recovered
        )
    )


    # ========================================================
    # TOTAL PIXELS
    # ========================================================

    total_pixels = int(
        original.size
    )


    # ========================================================
    # PIXEL ERROR RATE
    # ========================================================

    if total_pixels == 0:
        pixel_error_rate = 0.0

    else:
        pixel_error_rate = float(
            different_pixels / total_pixels
        )


    # ========================================================
    # PSNR
    # ========================================================

    if recovery_mse == 0:

        recovery_psnr = float("inf")

    else:

        recovery_psnr = float(
            10
            * np.log10(
                (255.0 ** 2)
                / recovery_mse
            )
        )


    # ========================================================
    # SSIM
    # ========================================================

    recovery_ssim = float(
        ssim(
            original,
            recovered,
            data_range=255
        )
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "mse": recovery_mse,
        "psnr": recovery_psnr,
        "ssim": recovery_ssim,
        "max_error": max_error,
        "exact_recovery": exact_recovery,
        "different_pixels": different_pixels,
        "total_pixels": total_pixels,
        "pixel_error_rate": pixel_error_rate
    }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    original_path = os.path.join(
        "received",
        "original_image.png"
    )

    stego_path = os.path.join(
        "received",
        "stego_image.png"
    )

    recovered_path = os.path.join(
        "recovered",
        "recovered_stego_image.png"
    )


    # ========================================================
    # CHECK STEGO IMAGE
    # ========================================================

    if not os.path.exists(stego_path):

        print(
            f"Warning: Stego image not found: "
            f"{stego_path}"
        )


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    result = calculate_receiver_metrics(
        original_path,
        recovered_path,
        stego_path
    )


    # ========================================================
    # FORMAT PSNR
    # ========================================================

    if np.isinf(result["psnr"]):

        psnr_str = "INF"

    else:

        psnr_str = (
            f"{result['psnr']:.2f} dB"
        )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print()
    print("=" * 50)
    print("RECEIVER REVERSIBILITY CHECK")
    print("=" * 50)

    print(
        f"MSE (Original vs Recovered) : "
        f"{result['mse']}"
    )

    print(
        f"PSNR                       : "
        f"{psnr_str}"
    )

    print(
        f"SSIM                       : "
        f"{result['ssim']:.6f}"
    )

    print(
        f"Maximum Pixel Error        : "
        f"{result['max_error']}"
    )

    print(
        f"Exact Recovery             : "
        f"{result['exact_recovery']}"
    )

    print(
        f"Different Pixels           : "
        f"{result['different_pixels']}"
    )

    print(
        f"Total Pixels               : "
        f"{result['total_pixels']}"
    )

    print(
        f"Pixel Error Rate           : "
        f"{result['pixel_error_rate']}"
    )

    print("=" * 50)
