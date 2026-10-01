import cv2
import numpy as np

from dwt import integer_dwt2, integer_idwt2
from rdh import rdh_extract


# =========================================================
# EXTRACT PAYLOAD FROM STEGO IMAGE
# =========================================================

def extract_payload(
    stego_path,
    metadata,
    recovered_image_path=None
):

    # -----------------------------------------------------
    # READ STEGO IMAGE
    # -----------------------------------------------------

    image = cv2.imread(
        stego_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            "Unable to read received stego image."
        )

    # -----------------------------------------------------
    # CHECK IMAGE DIMENSIONS
    # -----------------------------------------------------

    height, width = image.shape

    if height % 2 != 0:
        raise ValueError(
            "Stego image height must be even."
        )

    if width % 2 != 0:
        raise ValueError(
            "Stego image width must be even."
        )

    # -----------------------------------------------------
    # VALIDATE METADATA
    # -----------------------------------------------------

    if not isinstance(metadata, dict):
        raise ValueError(
            "Invalid RDH metadata."
        )

    positions = metadata.get(
        "positions"
    )

    original_values = metadata.get(
        "original_values"
    )

    bit_length = metadata.get(
        "bit_length"
    )

    payload_bytes = metadata.get(
        "payload_bytes"
    )

    if positions is None:
        raise ValueError(
            "RDH metadata does not contain "
            "'positions'."
        )

    if original_values is None:
        raise ValueError(
            "RDH metadata does not contain "
            "'original_values'."
        )

    if bit_length is None:
        raise ValueError(
            "RDH metadata does not contain "
            "'bit_length'."
        )

    if payload_bytes is None:
        raise ValueError(
            "RDH metadata does not contain "
            "'payload_bytes'."
        )

    # -----------------------------------------------------
    # CHECK TYPES
    # -----------------------------------------------------

    if not isinstance(
        positions,
        list
    ):
        raise ValueError(
            "'positions' must be a list."
        )

    if not isinstance(
        original_values,
        list
    ):
        raise ValueError(
            "'original_values' must be a list."
        )

    # -----------------------------------------------------
    # CHECK METADATA LENGTH
    # -----------------------------------------------------

    if len(positions) != len(
        original_values
    ):
        raise ValueError(
            "RDH metadata is inconsistent: "
            "positions and original_values "
            "have different lengths."
        )

    # -----------------------------------------------------
    # CHECK BIT LENGTH
    # -----------------------------------------------------

    bit_length = int(
        bit_length
    )

    payload_bytes = int(
        payload_bytes
    )

    if bit_length != len(
        positions
    ):
        raise ValueError(
            "RDH metadata is inconsistent: "
            "bit_length does not match "
            "the number of positions."
        )

    # -----------------------------------------------------
    # CHECK PAYLOAD SIZE
    # -----------------------------------------------------

    expected_bit_length = (
        payload_bytes * 8
    )

    if bit_length != expected_bit_length:
        raise ValueError(
            "RDH metadata is inconsistent: "
            "bit_length does not equal "
            "payload_bytes × 8."
        )

    # -----------------------------------------------------
    # CHECK AES PACKAGE SIZE
    # -----------------------------------------------------

    if payload_bytes < 32:
        raise ValueError(
            "Encrypted package is too small. "
            "It must contain at least "
            "16-byte salt + 16-byte IV."
        )

    # -----------------------------------------------------
    # INTEGER DWT
    # -----------------------------------------------------

    LL, LH, HL, HH_embedded = (
        integer_dwt2(image)
    )

    print()
    print("========================================")
    print("RDH EXTRACTION")
    print("========================================")

    print(
        "Stego image size:",
        width,
        "x",
        height
    )

    print(
        "HH shape:",
        HH_embedded.shape
    )

    print(
        "Payload bytes:",
        payload_bytes
    )

    print(
        "Embedded bits:",
        bit_length
    )

    print(
        "Metadata positions:",
        len(positions)
    )

    # -----------------------------------------------------
    # CHECK HH DIMENSIONS
    # -----------------------------------------------------

    hh_rows = metadata.get(
        "hh_rows"
    )

    hh_cols = metadata.get(
        "hh_cols"
    )

    if hh_rows is not None:

        if int(hh_rows) != HH_embedded.shape[0]:
            raise ValueError(
                "HH row count mismatch. "
                "The stego image does not match "
                "the RDH metadata."
            )

    if hh_cols is not None:

        if int(hh_cols) != HH_embedded.shape[1]:
            raise ValueError(
                "HH column count mismatch. "
                "The stego image does not match "
                "the RDH metadata."
            )

    # -----------------------------------------------------
    # RDH EXTRACTION
    # -----------------------------------------------------

    recovered_data, recovered_HH = (
        rdh_extract(
            HH_embedded,
            metadata
        )
    )

    if recovered_data is None:
        raise ValueError(
            "RDH extraction returned no "
            "encrypted data."
        )

    # -----------------------------------------------------
    # CHECK EXTRACTED PAYLOAD
    # -----------------------------------------------------

    extracted_size = len(
        recovered_data
    )

    if extracted_size != payload_bytes:

        raise ValueError(
            "Extracted payload size mismatch. "
            f"Expected {payload_bytes} bytes, "
            f"but received {extracted_size} bytes."
        )

    print(
        "Extracted encrypted package:",
        extracted_size,
        "bytes"
    )

    # -----------------------------------------------------
    # CHECK MINIMUM AES PACKAGE
    # -----------------------------------------------------

    if extracted_size < 32:
        raise ValueError(
            "Extracted package is too small."
        )

    # -----------------------------------------------------
    # INVERSE DWT
    # -----------------------------------------------------

    recovered_image = integer_idwt2(
        LL,
        LH,
        HL,
        recovered_HH
    )

    # -----------------------------------------------------
    # CHECK IMAGE SHAPE
    # -----------------------------------------------------

    if recovered_image.shape != image.shape:

        raise ValueError(
            "Recovered image dimensions do not "
            "match the original stego image dimensions."
        )

    # -----------------------------------------------------
    # CHECK IMAGE RANGE
    # -----------------------------------------------------

    minimum = recovered_image.min()
    maximum = recovered_image.max()

    if minimum < 0 or maximum > 255:

        raise ValueError(
            "Recovered image contains invalid "
            "pixel values."
        )

    recovered_image = recovered_image.astype(
        np.uint8
    )

    # -----------------------------------------------------
    # SAVE RECOVERED IMAGE
    # -----------------------------------------------------

    if recovered_image_path:

        success = cv2.imwrite(
            recovered_image_path,
            recovered_image
        )

        if not success:
            raise ValueError(
                "Failed to save recovered image."
            )

    # -----------------------------------------------------
    # SUCCESS
    # -----------------------------------------------------

    print()
    print("========================================")
    print("RDH EXTRACTION SUCCESSFUL")
    print("========================================")

    print(
        "Recovered image:",
        recovered_image_path
    )

    return (
        recovered_data,
        recovered_image
    )
