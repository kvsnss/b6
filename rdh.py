import json
import os

import cv2
import numpy as np


# =========================================================
# BYTE -> BITS
# =========================================================

def bytes_to_bits(data):
    """
    Convert bytes into a list of binary bits.
    """

    bits = []

    for byte in data:

        for i in range(7, -1, -1):

            bits.append(
                (byte >> i) & 1
            )

    return bits


# =========================================================
# BITS -> BYTES
# =========================================================

def bits_to_bytes(bits):
    """
    Convert a list of binary bits back into bytes.
    """

    if len(bits) % 8 != 0:

        raise ValueError(
            "Number of bits must be a multiple of 8."
        )

    result = bytearray()

    for i in range(
        0,
        len(bits),
        8
    ):

        value = 0

        for bit in bits[i:i + 8]:

            value = (
                value << 1
            ) | int(bit)

        result.append(
            value
        )

    return bytes(result)


# =========================================================
# RECONSTRUCT LOW-ROW COMPONENT
# =========================================================
#
# LL + HL are used to reconstruct the low-frequency
# row component that exists before the final horizontal
# inverse Haar operation.
#
# =========================================================

def reconstruct_low_rows(
    LL,
    HL
):

    LL = np.asarray(
        LL
    ).astype(
        np.int64
    )

    HL = np.asarray(
        HL
    ).astype(
        np.int64
    )

    half_height, half_width = LL.shape

    height = half_height * 2

    low_rows = np.zeros(
        (
            height,
            half_width
        ),
        dtype=np.int64
    )

    for c in range(
        half_width
    ):

        s = LL[:, c]

        d = HL[:, c]

        even = (
            s
            -
            np.floor_divide(
                d,
                2
            )
        )

        odd = (
            d
            +
            even
        )

        low_rows[
            0::2,
            c
        ] = even

        low_rows[
            1::2,
            c
        ] = odd

    return low_rows


# =========================================================
# RECONSTRUCT THE 2x2 BLOCK PRODUCED BY ONE HH VALUE
# =========================================================
#
# One HH coefficient influences one corresponding 2x2
# spatial block after inverse Haar reconstruction.
#
# This lets us test whether changing an HH coefficient
# by +1 or -1 would create pixel overflow/underflow.
#
# =========================================================

def reconstruct_hh_block(
    LH,
    low_rows,
    r,
    c,
    hh_value
):

    s = int(
        LH[r, c]
    )

    d = int(
        hh_value
    )

    # -----------------------------------------------------
    # Inverse vertical Haar
    # -----------------------------------------------------

    high_even = (
        s
        -
        np.floor_divide(
            d,
            2
        )
    )

    high_odd = (
        d
        +
        high_even
    )

    block = np.zeros(
        (2, 2),
        dtype=np.int64
    )

    # -----------------------------------------------------
    # First spatial row
    # -----------------------------------------------------

    low_value = int(
        low_rows[
            2 * r,
            c
        ]
    )

    even_pixel = (
        low_value
        -
        np.floor_divide(
            high_even,
            2
        )
    )

    odd_pixel = (
        high_even
        +
        even_pixel
    )

    block[0, 0] = even_pixel
    block[0, 1] = odd_pixel

    # -----------------------------------------------------
    # Second spatial row
    # -----------------------------------------------------

    low_value = int(
        low_rows[
            2 * r + 1,
            c
        ]
    )

    even_pixel = (
        low_value
        -
        np.floor_divide(
            high_odd,
            2
        )
    )

    odd_pixel = (
        high_odd
        +
        even_pixel
    )

    block[1, 0] = even_pixel
    block[1, 1] = odd_pixel

    return block


# =========================================================
# CHECK WHETHER AN HH COEFFICIENT VALUE IS SAFE
# =========================================================

def is_safe_hh_value(
    LH,
    low_rows,
    r,
    c,
    hh_value
):

    block = reconstruct_hh_block(
        LH,
        low_rows,
        r,
        c,
        hh_value
    )

    minimum = int(
        block.min()
    )

    maximum = int(
        block.max()
    )

    return (
        minimum >= 0
        and
        maximum <= 255
    )


# =========================================================
# FIND HH POSITIONS WHOSE COMPLETE 2x2 BLOCK IS NON-ROI
# =========================================================
#
# This is important.
#
# We do NOT want to modify an HH coefficient if its
# corresponding 2x2 spatial block overlaps the protected ROI.
#
# Therefore all four pixels must belong to Non-ROI.
#
# =========================================================

def get_nonroi_positions(
    non_roi_mask,
    hh_shape
):

    mask = np.asarray(
        non_roi_mask
    )

    if mask.ndim != 2:

        raise ValueError(
            "non_roi_mask must be a 2D array."
        )

    hh_rows = int(
        hh_shape[0]
    )

    hh_cols = int(
        hh_shape[1]
    )

    required_height = (
        hh_rows * 2
    )

    required_width = (
        hh_cols * 2
    )

    # -----------------------------------------------------
    # Resize mask to original image dimensions if needed
    # -----------------------------------------------------

    if mask.shape != (
        required_height,
        required_width
    ):

        mask = cv2.resize(
            mask.astype(np.uint8),
            (
                required_width,
                required_height
            ),
            interpolation=cv2.INTER_NEAREST
        )

    positions = []

    # -----------------------------------------------------
    # Check each HH coefficient's corresponding 2x2 block
    # -----------------------------------------------------

    for r in range(
        hh_rows
    ):

        y = 2 * r

        for c in range(
            hh_cols
        ):

            x = 2 * c

            block = mask[
                y:y + 2,
                x:x + 2
            ]

            if block.shape != (
                2,
                2
            ):

                continue

            # Entire 2x2 block must be Non-ROI
            if np.all(
                block > 0
            ):

                positions.append(
                    (
                        r,
                        c
                    )
                )

    return positions


# =========================================================
# RDH EMBEDDING
# =========================================================
#
# Method:
#
# Encrypted package
#       ↓
# Convert to bits
#       ↓
# Select HH coefficients in Non-ROI
#       ↓
# Required bit = coefficient LSB
#       ↓
# If different:
#       test coefficient - 1
#       test coefficient + 1
#       ↓
# choose only a safe value
#
# For exact recovery, the original HH coefficient value
# and its position are stored in metadata.
#
# =========================================================

def rdh_embed(
    HH,
    non_roi_mask,
    encrypted_package,
    LL=None,
    LH=None,
    HL=None,
    idwt_function=None
):

    # =====================================================
    # CHECK INPUTS
    # =====================================================

    if LL is None:

        raise ValueError(
            "LL is required for RDH embedding."
        )

    if LH is None:

        raise ValueError(
            "LH is required for RDH embedding."
        )

    if HL is None:

        raise ValueError(
            "HL is required for RDH embedding."
        )


    HH = np.asarray(
        HH
    ).astype(
        np.int64
    ).copy()


    if HH.ndim != 2:

        raise ValueError(
            "HH must be a 2D array."
        )


    # =====================================================
    # CONVERT ENCRYPTED PACKAGE TO BITS
    # =====================================================

    encrypted_package = bytes(
        encrypted_package
    )

    encrypted_bits = bytes_to_bits(
        encrypted_package
    )


    if len(encrypted_bits) == 0:

        raise ValueError(
            "Encrypted package is empty."
        )


    # =====================================================
    # RECONSTRUCT LOW ROWS
    # =====================================================

    low_rows = reconstruct_low_rows(
        LL,
        HL
    )


    # =====================================================
    # FIND VALID NON-ROI POSITIONS
    # =====================================================

    valid_positions = get_nonroi_positions(
        non_roi_mask,
        HH.shape
    )


    if len(valid_positions) == 0:

        raise ValueError(
            "No valid Non-ROI HH positions are available."
        )


    print(
        "Total Non-ROI HH positions:",
        len(valid_positions)
    )

    print(
        "Required embedding bits:",
        len(encrypted_bits)
    )


    # =====================================================
    # BASIC CAPACITY CHECK
    # =====================================================

    if len(valid_positions) < len(
        encrypted_bits
    ):

        raise ValueError(
            "Insufficient Non-ROI embedding capacity. "
            f"Required {len(encrypted_bits)} bits, "
            f"but only {len(valid_positions)} safe "
            "HH positions are available."
        )


    # =====================================================
    # EMBEDDING
    # =====================================================

    bit_index = 0

    used_positions = []

    original_values = []

    changed_count = 0

    unchanged_count = 0

    skipped_count = 0


    for r, c in valid_positions:

        # -------------------------------------------------
        # Stop after all payload bits are embedded
        # -------------------------------------------------

        if bit_index >= len(
            encrypted_bits
        ):

            break


        current_value = int(
            HH[r, c]
        )

        desired_bit = int(
            encrypted_bits[
                bit_index
            ]
        )

        current_bit = (
            current_value & 1
        )


        # =================================================
        # CASE 1: CURRENT VALUE ALREADY REPRESENTS BIT
        # =================================================

        if current_bit == desired_bit:

            used_positions.append(
                [
                    int(r),
                    int(c)
                ]
            )

            original_values.append(
                current_value
            )

            unchanged_count += 1

            bit_index += 1

            continue


        # =================================================
        # CASE 2: CHANGE PARITY
        # =================================================

        # Try -1 first
        candidate_minus = (
            current_value - 1
        )

        # Try +1 second
        candidate_plus = (
            current_value + 1
        )


        selected_value = None


        # -------------------------------------------------
        # Test -1
        # -------------------------------------------------

        if is_safe_hh_value(
            LH,
            low_rows,
            r,
            c,
            candidate_minus
        ):

            selected_value = (
                candidate_minus
            )


        # -------------------------------------------------
        # If -1 is unsafe, test +1
        # -------------------------------------------------

        elif is_safe_hh_value(
            LH,
            low_rows,
            r,
            c,
            candidate_plus
        ):

            selected_value = (
                candidate_plus
            )


        # -------------------------------------------------
        # Neither option is safe
        # -------------------------------------------------

        else:

            skipped_count += 1

            continue


        # =================================================
        # APPLY MODIFICATION
        # =================================================

        HH[r, c] = selected_value

        used_positions.append(
            [
                int(r),
                int(c)
            ]
        )

        original_values.append(
            current_value
        )

        changed_count += 1

        bit_index += 1


    # =====================================================
    # FINAL CAPACITY CHECK
    # =====================================================

    if bit_index < len(
        encrypted_bits
    ):

        raise ValueError(
            "RDH embedding failed because there are not "
            "enough overflow-safe Non-ROI HH positions. "
            f"Required bits: {len(encrypted_bits)}, "
            f"embedded bits: {bit_index}, "
            f"available Non-ROI HH positions: "
            f"{len(valid_positions)}, "
            f"skipped unsafe positions: {skipped_count}."
        )


    # =====================================================
    # FINAL IDWT SAFETY CHECK
    # =====================================================

    if idwt_function is not None:

        reconstructed = idwt_function(
            LL,
            LH,
            HL,
            HH
        )

        minimum = int(
            reconstructed.min()
        )

        maximum = int(
            reconstructed.max()
        )

        print(
            "RDH final IDWT range:",
            minimum,
            "to",
            maximum
        )


        if minimum < 0 or maximum > 255:

            raise ValueError(
                "Final RDH image contains pixel overflow/"
                "underflow. "
                f"Range: {minimum} to {maximum}."
            )


    # =====================================================
    # CREATE METADATA
    # =====================================================
    #
    # We store the ORIGINAL coefficient values.
    # This gives exact HH recovery at the receiver.
    #
    # =====================================================

    metadata = {

        "method":
            "Integer Haar DWT HH reversible embedding",

        "embedding_band":
            "HH",

        "embedding_region":
            "Non-ROI",

        "payload_bytes":
            int(
                len(encrypted_package)
            ),

        "bit_length":
            int(
                len(encrypted_bits)
            ),

        "hh_rows":
            int(
                HH.shape[0]
            ),

        "hh_cols":
            int(
                HH.shape[1]
            ),

        "num_positions":
            int(
                len(used_positions)
            ),

        "changed_coefficients":
            int(
                changed_count
            ),

        "unchanged_coefficients":
            int(
                unchanged_count
            ),

        "skipped_unsafe_positions":
            int(
                skipped_count
            ),

        "positions":
            used_positions,

        "original_values":
            original_values
    }


    # =====================================================
    # PRINT RESULT
    # =====================================================

    print()
    print(
        "========================================"
    )

    print(
        "RDH EMBEDDING SUCCESSFUL"
    )

    print(
        "========================================"
    )

    print(
        "Payload bytes:",
        len(encrypted_package)
    )

    print(
        "Payload bits:",
        len(encrypted_bits)
    )

    print(
        "Embedded positions:",
        len(used_positions)
    )

    print(
        "Changed coefficients:",
        changed_count
    )

    print(
        "Unchanged coefficients:",
        unchanged_count
    )

    print(
        "Skipped unsafe positions:",
        skipped_count
    )

    print(
        "========================================"
    )


    return (
        HH,
        metadata
    )


# =========================================================
# SAVE METADATA
# =========================================================

def save_metadata(
    metadata,
    path
):

    directory = os.path.dirname(
        path
    )

    if directory:

        os.makedirs(
            directory,
            exist_ok=True
        )


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )


# =========================================================
# LOAD METADATA
# =========================================================

def load_metadata(
    path
):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(
            file
        )


# =========================================================
# RDH EXTRACTION
# =========================================================
#
# This function:
#
# 1. Reads the modified HH.
# 2. Extracts LSB from every used position.
# 3. Restores the exact ORIGINAL HH coefficient value
#    using metadata.
#
# =========================================================

def rdh_extract(
    HH_embedded,
    metadata
):

    HH_embedded = np.asarray(
        HH_embedded
    ).astype(
        np.int64
    ).copy()


    positions = metadata.get(
        "positions",
        []
    )

    original_values = metadata.get(
        "original_values",
        []
    )

    bit_length = int(
        metadata.get(
            "bit_length",
            0
        )
    )


    # =====================================================
    # VALIDATE METADATA
    # =====================================================

    if len(positions) != len(
        original_values
    ):

        raise ValueError(
            "Invalid RDH metadata: number of positions "
            "does not match number of original values."
        )


    if bit_length > len(
        positions
    ):

        raise ValueError(
            "Invalid RDH metadata: bit length is larger "
            "than the number of stored positions."
        )


    extracted_bits = []

    HH_recovered = HH_embedded.copy()


    # =====================================================
    # EXTRACT
    # =====================================================

    for i in range(
        bit_length
    ):

        r = int(
            positions[i][0]
        )

        c = int(
            positions[i][1]
        )


        current_value = int(
            HH_recovered[r, c]
        )


        # Extract hidden bit
        extracted_bit = (
            current_value & 1
        )

        extracted_bits.append(
            extracted_bit
        )


        # Restore EXACT original coefficient
        HH_recovered[r, c] = int(
            original_values[i]
        )


    # =====================================================
    # CONVERT BITS TO BYTES
    # =====================================================

    encrypted_package = bits_to_bytes(
        extracted_bits
    )


    return (
        encrypted_package,
        HH_recovered
    )


# =========================================================
# SAVE EXTRACTED DATA / RECOVERED HH
# =========================================================

def extract_and_recover_HH(
    HH_embedded,
    metadata
):

    return rdh_extract(
        HH_embedded,
        metadata
    )