import os
import base64
import json
import uuid

import cv2
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from werkzeug.utils import secure_filename

from aes import (
    encrypt_data,
    decrypt_data
)

from roi import detect_roi

from dwt import (
    integer_dwt2,
    integer_idwt2
)

from rdh import (
    rdh_embed,
    save_metadata
)

from metrics import calculate_metrics


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(
    __name__,
    template_folder=BASE_DIR,
    static_folder=BASE_DIR
)


# =========================================================
# FOLDERS
# =========================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "output"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# =========================================================
# LOCAL DEMO STORAGE
# =========================================================

cipher_store = None
salt_store = None
iv_store = None
patient_store = None

rdh_metadata_store = None

stego_filename_store = None


# =========================================================
# NORMALIZE DWT BAND FOR SAVING
# =========================================================
#
# IMPORTANT:
# This is ONLY for creating visual DWT images.
# The normalized images are never used in reconstruction.
#
# =========================================================

def normalize_image(data):

    data = np.asarray(
        data
    ).astype(
        np.float64
    )

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
        255.0
    )

    return normalized.astype(
        np.uint8
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# STYLE
# =========================================================

@app.route("/style.css")
def style():

    return send_from_directory(
        BASE_DIR,
        "style.css"
    )


# =========================================================
# SERVE OUTPUT FILES
# =========================================================

@app.route(
    "/output/<path:filename>"
)
def output_file(filename):

    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


# =========================================================
# ENCRYPTION + ROI + DWT + RDH + IDWT
# =========================================================

@app.route(
    "/encrypt",
    methods=["POST"]
)
def encrypt():

    global cipher_store
    global salt_store
    global iv_store
    global patient_store
    global rdh_metadata_store
    global stego_filename_store


    print()
    print("========================================")
    print("TRANSMITTER PROCESS STARTED")
    print("========================================")


    # =====================================================
    # 1. PATIENT DATA
    # =====================================================

    name = request.form.get(
        "name",
        ""
    ).strip()

    age = request.form.get(
        "age",
        ""
    ).strip()

    gender = request.form.get(
        "gender",
        ""
    ).strip()

    blood = request.form.get(
        "blood",
        ""
    ).strip()

    disease = request.form.get(
        "disease",
        ""
    ).strip()

    doctor = request.form.get(
        "doctor",
        ""
    ).strip()

    hospital = request.form.get(
        "hospital",
        ""
    ).strip()

    password = request.form.get(
        "password",
        ""
    )


    # =====================================================
    # 2. PASSWORD CHECK
    # =====================================================

    if not password:

        return render_template(
            "result.html",
            error="Please enter the encryption password."
        )


    # =====================================================
    # 3. MEDICAL IMAGE
    # =====================================================

    image_file = request.files.get(
        "medical_image"
    )


    if image_file is None:

        return render_template(
            "result.html",
            error="Please upload a medical image."
        )


    if image_file.filename == "":

        return render_template(
            "result.html",
            error="No medical image was selected."
        )


    # =====================================================
    # 4. SECURE FILE NAME
    # =====================================================

    original_filename = secure_filename(
        image_file.filename
    )


    if not original_filename:

        return render_template(
            "result.html",
            error="Invalid medical image filename."
        )


    # =====================================================
    # 5. UNIQUE PROCESS ID
    # =====================================================

    process_id = uuid.uuid4().hex[:12]

    original_name = os.path.splitext(
        original_filename
    )[0]


    # =====================================================
    # 6. PATIENT DATA STRING
    # =====================================================

    patient_data = (
        f"Patient Name : {name}\n"
        f"Age          : {age}\n"
        f"Gender       : {gender}\n"
        f"Blood Group  : {blood}\n"
        f"Disease      : {disease}\n"
        f"Doctor       : {doctor}\n"
        f"Hospital     : {hospital}"
    )


    print()
    print("Patient data prepared.")


    # =====================================================
    # 7. SAVE UPLOADED IMAGE
    # =====================================================

    input_filename = (
        process_id
        + "_"
        + original_filename
    )

    image_path = os.path.join(
        UPLOAD_FOLDER,
        input_filename
    )


    try:

        image_file.save(
            image_path
        )

    except Exception as e:

        return render_template(
            "result.html",
            error=(
                "Unable to save uploaded image: "
                + str(e)
            )
        )


    print(
        "Uploaded image:",
        image_path
    )


    # =====================================================
    # 8. READ MEDICAL IMAGE
    # =====================================================

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )


    if image is None:

        return render_template(
            "result.html",
            error=(
                "Unable to read the uploaded medical image."
            )
        )


    image = image.astype(
        np.uint8
    )


    # =====================================================
    # 9. IMAGE SIZE
    # =====================================================

    height, width = image.shape


    print(
        "Image size:",
        width,
        "x",
        height
    )


    if height % 2 != 0:

        return render_template(
            "result.html",
            error=(
                "Image height must be even for "
                "Integer Haar DWT. "
                f"Current height: {height}"
            )
        )


    if width % 2 != 0:

        return render_template(
            "result.html",
            error=(
                "Image width must be even for "
                "Integer Haar DWT. "
                f"Current width: {width}"
            )
        )


    # =====================================================
    # STEP 1
    # AES-256 ENCRYPTION
    # =====================================================

    try:

        salt, iv, ciphertext = encrypt_data(
            patient_data,
            password
        )

    except Exception as e:

        return render_template(
            "result.html",
            error=(
                "AES-256 encryption failed: "
                + str(e)
            )
        )


    print(
        "AES-256 encryption successful."
    )


    # =====================================================
    # STORE AES DATA
    # =====================================================

    cipher_store = ciphertext
    salt_store = salt
    iv_store = iv
    patient_store = patient_data


    # =====================================================
    # ENCRYPTED PACKAGE
    #
    # SALT + IV + CIPHERTEXT
    # =====================================================

    encrypted_package = (
        salt
        +
        iv
        +
        ciphertext
    )


    print(
        "Encrypted package size:",
        len(encrypted_package),
        "bytes"
    )


    # =====================================================
    # BASE64 ENCODED TEXT
    # =====================================================

    encrypted_text = base64.b64encode(
        encrypted_package
    ).decode()


    # =====================================================
    # STEP 2
    # ROI DETECTION
    # =====================================================

    try:

        (
            roi_original,
            roi_mask,
            non_roi_mask,
            roi_image,
            non_roi_image
        ) = detect_roi(
            image_path
        )

    except Exception as e:

        return render_template(
            "result.html",
            error=(
                "ROI detection failed: "
                + str(e)
            )
        )


    print(
        "ROI detection successful."
    )


    # =====================================================
    # VERIFY IMAGE
    # =====================================================

    if not np.array_equal(
        image,
        roi_original
    ):

        return render_template(
            "result.html",
            error=(
                "ROI processing returned an "
                "inconsistent original image."
            )
        )


    # =====================================================
    # SAVE ORIGINAL IMAGE
    # =====================================================

    original_output_filename = (
        "original_"
        + process_id
        + ".png"
    )

    original_output_path = os.path.join(
        OUTPUT_FOLDER,
        original_output_filename
    )


    if not cv2.imwrite(
        original_output_path,
        image
    ):

        return render_template(
            "result.html",
            error="Unable to save original image."
        )


    # =====================================================
    # SAVE ROI IMAGE
    # =====================================================
    #
    # Stored in output/.
    # NOT displayed on website.
    #
    # =====================================================

    roi_filename = (
        "roi_"
        + process_id
        + ".png"
    )

    roi_path = os.path.join(
        OUTPUT_FOLDER,
        roi_filename
    )


    if not cv2.imwrite(
        roi_path,
        roi_image
    ):

        return render_template(
            "result.html",
            error="Unable to save ROI image."
        )


    # =====================================================
    # SAVE NON-ROI IMAGE
    # =====================================================
    #
    # Stored in output/.
    # NOT displayed on website.
    #
    # =====================================================

    non_roi_filename = (
        "non_roi_"
        + process_id
        + ".png"
    )

    non_roi_path = os.path.join(
        OUTPUT_FOLDER,
        non_roi_filename
    )


    if not cv2.imwrite(
        non_roi_path,
        non_roi_image
    ):

        return render_template(
            "result.html",
            error="Unable to save Non-ROI image."
        )


    print(
        "ROI image saved:",
        roi_path
    )

    print(
        "Non-ROI image saved:",
        non_roi_path
    )


    # =====================================================
    # STEP 3
    # INTEGER HAAR DWT
    # =====================================================

    try:

        LL, LH, HL, HH = integer_dwt2(
            image
        )

    except Exception as e:

        return render_template(
            "result.html",
            error=(
                "Integer Haar DWT failed: "
                + str(e)
            )
        )


    print(
        "Integer Haar DWT successful."
    )


    # =====================================================
    # VERIFY DWT / IDWT REVERSIBILITY
    # =====================================================

    try:

        test_reconstructed = integer_idwt2(
            LL,
            LH,
            HL,
            HH
        )

    except Exception as e:

        return render_template(
            "result.html",
            error=(
                "Integer Haar IDWT verification failed: "
                + str(e)
            )
        )


    if not np.array_equal(
        image.astype(np.int64),
        test_reconstructed
    ):

        return render_template(
            "result.html",
            error=(
                "Integer Haar DWT/IDWT is not "
                "exactly reversible."
            )
        )


    print(
        "DWT/IDWT exact reversibility verified."
    )


    # =====================================================
    # SAVE DWT LL
    # =====================================================

    dwt_ll_filename = (
        "dwt_ll_"
        + process_id
        + ".png"
    )

    dwt_ll_path = os.path.join(
        OUTPUT_FOLDER,
        dwt_ll_filename
    )


    cv2.imwrite(
        dwt_ll_path,
        normalize_image(LL)
    )


    # =====================================================
    # SAVE DWT LH
    # =====================================================

    dwt_lh_filename = (
        "dwt_lh_"
        + process_id
        + ".png"
    )

    dwt_lh_path = os.path.join(
        OUTPUT_FOLDER,
        dwt_lh_filename
    )


    cv2.imwrite(
        dwt_lh_path,
        normalize_image(LH)
    )


    # =====================================================
    # SAVE DWT HL
    # =====================================================

    dwt_hl_filename = (
        "dwt_hl_"
        + process_id
        + ".png"
    )

    dwt_hl_path = os.path.join(
        OUTPUT_FOLDER,
        dwt_hl_filename
    )


    cv2.imwrite(
        dwt_hl_path,
        normalize_image(HL)
    )


    # =====================================================
    # SAVE DWT HH
    # =====================================================

    dwt_hh_filename = (
        "dwt_hh_"
        + process_id
        + ".png"
    )

    dwt_hh_path = os.path.join(
        OUTPUT_FOLDER,
        dwt_hh_filename
    )


    cv2.imwrite(
        dwt_hh_path,
        normalize_image(HH)
    )


    print(
        "LL, LH, HL and HH bands saved."
    )


    # =====================================================
    # STEP 4
    # RDH EMBEDDING
    # =====================================================

    try:

        HH_embedded, rdh_metadata = rdh_embed(

            HH,

            non_roi_mask,

            encrypted_package,

            LL=LL,

            LH=LH,

            HL=HL,

            idwt_function=integer_idwt2

        )

    except Exception as e:

        print()
        print("========================================")
        print("RDH EMBEDDING FAILED")
        print("========================================")
        print(str(e))
        print("========================================")

        return render_template(
            "result.html",
            error=(
                "RDH embedding failed: "
                + str(e)
            )
        )


    print(
        "RDH embedding successful."
    )


    rdh_metadata_store = rdh_metadata


    # =====================================================
    # SAVE RDH METADATA
    # =====================================================

    metadata_filename = (
        "rdh_metadata_"
        + process_id
        + ".json"
    )

    metadata_path = os.path.join(
        OUTPUT_FOLDER,
        metadata_filename
    )


    try:

        save_metadata(
            rdh_metadata,
            metadata_path
        )

    except Exception as e:

        return render_template(
            "result.html",
            error=(
                "Unable to save RDH metadata: "
                + str(e)
            )
        )


    print(
        "RDH metadata saved:",
        metadata_path
    )


    # =====================================================
    # STEP 5
    # INTEGER IDWT
    # =====================================================

    try:

        idwt_image = integer_idwt2(
            LL,
            LH,
            HL,
            HH_embedded
        )

    except Exception as e:

        return render_template(
            "result.html",
            error=(
                "Integer IDWT failed: "
                + str(e)
            )
        )


    print(
        "Integer IDWT successful."
    )


    # =====================================================
    # CHECK PIXEL RANGE
    # =====================================================

    minimum = int(
        idwt_image.min()
    )

    maximum = int(
        idwt_image.max()
    )


    print(
        "Final IDWT range:",
        minimum,
        "to",
        maximum
    )


    if minimum < 0 or maximum > 255:

        return render_template(
            "result.html",
            error=(
                "Stego image contains invalid "
                "pixel values. "
                f"Range: {minimum} to {maximum}."
            )
        )


    # =====================================================
    # SAFE CONVERSION
    # =====================================================

    stego_image = idwt_image.astype(
        np.uint8
    )


    # =====================================================
    # STEP 6
    # SAVE IDWT IMAGE
    # =====================================================

    idwt_filename = (
        "idwt_"
        + process_id
        + ".png"
    )

    idwt_path = os.path.join(
        OUTPUT_FOLDER,
        idwt_filename
    )


    if not cv2.imwrite(
        idwt_path,
        stego_image
    ):

        return render_template(
            "result.html",
            error="Unable to save IDWT image."
        )


    print(
        "IDWT image saved:",
        idwt_path
    )


    # =====================================================
    # STEP 7
    # SAVE FINAL STEGO IMAGE
    # =====================================================

    stego_filename = (
        "stego_"
        + process_id
        + "_"
        + original_name
        + ".png"
    )

    stego_path = os.path.join(
        OUTPUT_FOLDER,
        stego_filename
    )


    if not cv2.imwrite(
        stego_path,
        stego_image
    ):

        return render_template(
            "result.html",
            error="Unable to save stego image."
        )


    # =====================================================
    # VERIFY STEGO FILE
    # =====================================================

    if not os.path.isfile(
        stego_path
    ):

        return render_template(
            "result.html",
            error=(
                "Stego image file does not exist "
                "after saving."
            )
        )


    stego_filename_store = stego_filename


    print()
    print("========================================")
    print("STEGO IMAGE SUCCESSFULLY CREATED")
    print("========================================")

    print(
        "Stego filename:",
        stego_filename
    )

    print(
        "Stego path:",
        os.path.abspath(
            stego_path
        )
    )

    print(
        "Stego file size:",
        os.path.getsize(
            stego_path
        ),
        "bytes"
    )

    print("========================================")


    # =====================================================
    # STEP 8
    # IMAGE QUALITY METRICS
    #
    # Original vs Stego
    # =====================================================

    mse = None
    psnr = None
    ssim_value = None


    try:

        metrics_result = calculate_metrics(
            original_output_path,
            stego_path
        )


        if isinstance(
            metrics_result,
            dict
        ):

            mse = metrics_result.get(
                "MSE"
            )

            psnr = metrics_result.get(
                "PSNR"
            )

            ssim_value = metrics_result.get(
                "SSIM"
            )


        elif (
            isinstance(
                metrics_result,
                (tuple, list)
            )
            and
            len(metrics_result) >= 3
        ):

            mse = metrics_result[0]

            psnr = metrics_result[1]

            ssim_value = metrics_result[2]


    except Exception as e:

        print(
            "Metric calculation failed:",
            str(e)
        )


    # =====================================================
    # STEP 9
    # EXACT RECOVERY VERIFICATION
    # =====================================================
    #
    # IMPORTANT:
    #
    # New RDH metadata uses:
    #
    # positions
    # original_values
    #
    # NOT original_lsb.
    #
    # =====================================================

    local_exact_recovery = False

    recovered_image = None


    try:

        positions = rdh_metadata.get(
            "positions",
            []
        )

        original_values = rdh_metadata.get(
            "original_values",
            []
        )


        if len(positions) != len(
            original_values
        ):

            raise ValueError(
                "RDH metadata is inconsistent: "
                "positions and original_values "
                "have different lengths."
            )


        recovered_HH = HH_embedded.copy()


        # -------------------------------------------------
        # Restore original HH coefficients
        # -------------------------------------------------

        for i, position in enumerate(
            positions
        ):

            r = int(
                position[0]
            )

            c = int(
                position[1]
            )

            recovered_HH[
                r,
                c
            ] = int(
                original_values[i]
            )


        # -------------------------------------------------
        # Reconstruct original image
        # -------------------------------------------------

        recovered_image = integer_idwt2(

            LL,

            LH,

            HL,

            recovered_HH

        )


        # -------------------------------------------------
        # Exact equality
        # -------------------------------------------------

        local_exact_recovery = np.array_equal(

            image.astype(
                np.int64
            ),

            recovered_image

        )

    except Exception as e:

        print(
            "Exact recovery verification failed:",
            str(e)
        )


    print(
        "Local exact recovery:",
        local_exact_recovery
    )


    # =====================================================
    # RECOVERY METRICS
    # =====================================================

    recovery_mse = None
    recovery_psnr = None


    if recovered_image is not None:

        try:

            difference = (
                image.astype(
                    np.float64
                )
                -
                recovered_image.astype(
                    np.float64
                )
            )


            recovery_mse = float(
                np.mean(
                    difference ** 2
                )
            )


            if recovery_mse == 0:

                recovery_psnr = float(
                    "inf"
                )

            else:

                recovery_psnr = float(
                    10
                    *
                    np.log10(
                        (255.0 ** 2)
                        /
                        recovery_mse
                    )
                )

        except Exception as e:

            print(
                "Recovery metrics failed:",
                str(e)
            )


    # =====================================================
    # SAVE PROCESS INFORMATION
    # =====================================================

    info_filename = (
        "process_info_"
        + process_id
        + ".json"
    )

    info_path = os.path.join(
        OUTPUT_FOLDER,
        info_filename
    )


    process_info = {

        "process_id":
            process_id,

        "encryption":
            "AES-256",

        "transform":
            "Integer Haar DWT",

        "dwt_bands": [
            "LL",
            "LH",
            "HL",
            "HH"
        ],

        "embedding_band":
            "HH",

        "embedding_region":
            "Non-ROI",

        "encrypted_package_bytes":
            len(encrypted_package),

        "embedded_bits":
            rdh_metadata.get(
                "bit_length",
                None
            ),

        "stego_minimum":
            minimum,

        "stego_maximum":
            maximum,

        "MSE_original_vs_stego":
            mse,

        "PSNR_original_vs_stego":
            psnr,

        "SSIM_original_vs_stego":
            ssim_value,

        "MSE_original_vs_recovered":
            recovery_mse,

        "PSNR_original_vs_recovered":
            recovery_psnr,

        "exact_recovery":
            bool(
                local_exact_recovery
            )
    }


    try:

        with open(
            info_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                process_info,
                file,
                indent=2
            )

    except Exception as e:

        print(
            "Process information could not be saved:",
            str(e)
        )


    # =====================================================
    # PRINT GENERATED FILES
    # =====================================================

    print()
    print("========================================")
    print("GENERATED OUTPUT FILES")
    print("========================================")

    print(
        "Original:",
        original_output_path
    )

    print(
        "ROI:",
        roi_path
    )

    print(
        "Non-ROI:",
        non_roi_path
    )

    print(
        "DWT LL:",
        dwt_ll_path
    )

    print(
        "DWT LH:",
        dwt_lh_path
    )

    print(
        "DWT HL:",
        dwt_hl_path
    )

    print(
        "DWT HH:",
        dwt_hh_path
    )

    print(
        "IDWT:",
        idwt_path
    )

    print(
        "Stego:",
        stego_path
    )

    print(
        "RDH Metadata:",
        metadata_path
    )

    print(
        "Process Info:",
        info_path
    )

    print("========================================")
    print("TRANSMITTER PROCESS COMPLETED")
    print("========================================")


    # =====================================================
    # RESULT PAGE
    # =====================================================
    #
    # ONLY THE STEGO IMAGE IS DISPLAYED.
    #
    # Intermediate images remain in output/.
    #
    # =====================================================

    return render_template(

        "result.html",

        status=(
            "Encryption Successful"
        ),

        encrypted_data=encrypted_text,

        stego_filename=stego_filename,

        # Also provide stego_image for compatibility
        # with an older result.html.
        stego_image=(
            "/output/"
            +
            stego_filename
        ),

        mse=mse,

        psnr=psnr,

        ssim=ssim_value,

        exact_recovery=(
            local_exact_recovery
        ),

        recovery_mse=(
            recovery_mse
        ),

        recovery_psnr=(
            recovery_psnr
        ),

        metadata_filename=(
            metadata_filename
        )
    )


# =========================================================
# LOCAL AES DECRYPTION DEMO
# =========================================================

@app.route(
    "/decrypt",
    methods=["POST"]
)
def decrypt():

    global cipher_store
    global salt_store
    global iv_store
    global patient_store
    global stego_filename_store


    password = request.form.get(
        "password",
        ""
    )


    # =====================================================
    # CHECK PASSWORD
    # =====================================================

    if not password:

        return render_template(
            "result.html",
            error="Please enter the password."
        )


    # =====================================================
    # CHECK STORED DATA
    # =====================================================

    if (
        salt_store is None
        or
        iv_store is None
        or
        cipher_store is None
    ):

        return render_template(
            "result.html",
            error=(
                "No encrypted patient data is available. "
                "Please perform encryption first."
            )
        )


    # =====================================================
    # AES DECRYPTION
    # =====================================================

    try:

        decrypted = decrypt_data(

            salt_store,

            iv_store,

            cipher_store,

            password

        )

    except Exception:

        return render_template(
            "result.html",
            error=(
                "Decryption failed. "
                "Wrong password or invalid encrypted data."
            )
        )


    print()
    print(
        "AES-256 decryption successful."
    )

    print(
        "Recovered Patient Data:"
    )

    print(
        decrypted
    )


    # =====================================================
    # RECREATE ENCRYPTED TEXT
    # =====================================================

    encrypted_package = (

        salt_store
        +
        iv_store
        +
        cipher_store
    )


    encrypted_text = base64.b64encode(
        encrypted_package
    ).decode()


    # =====================================================
    # KEEP STEGO IMAGE
    # =====================================================

    stego_filename = (
        stego_filename_store
    )


    # =====================================================
    # DISPLAY DECRYPTION RESULT
    # =====================================================

    return render_template(

        "result.html",

        status=(
            "AES-256 Decryption Successful"
        ),

        encrypted_data=encrypted_text,

        decrypted_data=decrypted,

        stego_filename=stego_filename,

        stego_image=(
            "/output/"
            + stego_filename
            if stego_filename
            else ""
        ),

        mse=None,

        psnr=None,

        ssim=None,

        exact_recovery=""

    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True
    )