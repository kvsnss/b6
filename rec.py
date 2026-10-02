import os
import json
import cv2
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for
)

from rrdh import extract_payload
from aes import decrypt_data


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RECEIVED_FOLDER = os.path.join(BASE_DIR, "received")
RECOVERED_FOLDER = os.path.join(BASE_DIR, "recovered")

os.makedirs(RECEIVED_FOLDER, exist_ok=True)
os.makedirs(RECOVERED_FOLDER, exist_ok=True)


# ============================================================
# FILE NAMES
# ============================================================

STEGO_FILENAME = "stego_image.png"
METADATA_FILENAME = "metadata.json"
RECOVERED_FILENAME = "recovered_image.png"
ORIGINAL_FILENAME = "original_image.png"


# ============================================================
# FILE PATHS
# ============================================================

STEGO_PATH = os.path.join(
    RECEIVED_FOLDER,
    STEGO_FILENAME
)

METADATA_PATH = os.path.join(
    RECEIVED_FOLDER,
    METADATA_FILENAME
)

ORIGINAL_IMAGE_PATH = os.path.join(
    RECEIVED_FOLDER,
    ORIGINAL_FILENAME
)

RECOVERED_PATH = os.path.join(
    RECOVERED_FOLDER,
    RECOVERED_FILENAME
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

last_encrypted_package = None
last_salt = None
last_iv = None
last_ciphertext = None

decrypted_patient_data = None

metrics = None

extraction_error = None
decryption_error = None


# ============================================================
# IMAGE METRICS
# ============================================================

def calculate_mse(reference, test):

    reference = reference.astype(np.float64)
    test = test.astype(np.float64)

    return float(
        np.mean((reference - test) ** 2)
    )


def calculate_psnr(reference, test):

    mse = calculate_mse(reference, test)

    if mse == 0:
        return float("inf")

    return float(
        10 * np.log10((255.0 ** 2) / mse)
    )


def calculate_ssim(reference, test):

    try:
        from skimage.metrics import structural_similarity

        reference = reference.astype(np.uint8)
        test = test.astype(np.uint8)

        return float(
            structural_similarity(
                reference,
                test,
                data_range=255
            )
        )

    except ImportError:
        return None


def calculate_exact_recovery(reference, recovered):

    if reference.shape != recovered.shape:
        return False

    return bool(
        np.array_equal(reference, recovered)
    )


def calculate_pixel_error_rate(reference, recovered):

    if reference.shape != recovered.shape:
        return 1.0

    different_pixels = np.count_nonzero(
        reference != recovered
    )

    total_pixels = reference.size

    if total_pixels == 0:
        return 0.0

    return float(
        different_pixels / total_pixels
    )


def evaluate_recovery(original_path, recovered_path):

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
            "Original image could not be loaded."
        )

    if recovered is None:
        raise ValueError(
            "Recovered image could not be loaded."
        )

    if original.shape != recovered.shape:
        raise ValueError(
            f"Image dimensions do not match. "
            f"Original: {original.shape}, "
            f"Recovered: {recovered.shape}"
        )

    mse = calculate_mse(
        original,
        recovered
    )

    psnr = calculate_psnr(
        original,
        recovered
    )

    ssim = calculate_ssim(
        original,
        recovered
    )

    exact = calculate_exact_recovery(
        original,
        recovered
    )

    different_pixels = int(
        np.count_nonzero(
            original != recovered
        )
    )

    total_pixels = int(
        original.size
    )

    pixel_error_rate = calculate_pixel_error_rate(
        original,
        recovered
    )

    return {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim,
        "Exact Recovery": exact,
        "Different Pixels": different_pixels,
        "Total Pixels": total_pixels,
        "Pixel Error Rate": pixel_error_rate
    }


# ============================================================
# LOAD RDH METADATA
# ============================================================

def load_rdh_metadata():

    if not os.path.exists(METADATA_PATH):

        raise FileNotFoundError(
            "RDH metadata not found.\n"
            f"Expected location:\n{METADATA_PATH}\n\n"
            "Make sure metadata.json belongs to "
            "the same stego image."
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    if not isinstance(metadata, dict):

        raise ValueError(
            "Invalid RDH metadata format."
        )

    required_fields = [
        "positions",
        "original_values",
        "bit_length",
        "payload_bytes"
    ]

    for field in required_fields:

        if field not in metadata:

            raise ValueError(
                f"RDH metadata does not contain "
                f"'{field}'."
            )

    if len(metadata["positions"]) != len(
        metadata["original_values"]
    ):

        raise ValueError(
            "RDH metadata is inconsistent: "
            "positions and original_values "
            "have different lengths."
        )

    if int(metadata["bit_length"]) != len(
        metadata["positions"]
    ):

        raise ValueError(
            "RDH metadata is inconsistent: "
            "bit_length does not match positions."
        )

    return metadata


# ============================================================
# EXTRACT PAYLOAD
# ============================================================

def perform_extraction():

    global last_encrypted_package
    global last_salt
    global last_iv
    global last_ciphertext
    global metrics
    global extraction_error

    extraction_error = None
    metrics = None

    if not os.path.exists(STEGO_PATH):

        extraction_error = (
            "Received stego image not found."
        )

        return False

    try:

        # ----------------------------------------------------
        # LOAD METADATA
        # ----------------------------------------------------

        metadata = load_rdh_metadata()

        print()
        print("=" * 60)
        print("RDH RECEIVER EXTRACTION")
        print("=" * 60)

        print("Stego image:")
        print(STEGO_PATH)

        print("Metadata:")
        print(METADATA_PATH)

        print(
            "Payload bytes:",
            metadata["payload_bytes"]
        )

        print(
            "Bit length:",
            metadata["bit_length"]
        )

        print(
            "Number of positions:",
            len(metadata["positions"])
        )


        # ----------------------------------------------------
        # EXTRACT ENCRYPTED DATA
        # AND RECOVER ORIGINAL IMAGE
        # ----------------------------------------------------

        recovered_data, recovered_image = extract_payload(
            STEGO_PATH,
            metadata,
            RECOVERED_PATH
        )

        if recovered_data is None:

            raise ValueError(
                "RDH extraction returned "
                "no encrypted package."
            )

        print(
            "Extracted encrypted package:",
            len(recovered_data),
            "bytes"
        )


        # ----------------------------------------------------
        # VERIFY PAYLOAD SIZE
        # ----------------------------------------------------

        expected_size = int(
            metadata["payload_bytes"]
        )

        if len(recovered_data) != expected_size:

            raise ValueError(
                "Extracted package size mismatch. "
                f"Expected {expected_size} bytes, "
                f"got {len(recovered_data)} bytes."
            )


        # ----------------------------------------------------
        # AES PACKAGE
        #
        # First 16 bytes  = salt
        # Next 16 bytes   = IV
        # Remaining       = ciphertext
        # ----------------------------------------------------

        if len(recovered_data) < 32:

            raise ValueError(
                "Extracted encrypted package "
                "is smaller than 32 bytes."
            )

        last_encrypted_package = recovered_data

        last_salt = recovered_data[:16]

        last_iv = recovered_data[16:32]

        last_ciphertext = recovered_data[32:]


        print(
            "Salt:",
            len(last_salt),
            "bytes"
        )

        print(
            "IV:",
            len(last_iv),
            "bytes"
        )

        print(
            "Ciphertext:",
            len(last_ciphertext),
            "bytes"
        )


        # ----------------------------------------------------
        # IMAGE RECOVERY METRICS
        # ----------------------------------------------------

        if os.path.exists(
            ORIGINAL_IMAGE_PATH
        ):

            metrics = evaluate_recovery(
                ORIGINAL_IMAGE_PATH,
                RECOVERED_PATH
            )

            print()
            print("=" * 60)
            print("RECEIVER IMAGE RECOVERY METRICS")
            print("=" * 60)

            print(
                "MSE:",
                metrics["MSE"]
            )

            print(
                "PSNR:",
                metrics["PSNR"],
                "dB"
            )

            print(
                "SSIM:",
                metrics["SSIM"]
            )

            print(
                "Exact Recovery:",
                metrics["Exact Recovery"]
            )

            print(
                "Different Pixels:",
                metrics["Different Pixels"]
            )

            print(
                "Total Pixels:",
                metrics["Total Pixels"]
            )

            print(
                "Pixel Error Rate:",
                metrics["Pixel Error Rate"]
            )

        else:

            print()
            print(
                "Original image not found."
            )

            print(
                "Recovery metrics skipped."
            )


        print()
        print("=" * 60)
        print("EXTRACTION SUCCESSFUL")
        print("=" * 60)

        return True


    except Exception as e:

        extraction_error = str(e)

        print()
        print("=" * 60)
        print("EXTRACTION ERROR")
        print("=" * 60)

        print(str(e))

        return False


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "receiver.html",

        stego_available=os.path.exists(
            STEGO_PATH
        ),

        recovered_available=os.path.exists(
            RECOVERED_PATH
        ),

        stego_filename=STEGO_FILENAME,

        recovered_filename=RECOVERED_FILENAME,

        extraction_error=extraction_error,

        decryption_error=decryption_error,

        decrypted_data=decrypted_patient_data,

        metrics=metrics
    )


# ============================================================
# EXTRACT ROUTE
# ============================================================

@app.route(
    "/extract",
    methods=["POST"]
)
def extract():

    success = perform_extraction()

    return redirect(
        url_for("index")
    )


# ============================================================
# SERVE RECEIVED IMAGE
# ============================================================

@app.route(
    "/received/<filename>"
)
def received_file(filename):

    return send_from_directory(
        RECEIVED_FOLDER,
        filename
    )


# ============================================================
# SERVE RECOVERED IMAGE
# ============================================================

@app.route(
    "/recovered/<filename>"
)
def recovered_file(filename):

    return send_from_directory(
        RECOVERED_FOLDER,
        filename
    )


# ============================================================
# AES-256 DECRYPTION
# ============================================================

@app.route(
    "/decrypt",
    methods=["POST"]
)
def decrypt():

    global decrypted_patient_data
    global decryption_error

    decryption_error = None

    password = request.form.get(
        "password",
        ""
    ).strip()


    # --------------------------------------------------------
    # CHECK PASSWORD
    # --------------------------------------------------------

    if not password:

        decryption_error = (
            "Please enter the AES-256 password."
        )

        return redirect(
            url_for("index")
        )


    # --------------------------------------------------------
    # CHECK EXTRACTION
    # --------------------------------------------------------

    if last_salt is None:

        decryption_error = (
            "No encrypted package has been "
            "extracted yet. Please extract "
            "the received stego image first."
        )

        return redirect(
            url_for("index")
        )


    # --------------------------------------------------------
    # DECRYPT
    # --------------------------------------------------------

    try:

        decrypted_patient_data = decrypt_data(
            last_salt,
            last_iv,
            last_ciphertext,
            password
        )

        print()
        print("=" * 60)
        print("AES-256 DECRYPTION SUCCESSFUL")
        print("=" * 60)

        print(
            "Recovered patient data:"
        )

        print(
            decrypted_patient_data
        )

        print("=" * 60)


    except Exception as e:

        decrypted_patient_data = None

        decryption_error = (
            "Decryption failed. "
            "Check the password and make sure "
            "the received stego image and metadata "
            "belong to the same transmission."
        )

        print()
        print(
            "AES DECRYPTION ERROR:",
            str(e)
        )


    return redirect(
        url_for("index")
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("RASPBERRY PI MEDICAL DATA RECEIVER")
    print("=" * 60)

    print(
        "Receiver folder:",
        RECEIVED_FOLDER
    )

    print(
        "Recovered folder:",
        RECOVERED_FOLDER
    )

    print(
        "Stego image:",
        STEGO_PATH
    )

    print(
        "RDH metadata:",
        METADATA_PATH
    )

    print(
        "Original image for metrics:",
        ORIGINAL_IMAGE_PATH
    )

    print()
    print(
        "Open browser at:"
    )

    print(
        "http://127.0.0.1:5001"
    )

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )
