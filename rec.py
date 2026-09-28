from flask import (
    Flask,
    request,
    render_template,
    send_from_directory
)

from werkzeug.utils import secure_filename

from aes import decrypt_data
from rrdh import extract_payload

import os
import base64


# FLASK APPLICATION
# =========================================================

app = Flask(
    __name__
)


# =========================================================
# FOLDERS
# =========================================================

RECEIVED_FOLDER = "received"
RECOVERED_FOLDER = "recovered"


os.makedirs(
    RECEIVED_FOLDER,
    exist_ok=True
)

os.makedirs(
    RECOVERED_FOLDER,
    exist_ok=True
)


# =========================================================
# RECEIVER STORAGE
# =========================================================

last_salt = None
last_iv = None
last_ciphertext = None

last_stego_path = None
last_filename = None

last_recovered_path = None


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(

        "receiver.html",

        status=(
            "Receiver is running. "
            "Waiting for stego image..."
        ),

        error=False,

        filename=None,

        stego_available=False,

        extracted=False,

        decrypted_data=None,

        recovered_image=None
    )


# =========================================================
# RECEIVE STEGO IMAGE
# =========================================================

@app.route(
    "/receive",
    methods=["POST"]
)
def receive():

    global last_salt
    global last_iv
    global last_ciphertext
    global last_stego_path
    global last_filename
    global last_recovered_path


    try:

        # -------------------------------------------------
        # CHECK IMAGE
        # -------------------------------------------------

        if "stego_image" not in request.files:

            return render_template(

                "receiver.html",

                status=(
                    "No stego image was received."
                ),

                error=True,

                filename=None,

                stego_available=False,

                extracted=False,

                decrypted_data=None,

                recovered_image=None

            ), 400


        image = request.files[
            "stego_image"
        ]


        if image.filename == "":

            return render_template(

                "receiver.html",

                status=(
                    "Invalid image filename."
                ),

                error=True,

                filename=None,

                stego_available=False,

                extracted=False,

                decrypted_data=None,

                recovered_image=None

            ), 400


        # -------------------------------------------------
        # GET PAYLOAD SIZE
        # -------------------------------------------------

        payload_size = request.form.get(
            "payload_size"
        )


        if payload_size is None:

            return render_template(

                "receiver.html",

                status=(
                    "Payload size was not "
                    "provided by transmitter."
                ),

                error=True,

                filename=None,

                stego_available=False,

                extracted=False,

                decrypted_data=None,

                recovered_image=None

            ), 400


        payload_size = int(
            payload_size
        )


        # -------------------------------------------------
        # GET RDH METADATA
        # -------------------------------------------------

        peak = int(
            request.form.get(
                "rdh_peak"
            )
        )

        zero = int(
            request.form.get(
                "rdh_zero"
            )
        )

        bit_length = int(
            request.form.get(
                "rdh_bit_length"
            )
        )


        metadata = {

            "peak": peak,

            "zero": zero,

            "bit_length": bit_length
        }


        # -------------------------------------------------
        # SECURE FILENAME
        # -------------------------------------------------

        filename = secure_filename(
            image.filename
        )


        if not filename:

            filename = (
                "received_stego.png"
            )


        # -------------------------------------------------
        # SAVE IMAGE
        # -------------------------------------------------

        stego_path = os.path.join(

            RECEIVED_FOLDER,

            filename
        )


        image.save(
            stego_path
        )


        print()
        print("=" * 60)
        print("STEGO IMAGE RECEIVED")
        print("=" * 60)

        print(
            "Filename:",
            filename
        )

        print(
            "Payload size:",
            payload_size
        )

        print(
            "RDH Peak:",
            peak
        )

        print(
            "RDH Zero:",
            zero
        )

        print(
            "RDH Bit Length:",
            bit_length
        )


        # -------------------------------------------------
        # RECOVERED IMAGE PATH
        # -------------------------------------------------

        recovered_filename = (
            "recovered_"
            + os.path.splitext(filename)[0]
            + ".png"
        )


        recovered_path = os.path.join(

            RECOVERED_FOLDER,

            recovered_filename
        )


        # -------------------------------------------------
        # DWT + RDH EXTRACTION + IDWT
        # -------------------------------------------------

        encrypted_package, recovered_image = (
            extract_payload(

                stego_path,

                metadata,

                recovered_path
            )
        )


        # -------------------------------------------------
        # CHECK PAYLOAD SIZE
        # -------------------------------------------------

        if len(encrypted_package) != payload_size:

            raise ValueError(
                "Extracted payload size does not "
                "match transmitter payload size."
            )


        # -------------------------------------------------
        # CHECK AES PACKAGE
        # -------------------------------------------------

        if len(encrypted_package) < 32:

            raise ValueError(
                "Encrypted package is too small."
            )


        # -------------------------------------------------
        # SPLIT PACKAGE
        #
        # SALT = 16 bytes
        # IV   = 16 bytes
        # -------------------------------------------------

        last_salt = (
            encrypted_package[:16]
        )

        last_iv = (
            encrypted_package[16:32]
        )

        last_ciphertext = (
            encrypted_package[32:]
        )


        last_stego_path = (
            stego_path
        )

        last_filename = (
            filename
        )

        last_recovered_path = (
            recovered_path
        )


        # -------------------------------------------------
        # DISPLAY
        # -------------------------------------------------

        encrypted_text = (
            base64.b64encode(
                encrypted_package
            ).decode()
        )


        print(
            "Encrypted package extracted successfully."
        )

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

        print(
            "Original image recovered:"
            ,
            recovered_path
        )

        print("=" * 60)


        return render_template(

            "receiver.html",

            status=(
                "✓ Stego image received. "
                "DWT-RDH extraction successful."
            ),

            error=False,

            filename=filename,

            stego_available=True,

            extracted=True,

            decrypted_data=None,

            recovered_image=(
                "/recovered/"
                + recovered_filename
            )
        )


    except Exception as e:

        print(
            "Receiver error:",
            str(e)
        )


        return render_template(

            "receiver.html",

            status=(
                "Receiver error: "
                + str(e)
            ),

            error=True,

            filename=None,

            stego_available=False,

            extracted=False,

            decrypted_data=None,

            recovered_image=None

        ), 500


# =========================================================
# SERVE RECEIVED STEGO
# =========================================================

@app.route(
    "/received/<filename>"
)
def received_image(filename):

    return send_from_directory(

        RECEIVED_FOLDER,

        filename
    )


# =========================================================
# SERVE RECOVERED IMAGE
# =========================================================

@app.route(
    "/recovered/<filename>"
)
def recovered_image(filename):

    return send_from_directory(

        RECOVERED_FOLDER,

        filename
    )


# =========================================================
# DECRYPT
# =========================================================

@app.route(
    "/decrypt",
    methods=["POST"]
)
def decrypt():

    global last_salt
    global last_iv
    global last_ciphertext


    if (
        last_salt is None
        or
        last_iv is None
        or
        last_ciphertext is None
    ):

        return render_template(

            "receiver.html",

            status=(
                "No encrypted medical data "
                "is available. Receive a stego "
                "image first."
            ),

            error=True,

            filename=last_filename,

            stego_available=True,

            extracted=False,

            decrypted_data=None,

            recovered_image=None

        ), 400


    password = request.form.get(
        "password",
        ""
    )


    if password == "":

        return render_template(

            "receiver.html",

            status=(
                "Please enter the password."
            ),

            error=True,

            filename=last_filename,

            stego_available=True,

            extracted=True,

            decrypted_data=None,

            recovered_image=(
                "/recovered/"
                + os.path.basename(
                    last_recovered_path
                )
                if last_recovered_path
                else None
            )

        ), 400


    try:

        decrypted_data = decrypt_data(

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
            "Recovered Patient Data:"
        )

        print(
            decrypted_data
        )

        print("=" * 60)


        return render_template(

            "receiver.html",

            status=(
                "✓ AES-256 Decryption Successful"
            ),

            error=False,

            filename=last_filename,

            stego_available=True,

            extracted=True,

            decrypted_data=decrypted_data,

            recovered_image=(
                "/recovered/"
                + os.path.basename(
                    last_recovered_path
                )
                if last_recovered_path
                else None
            )

        )


    except Exception as e:

        print(
            "Decryption failed:",
            str(e)
        )


        return render_template(

            "receiver.html",

            status=(
                "Wrong password or invalid "
                "encrypted data."
            ),

            error=True,

            filename=last_filename,

            stego_available=True,

            extracted=True,

            decrypted_data=None,

            recovered_image=(
                "/recovered/"
                + os.path.basename(
                    last_recovered_path
                )
                if last_recovered_path
                else None
            )

        ), 400


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("RASPBERRY PI MEDICAL DATA RECEIVER")
    print("=" * 60)

    print(
        "Receiver URL:"
    )

    print(
        "http://127.0.0.1:5001"
    )

    print(
        "Waiting for stego image..."
    )

    print("=" * 60)


    app.run(

        host="0.0.0.0",

        port=5001,

        debug=True
    )