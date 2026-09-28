from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

SALT_SIZE = 16
KEY_SIZE = 32
ITERATIONS = 100000


def encrypt_data(patient_data, password):

    salt = get_random_bytes(SALT_SIZE)

    key = PBKDF2(
        password,
        salt,
        dkLen=KEY_SIZE,
        count=ITERATIONS
    )

    cipher = AES.new(key, AES.MODE_CBC)

    iv = cipher.iv

    ciphertext = cipher.encrypt(
        pad(patient_data.encode(), AES.block_size)
    )

    return salt, iv, ciphertext
def decrypt_data(salt, iv, ciphertext, password):

    key = PBKDF2(
        password,
        salt,
        dkLen=KEY_SIZE,
        count=ITERATIONS
    )

    cipher = AES.new(
        key,
        AES.MODE_CBC,
        iv
    )

    plaintext = unpad(
        cipher.decrypt(ciphertext),
        AES.block_size
    )

    return plaintext.decode()