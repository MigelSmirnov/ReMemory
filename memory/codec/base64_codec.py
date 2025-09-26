# memory/codec/base64_codec.py
"""
Base64 text <-> token ID encoder and decoder.

This module converts text into a sequence of numeric token IDs using
base64 character pairs, and reconstructs the original text back from token IDs.
"""

import base64

# 64-character alphabet used in base64 encoding
BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def encode_text_to_token_ids(text: str) -> list[int]:
    """
    Encode a text string into a list of integer token IDs based on base64 pairs.

    Args:
        text: The input text string.

    Returns:
        A list of integer token IDs representing the encoded text.
    """
    # Encode text to base64 and remove padding
    b64_encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
    b64_encoded = b64_encoded.rstrip("=")

    token_ids = []

    # Convert every 2 base64 chars into a single integer token
    for i in range(0, len(b64_encoded) - 1, 2):
        chunk = b64_encoded[i:i + 2]
        if len(chunk) == 2:
            first = BASE64_CHARS.index(chunk[0])
            second = BASE64_CHARS.index(chunk[1])
            token = (first << 6) + second  # combine two 6-bit values into one 12-bit token
            token_ids.append(token)

    return token_ids


def decode_token_ids_to_text(token_ids: list[int]) -> str:
    """
    Decode a list of integer token IDs back into the original text.

    Args:
        token_ids: A list of integer token IDs produced by `encode_text_to_token_ids`.

    Returns:
        The original text string.
    """
    b64_chunks = []

    # Convert tokens back into pairs of base64 characters
    for token in token_ids:
        first = token >> 6
        second = token & 0b111111
        b64_chunks.append(BASE64_CHARS[first])
        b64_chunks.append(BASE64_CHARS[second])

    # Join all chunks into a single base64 string
    b64_string = "".join(b64_chunks)

    # Add padding if necessary (base64 length must be divisible by 4)
    pad = (-len(b64_string)) % 4
    if pad:
        b64_string += "=" * pad

    try:
        decoded_bytes = base64.b64decode(b64_string)
        return decoded_bytes.decode("utf-8")
    except Exception as e:
        return f"[Decoding error]: {e}"
