"""Secure aggregation for two clients (our extra-credit extension). Python standard library only.

Goal: the server can compute the AVERAGE of the phones' models without ever seeing one phone's model.

  1. Key agreement (Diffie-Hellman). Each phone picks a secret number x and publishes g^x mod p.
     The server only relays these public values. Each phone combines its own secret with the other
     phone's public value and both get the same shared number g^(xy) mod p, which the server cannot
     compute. The shared key is the SHA-256 hash of that number.
  2. Masking. For every round, both phones turn the shared key into the same list of random 64-bit
     numbers (the mask). Each phone converts  num_samples * weight  to a whole number; phone A ADDS
     the mask and phone B SUBTRACTS it, modulo 2^64. What is uploaded looks like random numbers.
  3. Unmasking by adding. The server adds the two uploads modulo 2^64. The masks cancel, leaving
     n_A*w_A + n_B*w_B, which divided by n_A + n_B is exactly the FedAvg result.

The same shared key also encrypts the local-only baseline models that the server relays between
the phones, so the server cannot read those either.

Limits (stated in the report): with only two clients each phone can work out the other's update from
the average; the protection is against the server. The server is assumed to follow the protocol
(it relays the public keys honestly). Sample counts are sent in the clear because the server needs them.

The phone side of this file is android/.../core/SecureAggregation.kt; both must stay identical.
"""
import hashlib
import hmac
import secrets

# 2048-bit Diffie-Hellman group 14 from RFC 3526 (a standard, public prime) and generator 2.
DH_PRIME = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
DH_GENERATOR = 2

SCALE = 1 << 24          # weights are sent as whole numbers: round(num_samples * weight * SCALE)
MOD = 1 << 64            # all masked arithmetic is modulo 2^64
MASK64 = MOD - 1


# ------------------------------------------------------------------ key agreement

def new_private_key():
    """A fresh 256-bit secret number. Never leaves the phone."""
    return secrets.randbits(256) | (1 << 255)


def public_key(private):
    """What a phone publishes: g^x mod p, as a hex string."""
    return format(pow(DH_GENERATOR, private, DH_PRIME), "x")


def shared_key(private, other_public_hex):
    """The 32-byte key both phones end up with: SHA-256 of g^(xy) mod p."""
    other = int(other_public_hex, 16)
    if not 1 < other < DH_PRIME - 1:
        raise ValueError("invalid public key")
    return hashlib.sha256(format(pow(other, private, DH_PRIME), "x").encode()).digest()


# ------------------------------------------------------------------ random numbers from the shared key

def random_words(key, label, count):
    """`count` pseudo-random 64-bit numbers. Both phones get the same list for the same key and label."""
    words = []
    block = 0
    while len(words) < count:
        digest = hashlib.sha256(key + f"{label}:{block}".encode()).digest()
        words += [int.from_bytes(digest[i:i + 8], "big") for i in range(0, 32, 8)]
        block += 1
    return words[:count]


def mask_label(run_id, round_number, attempt):
    """A different mask for every round (and every retry of a round), so a mask is never used twice."""
    return f"mask|{run_id}|{round_number}|{attempt}"


# ------------------------------------------------------------------ masking (phone side)

def mask_update(weights, num_samples, key, label, adds_mask):
    """weights: all model parameters as one list (W then b). Returns the masked values as hex strings."""
    mask = random_words(key, label, len(weights))
    out = []
    for w, m in zip(weights, mask):
        q = int(round(num_samples * w * SCALE))
        out.append(format((q + m if adds_mask else q - m) & MASK64, "016x"))
    return out


def adds_mask(client_id, other_id):
    """One phone adds the mask and the other subtracts it; the smaller id adds."""
    return client_id < other_id


# ------------------------------------------------------------------ unmasking (server side)

def to_signed(value):
    """Interpret a number modulo 2^64 as a signed 64-bit integer."""
    value &= MASK64
    return value - MOD if value >= MOD // 2 else value


def sum_masked(masked_updates, total_samples):
    """masked_updates: one list of hex strings per client. Adding them cancels the masks.
    Returns the sample-weighted average of the clients' weights (the FedAvg result)."""
    n = len(masked_updates[0])
    if any(len(u) != n for u in masked_updates):
        raise ValueError("masked updates have different lengths")
    return [to_signed(sum(int(u[i], 16) for u in masked_updates)) / (SCALE * total_samples) for i in range(n)]


def decode_alone(masked_update, num_samples):
    """What ONE masked upload looks like if someone tries to read it as weights (it is noise)."""
    return [to_signed(int(h, 16)) / (SCALE * num_samples) for h in masked_update]


def validate_masked(values, n_params):
    if not isinstance(values, list) or len(values) != n_params:
        raise ValueError(f"expected {n_params} masked values")
    for h in values:
        if not isinstance(h, str) or len(h) != 16:
            raise ValueError("masked values must be 16-digit hex strings")
        int(h, 16)


# ------------------------------------------------------------------ encrypting the relayed local models

def encrypt(key, label, text):
    """XOR the text with random bytes made from the shared key, and add a check value (HMAC-SHA256)
    so the receiver can tell whether it has the right key."""
    data = text.encode()
    stream = b"".join(w.to_bytes(8, "big") for w in random_words(key, "enc|" + label, (len(data) + 7) // 8))
    ciphertext = bytes(a ^ b for a, b in zip(data, stream))
    tag = hmac.new(key, ("tag|" + label).encode() + ciphertext, hashlib.sha256).hexdigest()
    return {"ciphertext": ciphertext.hex(), "tag": tag}


def decrypt(key, label, blob):
    """Returns the text, or None if the check value does not match (wrong or outdated key)."""
    ciphertext = bytes.fromhex(blob["ciphertext"])
    tag = hmac.new(key, ("tag|" + label).encode() + ciphertext, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(tag, blob.get("tag", "")):
        return None
    stream = b"".join(w.to_bytes(8, "big") for w in random_words(key, "enc|" + label, (len(ciphertext) + 7) // 8))
    return bytes(a ^ b for a, b in zip(ciphertext, stream)).decode()
