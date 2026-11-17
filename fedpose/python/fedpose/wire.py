"""Wire format helpers (client <-> server). Kotlin twin: Wire.kt / Quantizer.kt.

Weights and deltas travel as base64 of little-endian bytes inside a compact JSON body.
  codec "f32"  : float32 delta, 4 bytes per parameter (baseline)
  codec "int8" : symmetric int8 delta + one float32 scale, 1 byte per parameter
                 scale = max|delta| / 127 (rounded to float32); q = rint(delta / scale)
"""
from __future__ import annotations

import base64
import json

import numpy as np

CODEC_F32 = "f32"
CODEC_INT8 = "int8"
QMAX = 127


def dumps(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def f32_to_b64(arr) -> str:
    return base64.b64encode(np.asarray(arr, dtype="<f4").tobytes()).decode("ascii")


def b64_to_f32(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float64)


def encode_int8(delta) -> tuple[float, str]:
    d = np.asarray(delta, dtype=np.float64)
    max_abs = float(np.max(np.abs(d))) if d.size else 0.0
    scale = float(np.float32(max_abs / QMAX)) if max_abs > 0 else 1.0
    q = np.clip(np.rint(d / scale), -QMAX, QMAX).astype(np.int8)
    return scale, base64.b64encode(q.tobytes()).decode("ascii")


def decode_int8(scale: float, b64: str) -> np.ndarray:
    q = np.frombuffer(base64.b64decode(b64), dtype=np.int8).astype(np.float64)
    return q * float(np.float32(scale))


def encode_delta(delta, codec: str) -> dict:
    if codec == CODEC_F32:
        return {"codec": CODEC_F32, "payload": f32_to_b64(delta)}
    if codec == CODEC_INT8:
        scale, payload = encode_int8(delta)
        return {"codec": CODEC_INT8, "scale": scale, "payload": payload}
    raise ValueError(f"unknown codec {codec}")


def decode_delta(body: dict, n_params: int) -> np.ndarray:
    """Decode and validate. Raises ValueError on any malformed payload."""
    codec = body.get("codec")
    payload = body.get("payload")
    if not isinstance(payload, str):
        raise ValueError("missing payload")
    try:
        if codec == CODEC_F32:
            d = b64_to_f32(payload)
        elif codec == CODEC_INT8:
            scale = float(body["scale"])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("bad scale")
            d = decode_int8(scale, payload)
        else:
            raise ValueError(f"unknown codec {codec!r}")
    except (KeyError, TypeError, base64.binascii.Error) as e:  # type: ignore[attr-defined]
        raise ValueError(f"malformed payload: {e}") from e
    if d.shape != (n_params,):
        raise ValueError(f"expected {n_params} params, got {d.size}")
    if not np.all(np.isfinite(d)):
        raise ValueError("non finite values")
    return d


def update_body(client_id, rnd, base_version, n_samples, delta, codec, extra=None) -> dict:
    body = {
        "client_id": client_id,
        "round": int(rnd),
        "base_version": int(base_version),
        "n_samples": int(n_samples),
        "skipped": False,
    }
    body.update(encode_delta(delta, codec))
    if extra:
        body.update(extra)
    return body


def skip_body(client_id, rnd, base_version, update_norm) -> dict:
    return {
        "client_id": client_id,
        "round": int(rnd),
        "base_version": int(base_version),
        "n_samples": 0,
        "skipped": True,
        "update_norm": float(update_norm),
    }
