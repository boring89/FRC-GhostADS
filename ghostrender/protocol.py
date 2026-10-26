"""Wire format between Python (ghostsim) and the Godot renderer. TCP on 127.0.0.1, little-endian.

frame := b"GHST" | u32 header_len | u32 blob_len | header (UTF-8 JSON) | blob

Python -> Godot
  configure   {type, version, n_envs, width, height, max_cams, scene, pieces, rigs, tag_bits}
  appearance  {type, env, robots: [6 x {bumper_rgb, body_rgb, team, font, size, bumper_z, seed}], dr: {...}}
  render      {type, frame, cams: [[env, robot, rig], ...], post: [[gain, noise, blur_dx, blur_dy, seed], ...],
               check: bool, arrays: [[name, shape], ...]}
              blob = float32 arrays in the order of 'arrays' (robots, coral, algae, cages[, markers])
  quit        {type}
Godot -> Python
  ok          {type}
  frame       {type, frame, render_ms, shape: [K, H, W, 3], echo: [[env, robot, x, y, z, yaw], ...]}
              blob = uint8 RGB images, K x H x W x 3, in the order of 'cams'

All coordinates are WPILib field frame (x toward red, y left, z up, metres, radians).
Godot conversion is done only on the Godot side: godot(x, y, z) = (x, z, -y).
"""
from __future__ import annotations

import json
import socket
import struct

import numpy as np

MAGIC = b"GHST"
VERSION = 1


def send(sock: socket.socket, header: dict, blob: bytes = b"") -> None:
    h = json.dumps(header, separators=(",", ":")).encode()
    sock.sendall(MAGIC + struct.pack("<II", len(h), len(blob)) + h + blob)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(min(n - len(out), 1 << 20))
        if not chunk:
            raise ConnectionError("renderer closed the connection")
        out += chunk
    return bytes(out)


def recv(sock: socket.socket) -> tuple[dict, bytes]:
    head = _recv_exact(sock, 12)
    if head[:4] != MAGIC:
        raise ValueError("bad magic from renderer")
    hl, bl = struct.unpack("<II", head[4:])
    header = json.loads(_recv_exact(sock, hl).decode())
    return header, _recv_exact(sock, bl)


def pack_arrays(arrays: dict[str, np.ndarray]) -> tuple[list, bytes]:
    meta, parts = [], []
    for name, a in arrays.items():
        a = np.ascontiguousarray(a, np.float32)
        meta.append([name, list(a.shape)])
        parts.append(a.tobytes())
    return meta, b"".join(parts)
