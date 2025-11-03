"""
opticks/sysrap/sphoton.py
==========================

Structured view of the 64-byte ``sphoton`` C++ struct.

Binary layout (little-endian, 16 x uint32 → 64 B)

    quad 0 :  pos.x   pos.y   pos.z   time
    quad 1 :  mom.x   mom.y   mom.z   orient_iindex
    quad 2 :  pol.x   pol.y   pol.z   wavelength
    quad 3 :  boundary_flag   identity   index   flagmask
"""

from __future__ import annotations

import numpy as np
from opticks.ana.fold import append_fields


class SPhoton:
    """
    64-byte photon record → NumPy structured array.

    All public API is class-methods – no instance needed.
    """

    # ------------------------------------------------------------------
    # 2. The raw 16-uint32 dtype (exactly matches the C++ layout)
    # ------------------------------------------------------------------
    DTYPE = np.dtype(
        [
            # ---- quad 0 -------------------------------------------------
            ("pos", "<f4", (3,)),          # 0-11  (x,y,z)
            ("time", "<f4"),               # 12-15

            # ---- quad 1 -------------------------------------------------
            ("mom", "<f4", (3,)),          # 16-27
            ("orient_iindex", "<u4"),      # 28-31

            # ---- quad 2 -------------------------------------------------
            ("pol", "<f4", (3,)),          # 32-43
            ("wavelength", "<f4"),         # 44-47

            # ---- quad 3 -------------------------------------------------
            ("boundary_flag", "<u4"),      # 48-51
            ("identity", "<u4"),           # 52-55
            ("index", "<u4"),              # 56-59
            ("flagmask", "<u4"),           # 60-63
        ],
        align=True,
    )
    assert DTYPE.itemsize == 64, f"expected 64 B, got {DTYPE.itemsize}"

    # ------------------------------------------------------------------
    # 3. Core conversion: uint32[... ,16] → structured array
    # ------------------------------------------------------------------
    @classmethod
    def view_(cls, buf: np.ndarray) -> np.ndarray:
        """
        Convert a ``float32`` buffer of shape ``(..., 4, 4)`` into a flat
        structured array.

        Parameters
        ----------
        buf
            Must be ``dtype=float32`` and ``buf.shape[-2],buf.shape[-1] == 4,4``.

        Returns
        -------
        np.ndarray
            1-D structured array (plain ``ndarray``).
        """
        assert len(buf.shape)>2 and buf.shape[-1] == 4 and buf.shape[-2] == 4
        buf = buf.reshape(-1,16)

        if buf.dtype != np.float32:
            raise TypeError(f"buf must be float32, got {buf.dtype}")
        if buf.shape[-1] != 16:
            raise ValueError(f"last dimensions should be 16, got {buf.shape[-1]}")

        return buf.view(cls.DTYPE).reshape(-1)

    # ------------------------------------------------------------------
    # 4. recarray version → dot access (p.pos, p.time, …)
    # ------------------------------------------------------------------
    @classmethod
    def view0(cls, buf: np.ndarray) -> np.recarray:
        """
        Same as ``view_()`` but returns a ``recarray`` so you can write
        ``p.pos``, ``p.wavelength``, ``p.flagmask`` etc.
        """
        return cls.view_(buf).view(np.recarray)

    # ------------------------------------------------------------------
    # 6. Bit-field unpackers (orient_iindex, boundary_flag, identity)
    # ------------------------------------------------------------------
    @staticmethod
    def _hi_lo(u: np.ndarray, hi_bits: int) -> tuple[np.ndarray, np.ndarray]:
        """Split a uint32 into high-bits and low-bits."""
        lo = u & ((1 << (32 - hi_bits)) - 1)
        hi = u >> (32 - hi_bits)
        return hi.astype(np.uint32), lo.astype(np.uint32)

    @classmethod
    def orient_iindex_split(cls, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        orient_iindex:
            bit 31 : orientation flag
            bits 0-30 : iindex
        """
        hi, lo = cls._hi_lo(arr["orient_iindex"], hi_bits=1)
        return hi, lo                     # orient, iindex

    @classmethod
    def boundary_flag_split(cls, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        boundary_flag:
            upper 16 bits : boundary
            lower 16 bits : flag
        """
        hi, lo = cls._hi_lo(arr["boundary_flag"], hi_bits=16)
        return hi, lo                     # boundary, flag

    @classmethod
    def identity_split(cls, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        identity:
            upper 8 bits  : extended range
            lower 24 bits : identity
        """
        hi, lo = cls._hi_lo(arr["identity"], hi_bits=8)
        return hi, lo                     # ext, identity

    # ------------------------------------------------------------------
    # 7. Optional: add the split fields as new columns (recarray only)
    # ------------------------------------------------------------------
    @classmethod
    def view(cls, buf: np.ndarray) -> np.recarray:
        """
        Like ``view_recarray`` but also appends the unpacked bit-fields:

            orient, 
            iindex,
            boundary, 
            bflag,
            index_ext, 
            ident,
            idx

        """
        rec = cls.view_(buf).view(np.recarray)

        orient, iindex = cls.orient_iindex_split(rec)
        boundary, bflag = cls.boundary_flag_split(rec)
        index_ext, ident = cls.identity_split(rec)

        idx = ( index_ext.astype(np.uint64) << 32 ) | rec["index"].astype(np.uint64) 

        rec = append_fields(
            rec,
            [
                "orient", 
                "iindex",
                "boundary", 
                "bflag",
                "index_ext", 
                "ident",
                "idx"
            ],
            [
                orient, 
                iindex,
                boundary, 
                bflag,
                index_ext, 
                ident,
                idx
            ],
            dtypes=[np.uint32] * 6 + [np.uint64],
            usemask=False,
        )
        return rec.view(np.recarray)

    # ------------------------------------------------------------------
    # 8. Pretty-print a few records (handy in notebooks)
    # ------------------------------------------------------------------
    @classmethod
    def pprint(cls, rec: np.ndarray | np.recarray, n: int = 5) -> None:
        """Print the first *n* records in a compact table."""
        print(rec[:n])

    # ------------------------------------------------------------------
    # 9. Empty array of a given length
    # ------------------------------------------------------------------
    @classmethod
    def empty(cls, size: int) -> np.ndarray:
        """Return a zero-filled structured array of length *size*."""
        return np.zeros(size, dtype=cls.DTYPE)


