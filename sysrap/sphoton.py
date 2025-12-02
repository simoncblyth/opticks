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
            ("hitcount_iindex", "<u4"),      # 28-31

            # ---- quad 2 -------------------------------------------------
            ("pol", "<f4", (3,)),          # 32-43
            ("wavelength", "<f4"),         # 44-47

            # ---- quad 3 -------------------------------------------------
            ("orient_boundary_flag", "<u4"),      # 48-51
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

    @staticmethod
    def split_uint32(u: np.ndarray, bit_widths: tuple[int, ...]) -> tuple[np.ndarray, ...]:
        """
        Split a uint32 array into multiple fields according to bit_widths.

        Parameters
        ----------
        u : np.ndarray, dtype=uint32
            Input array of 32-bit unsigned integers.
        bit_widths : tuple[int, ...]
            Tuple of bit counts for each output field, from MSB to LSB.
            Must sum to exactly 32.

        Returns
        -------
        tuple[np.ndarray, ...]
            One uint32 array per field, in the order given (MSB-first by default).

        Example
        -------
        >>> a = np.array([0b_1111_0000_1010_1100_0011_0011_0101_0110], dtype=np.uint32)
        >>> hi, mid1, mid2, lo = split_uint32(a, (4, 12, 8, 8))
        # splits into: bits 31-28, 27-16, 15-8, 7-0
        """
        if sum(bit_widths) != 32:
            raise ValueError(f"bit_widths must sum to 32, got sum={sum(bit_widths)}")

        u = np.asarray(u, dtype=np.uint32)
        results = []
        remaining = u
        shift = 32

        for bits in bit_widths:
            shift -= bits
            mask = (1 << bits) - 1
            field = (remaining >> shift) & mask
            results.append(field.astype(np.uint32))

        return tuple(results)



    @classmethod
    def hitcount_iindex_split(cls, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cls.split_uint32(arr["hitcount_iindex"], bit_widths=(16,16))

    @classmethod
    def orient_boundary_flag_split(cls, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return cls.split_uint32(arr["orient_boundary_flag"], bit_widths=(1,15,16))

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
            idx,
            phi     np.atan2( pos.y, pos.x ) : most useful for local frame arrays like "hitlocal"

        """
        rec = cls.view_(buf).view(np.recarray)

        hitcount, iindex = cls.hitcount_iindex_split(rec)
        orient, boundary, bflag = cls.orient_boundary_flag_split(rec)
        index_ext, ident = cls.identity_split(rec)

        idx = ( index_ext.astype(np.uint64) << 32 ) | rec["index"].astype(np.uint64)

        phi = np.atan2( rec["pos"][:,1], rec["pos"][:,0] )   # Y,X like scuda.h:normalize_fphi
        posr = np.linalg.norm( rec["pos"], axis=1 )
        cost = rec["pos"][:,2]/posr

        rec = append_fields(
            rec,
            [
                "orient",
                "iindex",
                "boundary",
                "bflag",
                "index_ext",
                "ident",
                "hitcount",
                "idx",
                "phi",
                "posr",
                "cost"
            ],
            [
                orient,
                iindex,
                boundary,
                bflag,
                index_ext,
                ident,
                hitcount,
                idx,
                phi,
                posr,
                cost
            ],
            dtypes=[np.uint32] * 7 + [np.uint64] + [np.float32] * 3,
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


