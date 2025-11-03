"""
opticks/sysrap/sphotonlite.py
==============================

::

    f = Fold.Load(symbol="f")    
    from opticks.sysrap.sphotonlite import SPhotonLite
    p = SPhotonLite.view(f.photonlite)  # photonlite shape (N,4) uint32




"""
from __future__ import annotations
import numpy as np
from opticks.ana.fold import append_fields 


class SPhotonLite:
    """
    Structured view of the 16-byte ``sphotonlite`` record.

    The binary layout (little-endian, 16 B total) is:

        0-1   : identity   (u2)
        2-3   : hitcount   (u2)
        4-7   : time       (f4)
        8-9   : lposfphi   (u2)  → normalised to lposfphi_f  (f4)
       10-11  : lposcost   (u2)  → normalised to lposcost_f  (f4)
       12-15  : flagmask   (u4)

    All public API is provided as **class-methods** – no instance needed.
    """

    # ------------------------------------------------------------------
    # 2. The dtype – defined once, reused everywhere
    # ------------------------------------------------------------------
    DTYPE = np.dtype(
        [
            ("identity", "<u2"),   # offset 0
            ("hitcount", "<u2"),   # offset 2
            ("time", "<f4"),       # offset 4
            ("lposfphi", "<u2"),   # offset 8
            ("lposcost", "<u2"),   # offset 10
            ("flagmask", "<u4"),   # offset 12
        ],
        align=True,
    )
    assert DTYPE.itemsize == 16, f"expected 16 B, got {DTYPE.itemsize}"

    # ------------------------------------------------------------------
    # 3. Helper: uint16 → float in [0, 1] (exact, no rounding error)
    # ------------------------------------------------------------------
    @classmethod
    def unpack_uint16_to_float(cls, u16: np.ndarray) -> np.ndarray:
        """0xffff → 1.0, 0 → 0.0 (exact, no rounding error)"""
        return u16.astype(np.float32) / 0xFFFF

    # ------------------------------------------------------------------
    # 4. Core conversion: uint32[N,4] → structured array
    # ------------------------------------------------------------------
    @classmethod
    def view_(cls, buf: np.ndarray) -> np.ndarray:
        """
        Convert a ``uint32`` buffer of shape ``(..., 4)`` into a flat
        structured array with the extra float fields ``lposfphi_f``
        and ``lposcost_f``.

        Parameters
        ----------
        buf : np.ndarray
            Must be ``dtype=uint32`` and ``buf.shape[-1] == 4``.

        Returns
        -------
        np.ndarray
            Flat (1-D) structured array.
        """
        if buf.dtype != np.uint32:
            raise TypeError(f"buf must be uint32, got {buf.dtype}")
        if buf.shape[-1] != 4:
            raise ValueError(f"last dimension must be 4, got {buf.shape[-1]}")

        rec = buf.view(cls.DTYPE).reshape(-1)

        rec = append_fields(
            rec,
            ["lposfphi_f", "lposcost_f"],
            [
                cls.unpack_uint16_to_float(rec["lposfphi"]),
                cls.unpack_uint16_to_float(rec["lposcost"]),
            ],
            [np.float32, np.float32],
            usemask=False,
        )
        return rec

    @classmethod
    def view(cls, buf: np.ndarray) -> np.recarray:
        rec = cls.view_(buf)
        return rec.view(np.recarray)


    # ------------------------------------------------------------------
    # 6. Pretty-print a few records (useful in notebooks / REPL)
    # ------------------------------------------------------------------

    @classmethod
    def pprint(cls, rec: np.ndarray, n: int = 5) -> None:
        """Print the first *n* records in a readable table."""
        print(rec[:n])

    # ------------------------------------------------------------------
    # 7. Optional: create an *empty* array of a given length
    # ------------------------------------------------------------------
    @classmethod
    def empty(cls, size: int) -> np.ndarray:
        """Return a zero-filled structured array of length *size*."""
        return np.zeros(size, dtype=cls.DTYPE)
