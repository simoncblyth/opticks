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

EFFICIENCY_COLLECT = 0x1 << 13


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


        fphi = cls.unpack_uint16_to_float(rec["lposfphi"])
        cost = cls.unpack_uint16_to_float(rec["lposcost"])
        phi = ( fphi * 2.0 - 1. ) * np.pi  # scuda.h:phi_from_fphi



        rec = append_fields(
            rec
            ,
            [
               "lposfphi_f",
               "lposcost_f",
               "phi"
            ]
            ,
            [
                fphi,
                cost,
                phi
            ]
            ,
            [
            np.float32,
            np.float32,
            np.float32,
            ]
            ,
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






class SPhotonLite_Merge:
    """
    Select hits and merge them using a NumPy implementation
    that follows the thrust::reduce_by_key based approach
    of sysrap/SPM.cu SPM::merge_partial_select

    For a small number of photons (eg M1)
    this precisely duplicates in python the hit selection and merging
    done with the CUDA impl
    """

    def __init__(self, tw=1.0, select_mask = EFFICIENCY_COLLECT ):
        self.tw = tw
        self.select_mask = select_mask

    def __call__(self, _pl):
        tw = self.tw
        select_mask = self.select_mask

        # Step 1: Filter hits (same as (p.flagmask & select_mask) == 0)
        flagmask = _pl[:,3]

        select = (flagmask & select_mask) != 0
        if not np.any(select): return np.zeros(0, dtype=_pl.dtype )

        _hl = _pl[select]
        # _hl = _pl[ np.where( _pl[:,3] & (0x1 << 13) )]

        # Step 2: Build 64-bit key: (pmt_id << 48) | bucket
        identity = _hl[:,0] & 0xFFFF                    # lower 16 bits = PMT ID
        time = _hl[:,1].view(np.float32)
        bucket = np.floor(time / tw).astype(np.uint64)
        key = (identity.astype(np.uint64) << 48) | bucket

        # key = ( ( _hl[:,0] & 0xFFFF ).astype(np.uint64) << 48 ) | np.floor( _hl[:,1].view(np.float32)/1. ).astype(np.uint64)

        # Step 3: Sort by key (exactly what Thrust does internally)
        sort_idx = np.argsort(key, kind='stable')  # stable = same as thrust::stable_sort_by_key
        key_sorted = key[sort_idx]
        _hl_sorted = _hl[sort_idx]


        # Step 4: Find group boundaries
        key_diff = np.diff(key_sorted, prepend=key_sorted[0]-1)     # force first group start
        group_start = np.where(key_diff != 0)[0]

        # Number of output groups
        n_groups = len(group_start)

        # Step 5: Reduce each group (vectorized version of sphotonlite_reduce_op)
        out = np.zeros( (n_groups,4), dtype=_pl.dtype )

        # Take first hit in group as base (like your CUDA reduce does with 'a')
        first_in_group = group_start

        out[:] = _hl_sorted[first_in_group]

        # Now reduce the rest of each group
        for i in range(n_groups):
            start = group_start[i]
            end = group_start[i+1] if i+1 < n_groups else len(key)

            if end - start == 1:
                continue

            group_slice = slice(start, end)

            # min time
            out[i,1] = np.min(_hl_sorted[group_slice,1].view(np.float32)).view(np.uint32)

            # OR all flagmasks
            out[i,3] |= np.bitwise_or.reduce(_hl_sorted[group_slice,3])

            all_identity = _hl_sorted[group_slice,0] & 0xffff

            # sum hitcounts
            sum_hitcount = np.sum(_hl_sorted[group_slice,0] >> 16, dtype=np.uint32)
            out[i,0] = sum_hitcount << 16 | all_identity[0]
        pass
        return out
    pass
pass



