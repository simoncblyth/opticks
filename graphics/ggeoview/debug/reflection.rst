Reflection Debug
==================


BoxInBox Pyrex block in Vacuum lit by spherical focus light targetting single point::

   ggv.sh  --test --save \
        --eye 0.5,0.5,0.0 \
        --animtimemax 7 \
        --testconfig "mode=BoxInBox_dimensions=500,300,0,0_boundary=Rock//perfectAbsorbSurface/Vacuum_boundary=Vacuum///Pyrex_" \
        --torchconfig "polz=spol_frame=1_type=refltest_source=0,0,300_target=0,0,1_radius=102_zenithazimuth=0,0.5,0,1_material=Vacuum" \
         $*


Rec.py seqhis selection 

* TORCH BR SA : one reflection then gets surface absorbed  
* TORCH BR AB : one reflection then bulk absorbed


Some 6 percent miss the target::

    In [78]: pos1[:,2]
    Out[78]: array([ 299.997,  299.997,  299.997, ...,  299.997,  299.997,  299.997], dtype=float32)

    In [91]: pos1[pos1[:,2] != pos1[0,2]].shape
    Out[91]: (4728, 3)

    In [82]: p1z
    Out[82]: array([ 299.997,  299.997,  299.997, ...,  299.997,  299.997,  299.997], dtype=float32)

    In [83]: p1z[p1z<299]
    Out[83]: array([ 12.009,  47.792, -18.876, ..., -15.503, -42.802,  68.941], dtype=float32)

    In [87]: 4728./82600
    Out[87]: 0.05723970944309927


