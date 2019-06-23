opticksdata-cleanup
==========================


ISSUE : repository contains geocache files that should not have been comitted
---------------------------------------------------------------------------------

This results in a mesh of merge conflicts from geocache creations done on different machines.

* https://bitbucket.org/simoncblyth/opticksdata/src/default/export/DayaBay_VGDX_20140414-1300/extras/
* https://bitbucket.org/simoncblyth/opticksdata/src/default/export/DayaBay_VGDX_20140414-1300/extras/186/


Objective
------------

* opticksdata needs to be for real data only, not derived geocache files, ie gdml/dae a few json
* operational geocache (even in legacy route) should not be using a repository directory


How to get there ?
--------------------

1. Adjust legacy mode resource handling to read/write from /home/blyth/local/opticks/geocache
   with opticksdata only used for primaries .gdml .dae 

2. verify with strace

3. "hg rm" geocache transients from opticksdata 


OKTest
--------

::

   OKTest --xanalytic --gltf 1

   strace -o /tmp/strace.log -e open OKTest --xanalytic --gltf 1


Seems the extras are not loaded::

    [blyth@localhost opticks]$ strace.py -f O_RDONLY | grep opticksdata
     /home/blyth/local/opticks/opticksdata/config/opticksdata.ini                     :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay/ChromaMaterialMap.json      :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GBndLib/GBndLibIndex.npy :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay/GMaterialLib/color.json     :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay/GMaterialLib/abbrev.json    :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay/GMaterialLib/order.json     :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMaterialLib/GMaterialLib.npy :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMaterialLib/GPropertyLibMetadata.json :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GItemList/GMaterialLib.txt :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay/GSurfaceLib/color.json      :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay/GSurfaceLib/order.json      :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GSurfaceLib/GSurfaceLib.npy :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GSurfaceLib/GPropertyLibMetadata.json :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GItemList/GSurfaceLib.txt :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GSurfaceLib/GSurfaceLibOptical.npy :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GScintillatorLib.npy :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GItemList/GScintillatorLib.txt :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GdDopedLS/ABSLENGTH.npy :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GdDopedLS/AlphaFASTTIMECONSTANT.npy :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GdDopedLS/AlphaSLOWTIMECONSTANT.npy :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GdDopedLS/AlphaYIELDRATIO.npy :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GdDopedLS/FASTCOMPONENT.npy :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GdDopedLS/FASTTIMECONSTANT.npy :                  O_RDONLY :    -1 
     /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GdDopedLS/GammaFASTTIMECONSTANT.npy :                  O_RDONLY :    -1 






