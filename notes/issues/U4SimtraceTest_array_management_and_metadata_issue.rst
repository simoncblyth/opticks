U4SimtraceTest_array_management_and_metadata_issue
=====================================================

Looks like more of the geometry config should feed into the BASE directory, 
to avoid mixing outputs from separate runs in the same directory.
Can also add metadata onto trs.npy array::

    epsilon:tests blyth$ l /tmp/blyth/opticks/GEOM/FewPMT/U4SimtraceTest/1/
    total 16
    8 -rw-r--r--   1 blyth  wheel  1152 Mar 15 19:53 trs.npy
    8 -rw-r--r--   1 blyth  wheel   129 Mar 15 19:53 trs_names.txt

    0 drwxr-xr-x   9 blyth  wheel   288 Mar 15 15:54 hama_pmt_solid_1_4
    0 drwxr-xr-x   9 blyth  wheel   288 Mar 15 15:54 hama_inner_solid_1_4
    0 drwxr-xr-x   9 blyth  wheel   288 Mar 15 15:54 hama_shield_solid
    0 drwxr-xr-x   9 blyth  wheel   288 Mar 15 15:54 hama_grid_solid
    0 drwxr-xr-x   9 blyth  wheel   288 Mar 15 15:54 hama_dynode_tube_solid
    0 drwxr-xr-x   9 blyth  wheel   288 Mar 15 15:54 hama_inner_ring_solid
    0 drwxr-xr-x   9 blyth  wheel   288 Mar 15 15:54 hama_inner_edge_solid
    0 drwxr-xr-x   9 blyth  wheel   288 Mar 15 15:54 hama_outer_edge_solid
    0 drwxr-xr-x   9 blyth  wheel   288 Mar 15 15:54 hama_plate_solid

    0 drwxr-xr-x   9 blyth  wheel   288 Mar  8 12:25 Rock_solid
    0 drwxr-xr-x   9 blyth  wheel   288 Mar  8 12:25 Water_solid
    0 drwxr-xr-x   9 blyth  wheel   288 Mar  8 12:25 nnvt_pmt_solid_head
    0 drwxr-xr-x   9 blyth  wheel   288 Mar  8 12:25 nnvt_inner_solid_head
    0 drwxr-xr-x   9 blyth  wheel   288 Mar  8 12:25 nnvt_mcp_solid
    0 drwxr-xr-x   9 blyth  wheel   288 Mar  8 12:25 nnvt_tube_solid
    0 drwxr-xr-x   9 blyth  wheel   288 Mar  8 12:25 nnvt_plate_solid
    0 drwxr-xr-x   9 blyth  wheel   288 Mar  8 12:25 nnvt_edge_solid
    0 drwxr-xr-x  21 blyth  wheel   672 Mar  8 12:25 .
    0 drwxr-xr-x   4 blyth  wheel   128 Feb 27 16:20 ..
    epsilon:tests blyth$ 


::

    epsilon:u4 blyth$ cat /tmp/blyth/opticks/GEOM/FewPMT/U4SimtraceTest/1/trs_names.txt 
    Rock_solid
    Water_solid
    nnvt_pmt_solid_head
    nnvt_inner_solid_head
    nnvt_edge_solid
    nnvt_plate_solid
    nnvt_tube_solid
    nnvt_mcp_solid
    epsilon:u4 blyth$ 



U4SimtraceTest.sh::

    134 export VERSION=${N:-0}
    135 export GEOM=FewPMT
    136 export GEOMFOLD=/tmp/$USER/opticks/GEOM/$GEOM
    137 export BASE=$GEOMFOLD/$bin
    138 export FOLD=$BASE/$VERSION   ## controls where the executable writes geometry


HMM: Looking at FewPMT.sh theres lots of things that change geometry, eg POM, FewPMT_GEOMList


