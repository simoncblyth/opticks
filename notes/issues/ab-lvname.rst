ab-lvname : MeshIndex lvIdx to lvName mapping is totally off for live geometry 
=================================================================================

Noticed this from getting incorrect lvNames from ab-prim when 
it was using IDPATH from the direct geocache. 

::

    epsilon:1 blyth$ ab-;ab-lv2name
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/104
    -rw-r--r--  1 blyth  staff  9448 Aug  7 13:18 MeshIndex/GItemIndexSource.json
      0 : near_top_cover_box0xc23f970 
      1 : RPCStrip0xc04bcb0 
      2 : RPCGasgap140xbf4c660 
      3 : RPCBarCham140xc2ba760 
      4 : RPCGasgap230xbf50468 
    ...
    244 : near-radslab-box-80xcd308c0 
    245 : near-radslab-box-90xcd31ea0 
    246 : near_hall_bot0xbf3d718 
    247 : near_rock0xc04ba08 
    248 : WorldBox0xc15cf40 
    /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/0dce832a26eb41b58a000497a3127cb8/1
    -rw-r--r--  1 blyth  staff  9472 Aug  7 13:19 MeshIndex/GItemIndexSource.json
      0 : WorldBox0xc15cf40 
      1 : near_rock0xc04ba08 
      2 : near_hall_top_dwarf0xc0316c8 
      3 : near_top_cover_box0xc23f970 
      4 : RPCMod0xc13bfd8 
    ...
    244 : near-radslab-box-50xccefd60 
    245 : near-radslab-box-60xccefda0 
    246 : near-radslab-box-70xccefde0 
    247 : near-radslab-box-80xcd308c0 
    248 : near-radslab-box-90xcd31ea0 
    epsilon:1 blyth$ 


MeshIndex is written by GMeshLib
----------------------------------



