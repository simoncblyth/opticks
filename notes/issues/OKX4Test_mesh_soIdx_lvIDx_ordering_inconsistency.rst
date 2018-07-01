OKX4Test_mesh_soIdx_lvIDx_ordering_inconsistency
===================================================

Hmm different IDPATH giving different lvIdx/soIdx to name mappings ???


::

    epsilon:~ blyth$ mesh.py 47 46 43 44 45
    INFO:__main__:Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 
     47 : pmt-hemi0xc0fed90 
     46 : pmt-hemi-vac0xc21e248 
     43 : pmt-hemi-cathode0xc2f1ce8 
     44 : pmt-hemi-bot0xc22a958 
     45 : pmt-hemi-dynode0xc346c50 

    epsilon:~ blyth$ mesh.py 0 248
    INFO:__main__:Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 
      0 : near_top_cover_box0xc23f970 
    248 : WorldBox0xc15cf40 
    epsilon:~ blyth$ 


    epsilon:0 blyth$ mesh.py 47 46 43 44 45
    INFO:__main__:Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
     47 : IavBotRib0xc2cd8b8 
     46 : OavBotHub0xc355030 
     43 : OcrGdsTfbInLso0xbfa2370 
     44 : OcrGdsInLso0xbfa2190 
     45 : OavBotRib0xbfaafe0 
    epsilon:0 blyth$ 

    epsilon:0 blyth$ mesh.py 54 55 56 57 58 
    INFO:__main__:Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
     54 : pmt-hemi0xc0fed90 
     55 : pmt-hemi-vac0xc21e248 
     56 : pmt-hemi-cathode0xc2f1ce8 
     57 : pmt-hemi-bot0xc22a958 
     58 : pmt-hemi-dynode0xc346c50 
    epsilon:0 blyth$ 

::

    epsilon:optickscore blyth$ ab-a
    epsilon:0 blyth$ mesh.py 0 248 
    INFO:__main__:Mesh for idpath : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/103 
      0 : near_top_cover_box0xc23f970 
    248 : WorldBox0xc15cf40 

    epsilon:0 blyth$ ab-b
    epsilon:0 blyth$ mesh.py 0 248 
    INFO:__main__:Mesh for idpath : /usr/local/opticks/geocache/OKX4Test_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1 
      0 : WorldBox0xc15cf40 
    248 : near-radslab-box-90xcd31ea0 




