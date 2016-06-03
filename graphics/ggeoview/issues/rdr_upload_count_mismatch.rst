[FIXED] Rdr upload count mismatch
====================================

* fixed by removing the flattening in OpticksEvent::load


CFG4 load count mismatch assert::

    simon:geant4_opticks_integration blyth$ ggv-pmt-test --cfg4  --load
    ...

    [2016-Jun-03 11:29:10.615177]:info: Rdr::address (glVertexAttribPointer)        rec name rpos type                SHORT index 0 norm  size 4 stride 8 offset_ 0
    [2016-Jun-03 11:29:10.615355]:info: Rdr::address (glVertexAttribPointer)        rec name rpol type        UNSIGNED_BYTE index 1 norm  size 4 stride 8 offset_ 2
    [2016-Jun-03 11:29:10.615495]:info: Rdr::address (glVertexAttribPointer)        rec name rflg type       UNSIGNED_SHORT index 2 norm  size 2 stride 8 offset_ 6
    [2016-Jun-03 11:29:10.615636]:info: Rdr::address (glVertexAttribPointer)        rec name rflq type        UNSIGNED_BYTE index 4 norm  size 4 stride 8 offset_ 6

    [2016-Jun-03 11:29:10.627441]:info: Rdr::address (glVertexAttribPointer)     altrec name rpos type                SHORT index 0 norm  size 4 stride 8 offset_ 0
    [2016-Jun-03 11:29:10.627601]:info: Rdr::address (glVertexAttribPointer)     altrec name rpol type        UNSIGNED_BYTE index 1 norm  size 4 stride 8 offset_ 2
    [2016-Jun-03 11:29:10.627744]:info: Rdr::address (glVertexAttribPointer)     altrec name rflg type       UNSIGNED_SHORT index 2 norm  size 2 stride 8 offset_ 6
    [2016-Jun-03 11:29:10.627866]:info: Rdr::address (glVertexAttribPointer)     altrec name rflq type        UNSIGNED_BYTE index 4 norm  size 4 stride 8 offset_ 6

    [2016-Jun-03 11:29:10.638947]:info: Rdr::address (glVertexAttribPointer)     devrec name rpos type                SHORT index 0 norm  size 4 stride 8 offset_ 0
    [2016-Jun-03 11:29:10.639119]:info: Rdr::address (glVertexAttribPointer)     devrec name rpol type        UNSIGNED_BYTE index 1 norm  size 4 stride 8 offset_ 2
    [2016-Jun-03 11:29:10.639261]:info: Rdr::address (glVertexAttribPointer)     devrec name rflg type       UNSIGNED_SHORT index 2 norm  size 2 stride 8 offset_ 6
    [2016-Jun-03 11:29:10.639403]:info: Rdr::address (glVertexAttribPointer)     devrec name rflq type        UNSIGNED_BYTE index 4 norm  size 4 stride 8 offset_ 6
    [2016-Jun-03 11:29:10.639569]:info: Rdr::upload glBufferData   sequence_attr phis count   500000 shape           500000,1,2 buffer_id    10 data      0x127a5e000 hasData     Y nbytes    8000000 GL_STATIC_DRAW
    [2016-Jun-03 11:29:10.646108]:info: Rdr::upload glBufferData     phosel_attr psel count   500000 shape           500000,1,4 buffer_id    11 data      0x128200000 hasData     Y nbytes    2000000 GL_STATIC_DRAW

    [2016-Jun-03 11:29:10.647784]:fatal: Rdr::upload COUNT MISMATCH  tag rec mvn recsel_attr expected  10000000 found 5000000
    [2016-Jun-03 11:29:10.647894]:info: Rdr::dump_uploads_table Rdr tag: rec
        record_attr 0/ 4 vnpy       rpos  10000000 npy 5000000,2,4 npy.hasData 1
        record_attr 1/ 4 vnpy       rpol  10000000 npy 5000000,2,4 npy.hasData 1
        record_attr 2/ 4 vnpy       rflg  10000000 npy 5000000,2,4 npy.hasData 1
        record_attr 3/ 4 vnpy       rflq  10000000 npy 5000000,2,4 npy.hasData 1
        recsel_attr 0/ 1 vnpy       rsel   5000000 npy 5000000,1,4 npy.hasData 1
    Assertion failed: (count_match && "all buffers fed to the Rdr pipeline must have the same counts"), function upload, file /Users/blyth/env/graphics/oglrap/Rdr.cc, line 132.
    /Users/blyth/env/bin/op.sh: line 372: 10092 Abort trap: 6           /usr/local/opticks/bin/GGeoView --test --testconfig mode=PmtInBox_pmtpath=/usr/local/env/geant4/geometry/export/dpib/cfg4.6f627a3ec05405cbcfff6bd479fbdd37.dae/GMergedMesh/0_control=1,0,0,0_analytic=1_groupvel=0_shape=box_boundary=Rock//perfectAbsorbSurface/MineralOil_parameters=0,0,0,300 --torch --torchconfig type=disc_photons=500000_wavelength=380_frame=1_source=0,0,300_target=0,0,0_radius=100_zenithazimuth=0.0001,1,0,1_material=Vacuum_mode=_polarization= --timemax 10 --animtimemax 10 --cat PmtInBox --tag -4 --save --eye 0.0,-0.5,0.0 --geocenter --cfg4 --load
    simon:geant4_opticks_integration blyth$ 
    simon:geant4_opticks_integration blyth$ 

Huh ... was it flattened on load ?  Raw recsel is (500000, 10, 1, 4)

::

    In [1]: run pmt_test_evt.py 
    Evt(-4,"torch","PmtInBox","PmtInBox/torch/-4 : ", seqs="[]")
     fdom :            (3, 1, 4) : (metadata) 3*float4 domains of position, time, wavelength (used for compression) 
     idom :            (1, 1, 4) : (metadata) int domain 
       ox :       (500000, 4, 4) : (photons) final photon step 
       wl :            (500000,) : (photons) wavelength 
     post :          (500000, 4) : (photons) final photon step: position, time 
     dirw :          (500000, 4) : (photons) final photon step: direction, weight  
     polw :          (500000, 4) : (photons) final photon step: polarization, wavelength  
    flags :            (500000,) : (photons) final photon step: flags  
       c4 :            (500000,) : (photons) final photon step: dtype split uint8 view of ox flags 
    rx_raw :   (500000, 10, 2, 4) : (records) photon step records RAW:before reshaping 
       rx :   (500000, 10, 2, 4) : (records) photon step records 
       ph :       (500000, 1, 2) : (records) photon history flag/material sequence 
       ps :       (500000, 1, 4) : (photons) phosel sequence frequency index lookups (uniques 34) 
       rs :   (500000, 10, 1, 4) : (records) RAW recsel sequence frequency index lookups (uniques 34) 
      rsr :   (500000, 10, 1, 4) : (records) RESHAPED recsel sequence frequency index lookups (uniques 34) 



No shifting needed as all zero::

     485 void OpticksEvent::setRecselData(NPY<unsigned char>* recsel_data)
     486 {
     487     m_recsel_data = recsel_data ;
     488 
     489     if(!m_recsel_data) return ;
     490     //                                               j k l sz   type                norm   iatt   item_from_dim
     491     ViewNPY* rsel = new ViewNPY("rsel",m_recsel_data,0,0,0,4,ViewNPY::UNSIGNED_BYTE,false,  true, 2);
     492     // structured recsel array, means the count needs to come from product of 1st two dimensions, 
     493 
     494     m_recsel_attr = new MultiViewNPY("recsel_attr");
     495     m_recsel_attr->add(rsel);
     496 }


