NPFold_NP_WriteNames_SIGSEGV
=============================



Crazy num_names in NP::WriteNames -> time for a clean builds of opticks + junosw 
-------------------------------------------------------------------------------------

::

    (gdb) f 6
    #6  0x00007fffc7e8572e in NPFold::_save_subfold_r (this=0xa461750, 
        base=0x40050810 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim/extra/jpmt") at /home/blyth/junotop/opticks/sysrap/NPFold.h:1462
    1462	        sf->save(base, f );  
    (gdb) p f
    $6 = 0xa542200 "PMTParamData"
    (gdb) f 5
    #5  0x00007fffc7e85386 in NPFold::save (this=0xa461810, base_=0x40050810 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim/extra/jpmt", 
        rel=0xa542200 "PMTParamData") at /home/blyth/junotop/opticks/sysrap/NPFold.h:1367
    1367	    save(base.c_str()); 
    (gdb) f 4
    #4  0x00007fffc7e85476 in NPFold::save (this=0xa461810, 
        base_=0x40050d40 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim/extra/jpmt/PMTParamData")
        at /home/blyth/junotop/opticks/sysrap/NPFold.h:1394
    1394	    _save(base) ; 
    (gdb) p base
    $7 = 0x40051410 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim/extra/jpmt/PMTParamData"
    (gdb) f 3
    #3  0x00007fffc7e855d3 in NPFold::_save (this=0xa461810, 
        base=0x40051410 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim/extra/jpmt/PMTParamData")
        at /home/blyth/junotop/opticks/sysrap/NPFold.h:1425
    1425	    if(names.size() > 0 )  NP::WriteNames(base, NAMES, names) ; 
    (gdb) f 2
    #2  0x00007fffc7e7e183 in NP::WriteNames (dir=0x40051410 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim/extra/jpmt/PMTParamData", 
        name=0x7fffc7f76c79 "NPFold_names.txt", names=..., num_names_=0, append=false) at /home/blyth/junotop/opticks/sysrap/NP.hh:5451
    5451	    WriteNames(path.c_str(), names, num_names_, append  ); 
    (gdb) f 1
    #1  0x00007fffc7e7e319 in NP::WriteNames (
        path=0x40050e60 "/home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0/CSGFoundry/SSim/extra/jpmt/PMTParamData/NPFold_names.txt", names=..., num_names_=0, 
        append=false) at /home/blyth/junotop/opticks/sysrap/NP.hh:5490
    5490	    for( unsigned i=0 ; i < num_names ; i++) stream << names[i] << std::endl ; 
    (gdb) p num_names
    $8 = 1627389952
    (gdb) 



jcv _PMTParamData::

     44 inline NPFold* _PMTParamData::serialize() const
     45 {
     46     NPFold* f = new NPFold ;
     47     f->add("pmtCat", NPX::ArrayFromDiscoMap<int>(data.m_pmt_categories));
     48     return f ;
     49 }


HMM: no reason for NPFold_names.txt anyhow ? 

