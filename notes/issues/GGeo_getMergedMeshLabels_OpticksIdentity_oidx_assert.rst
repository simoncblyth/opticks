GGeo_getMergedMeshLabels_OpticksIdentity_oidx_assert
=========================================================


Getting error on P (geocache from tds3) but not on G(geocache from geocache-create from GDML)::

    gdb GGeoTest 
    ...

    (gdb) f 5
    #5  0x00007ffff7b39408 in GGeo::getMergedMeshLabel[abi:cxx11](unsigned int, bool, bool) const (this=0x667920, ridx=2, numvol=true, trim=true) at /home/blyth/opticks/ggeo/GGeo.cc:943
    943	    glm::uvec4 id = getIdentity(ridx, pidx, oidx, check); 
    (gdb) p id
    $2 = {{x = 194244, r = 194244, s = 194244}, {y = 16777216, g = 16777216, t = 16777216}, {z = 7602201, b = 7602201, p = 7602201}, {w = 0, a = 0, q = 0}}
    (gdb) f 4
    #4  0x00007ffff7b3d05f in GGeo::getIdentity (this=0x667920, ridx=2, pidx=0, oidx=0, check=true) at /home/blyth/opticks/ggeo/GGeo.cc:1766
    1766	        assert( OpticksIdentity::OffsetIndex(triplet)    == oidx ); 
    (gdb) p oidx
    $3 = 0
    (gdb) p triplet
    $4 = 33554433
    (gdb) p triplet & 0xff
    $5 = 1
    (gdb) p (triplet & 0xffff00) >> 8
    $6 = 0
    (gdb) p (triplet & 0xff000000) >> 24
    $7 = 2
    (gdb) 

