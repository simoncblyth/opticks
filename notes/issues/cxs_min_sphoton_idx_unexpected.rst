cxs_min_sphoton_idx_unexpected
=================================

::

    In [26]: np.unique(  a.f.photon.view(np.uint32)[:,3,2], return_counts=True   )
    Out[26]: (array([         0, 2147483648], dtype=uint32), array([45192, 54808]))

Thats just the orient bit flipped one way or other::

    In [38]: 0x80000000
    Out[38]: 2147483648


Added at tail of qsim::generate_photon::

      p.set_idx(photon_id);



::

    epsilon:sysrap blyth$ opticks-f set_idx 
    ./ana/p.py:     73     SPHOTON_METHOD void set_idx( unsigned idx ){  orient_idx = ( orient_idx & 0x80000000u ) | ( 0x7fffffffu & idx ) ; }   // retain bit 31 asis 
    ./sysrap/squad.h:    SQUAD_METHOD void set_idx( unsigned  idx); 
    ./sysrap/squad.h:SQUAD_METHOD void quad4::set_idx( unsigned  idx)
    ./sysrap/ABR.py:    def _set_idx(self, idx):
    ./sysrap/ABR.py:    idx = property(_get_idx, _set_idx)
    ./sysrap/tests/squadTest.cc:void test_quad4_set_idx_set_prd_get_idx_get_prd()
    ./sysrap/tests/squadTest.cc:        p.set_idx(idx[0]); 
    ./sysrap/tests/squadTest.cc:        p.set_idx(idx[0]); 
    ./sysrap/tests/squadTest.cc:        p.set_idx(idx[0]); 
    ./sysrap/tests/squadTest.cc:    test_quad4_set_idx_set_prd_get_idx_get_prd(); 
    ./sysrap/sphoton.h:    SPHOTON_METHOD void set_idx( unsigned idx ){  orient_idx = ( orient_idx & 0x80000000u ) | ( 0x7fffffffu & idx ) ; }   // retain bit 31 asis 
    ./sysrap/SEvt.cc:3. start filling current_ctx.p sphoton with set_idx and set_flag  
    ./sysrap/SEvt.cc:    ctx.p.set_idx(idx); 
    ./qudarap/QSim.cu:    p.set_idx(idx); 
    ./dev/csg/intersect.py:    def _set_idx(self, u):
    ./dev/csg/intersect.py:    idx = property(_get_idx, _set_idx)
    epsilon:opticks blyth$ 


