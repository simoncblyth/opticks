ckm-okg4-initial-comparisons-reveal-idom-bug
==================================================

context
----------

Just shaken out initial obvious bugs in getting Geant4 running from gensteps to work. 
Now some comparisons to find some less obvious bugs. 


geocache
-----------

::

    [blyth@localhost 1]$ kcd
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1
    rundate
    20190529_220906
    runstamp
    1559138946
    argline
     /home/blyth/local/opticks/lib/CerenkovMinimal



so : source photons, G4(full)-G4(optical) comparison 
--------------------------------------------------------

Shows good match in the external G4 Cerenkov generation from gensteps.

::

    [blyth@localhost 1]$ np.py source/evt/g4live/natural/-1/so.npy tmp/blyth/OKG4Test/evt/g4live/natural/-1/so.npy
    a :                          source/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : d1b4242225f7ffc7f0ad38a9669562a4 : 20190529-2209 
    b :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : 882f44b7864bfcde55fe2ebe922895e5 : 20190530-2044 
     max(a-b)   5.96e-08  min(a-b)  -5.96e-08 


G4(full) available npy : very few, as CerenkovMinimal is simple example without the full CFG4 instrumentation
-----------------------------------------------------------------------------------------------------------------

::

    [blyth@localhost 1]$ np.py source/evt/g4live/natural/-1
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/source/evt/g4live/natural/-1
    . :                          source/evt/g4live/natural/-1/ht.npy :          (108, 4, 4) : f151301a12d1874e9447fd916e7f8719 : 20190529-2209 
    . :                          source/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : d1b4242225f7ffc7f0ad38a9669562a4 : 20190529-2209 


G4(optical) : full Opticks Event 
-----------------------------------

::

    [blyth@localhost 1]$ np.py tmp/blyth/OKG4Test/evt/g4live/natural/-1
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/tmp/blyth/OKG4Test/evt/g4live/natural/-1
             tmp/blyth/OKG4Test/evt/g4live/natural/-1/report.txt :                   35 : 90ca98a06ac66bbd07b39eb2d0f0143e : 20190530-2044 
    . :            tmp/blyth/OKG4Test/evt/g4live/natural/-1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190530-2044 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190530-2044 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/ht.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190530-2044 
    . :            tmp/blyth/OKG4Test/evt/g4live/natural/-1/idom.npy :            (1, 1, 4) : 50c5e9a47bb603b419cf5a57c7b7c77e : 20190530-2044 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/ox.npy :          (221, 4, 4) : 0c933fd9fdab9d2975af9e6871351e46 : 20190530-2044 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/ph.npy :          (221, 1, 2) : 0a50e4992b98714e0391cd6d8deadc9e : 20190530-2044 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/ps.npy :          (221, 1, 4) : 2f17ee76054cc1040f30bee0a8a0153e : 20190530-2044 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/rs.npy :      (221, 10, 1, 4) : 629500c344dc05dbc6777ccf6f386fe5 : 20190530-2044 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/rx.npy :      (221, 10, 2, 4) : 2ce8d2aafab81f6d6f0e6a1cc1877646 : 20190530-2044 
    . :              tmp/blyth/OKG4Test/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : 882f44b7864bfcde55fe2ebe922895e5 : 20190530-2044 
    tmp/blyth/OKG4Test/evt/g4live/natural/-1/20190530_150253/report.txt :                   35 : 1805bf3c30c690f3fd13849e933df614 : 20190530-1502 
    tmp/blyth/OKG4Test/evt/g4live/natural/-1/20190530_151658/report.txt :                   35 : 72f84aaa5b76bb49651e6360e3564bc2 : 20190530-1517 
    tmp/blyth/OKG4Test/evt/g4live/natural/-1/20190530_151715/report.txt :                   35 : 6e53d72d3f5f99f469bba12e7bd1f6d9 : 20190530-1517 
    tmp/blyth/OKG4Test/evt/g4live/natural/-1/20190530_194634/report.txt :                   35 : ba5f6fe164d9c8e9e58140bf96894f6b : 20190530-1946 
    tmp/blyth/OKG4Test/evt/g4live/natural/-1/20190530_195703/report.txt :                   35 : 961f77d00803d2bf8ee38d85ecd7b8d3 : 20190530-1957 
    tmp/blyth/OKG4Test/evt/g4live/natural/-1/20190530_204454/report.txt :                   35 : 90ca98a06ac66bbd07b39eb2d0f0143e : 20190530-2044 
    [blyth@localhost 1]$ 



Opticks events from 1st and 2nd executables
-----------------------------------------------

::

    [blyth@localhost 1]$ np.py tmp/blyth/OKG4Test/evt/g4live/natural/1
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/tmp/blyth/OKG4Test/evt/g4live/natural/1
              tmp/blyth/OKG4Test/evt/g4live/natural/1/report.txt :                   39 : 38b09f2b3aab8ac3438d32352fc19a61 : 20190530-2044 
    . :             tmp/blyth/OKG4Test/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190530-2044 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190530-2044 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : a5dcecca8a61f6ef3e324edac8f36361 : 20190530-2044 
    . :             tmp/blyth/OKG4Test/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : 5cbfe278142f35f714d1487591c69ef9 : 20190530-2044   ** 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : bbf07d0da33758272b447ba44655decd : 20190530-2044 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 : 20190530-2044 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c : 20190530-2044 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 : 20190530-2044 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 : 20190530-2044 

    blyth@localhost 1]$ np.py tmp/blyth/OKG4Test/evt/g4live/natural/1    ## after fix, 
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/tmp/blyth/OKG4Test/evt/g4live/natural/1
              tmp/blyth/OKG4Test/evt/g4live/natural/1/report.txt :                   39 : dd79ea170250c1665eea35c40c1add4a : 20190530-2246 
    . :             tmp/blyth/OKG4Test/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : a5dcecca8a61f6ef3e324edac8f36361 : 20190530-2246 
    . :             tmp/blyth/OKG4Test/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190530-2246   ####
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : bbf07d0da33758272b447ba44655decd : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 : 20190530-2246 
    . :               tmp/blyth/OKG4Test/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 : 20190530-2246 



    [blyth@localhost 1]$ np.py source/evt/g4live/natural/1
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/source/evt/g4live/natural/1
                          source/evt/g4live/natural/1/report.txt :                   39 : d4fea377a8393acb6d4df6299026899f : 20190529-2209 
    . :                         source/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190529-2209 
    . :                           source/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190529-2209 
    . :                           source/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : a5dcecca8a61f6ef3e324edac8f36361 : 20190529-2209 
    . :                         source/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : d7acfbeef40f01f422f9c4aec021dc17 : 20190529-2209   ** 
    . :                           source/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : bbf07d0da33758272b447ba44655decd : 20190529-2209 
    . :                           source/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 : 20190529-2209 
    . :                           source/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c : 20190529-2209 
    . :                           source/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 : 20190529-2209 
    . :                           source/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 : 20190529-2209 

    [blyth@localhost 1]$ np.py source/evt/g4live/natural/1
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/source/evt/g4live/natural/1
                          source/evt/g4live/natural/1/report.txt :                   39 : d5f1b99ab6ad378bb1000f9e92f32513 : 20190530-2247 
    . :                         source/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : a5dcecca8a61f6ef3e324edac8f36361 : 20190530-2247 
    . :                         source/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190530-2247   #### 
    . :                           source/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : bbf07d0da33758272b447ba44655decd : 20190530-2247 
    . :                           source/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c : 20190530-2247 
    . :                           source/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 : 20190530-2247 







FIXED : Unexpected idom.x difference : An uninitialized bug
----------------------------------------------------------------

::

    [blyth@localhost 1]$ np.py source/evt/g4live/natural/1/idom.npy tmp/blyth/OKG4Test/evt/g4live/natural/1/idom.npy -v -iF
    a :                         source/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : d7acfbeef40f01f422f9c4aec021dc17 : 20190529-2209 
    (1, 1, 4)
    ('i32\n', array([[[1299607403,    3000000,          9,         10]]], dtype=int32))
    b :             tmp/blyth/OKG4Test/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : 5cbfe278142f35f714d1487591c69ef9 : 20190530-2044 
    (1, 1, 4)
    ('i32\n', array([[[795374593,   3000000,         9,        10]]], dtype=int32))
     max(a-b)   5.04e+08  min(a-b)          0 
    [blyth@localhost 1]$ 


::

    075 void OpticksDomain::updateBuffer()
     76 {
     77     NPY<float>* fdom = getFDomain();
     78     if(fdom)
     79     {
     80         fdom->setQuad(m_space_domain     , 0);
     81         fdom->setQuad(m_time_domain      , 1);
     82         fdom->setQuad(m_wavelength_domain, 2);
     83     }
     84     else
     85     {
     86         LOG(warning) << "OpticksDomain::updateBuffer fdom NULL " ;
     87     }
     88 
     89     NPY<int>* idom = getIDomain();
     90     if(idom)
     91         idom->setQuad(m_settings, 0 );
     92     else
     93         LOG(warning) << "OpticksDomain::updateBuffer idom NULL " ;
     94    
     95 }
     96 


    132 unsigned OpticksDomain::getMaxRng() const
    133 {
    134     return m_settings.y ;
    135 }
    136 void OpticksDomain::setMaxRng(unsigned maxrng)
    137 {
    138     m_settings.y = maxrng ;
    139 }
    140 
    141 unsigned int OpticksDomain::getMaxBounce() const
    142 {
    143     return m_settings.z ;
    144 }
    145 void OpticksDomain::setMaxBounce(unsigned int maxbounce)
    146 {
    147     m_settings.z = maxbounce ;
    148 }
    149 
    150 unsigned int OpticksDomain::getMaxRec() const
    151 {
    152     return m_settings.w ;
    153 }
    154 void OpticksDomain::setMaxRec(unsigned int maxrec)
    155 {
    156     m_settings.w = maxrec ;
    157 }
    158 
    159 


Looks like an uninitialized ivec4. YEP, after fix in OpticksDomain::

    [blyth@localhost 1]$ np.py source/evt/g4live/natural/1/idom.npy tmp/blyth/OKG4Test/evt/g4live/natural/1/idom.npy -v -iF
    a :                         source/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190530-2247 
    (1, 1, 4)
    ('i32\n', array([[[      0, 3000000,       9,      10]]], dtype=int32))
    b :             tmp/blyth/OKG4Test/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190530-2246 
    (1, 1, 4)
    ('i32\n', array([[[      0, 3000000,       9,      10]]], dtype=int32))
     max(a-b)          0  min(a-b)          0 
    [blyth@localhost 1]$ 
    [blyth@localhost 1]$ 




