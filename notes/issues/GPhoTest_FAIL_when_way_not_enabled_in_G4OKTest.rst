GPhoTest_FAIL_when_way_not_enabled_in_G4OKTest
===============================================

Issue 
--------

GPhoTest gets mismatches due to header only wy.npy that get written by G4OKTest when
way capture is not enabled for it.

::

    GPhoTest
    ...
    totVertices    116395  totFaces    202152 
    vtotVertices  63603714 vtotFaces 125348744 (virtual: scaling by transforms)
    vfacVertices   546.447 vfacFaces   620.072 (virtual to total ratio)
    2021-04-10 22:54:37.797 INFO  [455199] [main@61]  ox_path $TMP/G4OKTest/evt/g4live/natural/1/ox.npy ox 5000,4,4
    2021-04-10 22:54:37.797 INFO  [455199] [main@65]  wy_path $TMP/G4OKTest/evt/g4live/natural/1/wy.npy wy 5000,2,4
    2021-04-10 22:54:37.798 INFO  [455199] [GPho::wayConsistencyCheck@156]  mismatch_flags 5000 mismatch_index 4999
    2021-04-10 22:54:37.798 ERROR [455199] [GPho::setPhotons@114]  mismatch 9999
    GPhoTest: /home/blyth/opticks/ggeo/GPho.cc:118: void GPho::setPhotons(const NPY<float>*): Assertion `mismatch == 0' failed.


    O[blyth@localhost opticks]$ xxd $TMP/G4OKTest/evt/g4live/natural/1/wy.npy
    0000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
    0000010: 7227 3a20 273c 6634 272c 2027 666f 7274  r': '<f4', 'fort
    0000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    0000030: 652c 2027 7368 6170 6527 3a20 2835 3030  e, 'shape': (500
    0000040: 302c 2032 2c20 3429 2c20 7d20 2020 200a  0, 2, 4), }    .
    O[blyth@localhost opticks]$ 



G4OKTest
---------

By default running G4OKTest is not way enabled. Currently that produces header only wy.npy 
that crash numpy on trying to load::

    epsilon:g4ok blyth$ l /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/*/wy.npy
    8 -rw-r--r--  1 blyth  wheel  80 Apr 11 12:07 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/10/wy.npy
    8 -rw-r--r--  1 blyth  wheel  80 Apr 11 12:07 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/9/wy.npy
    8 -rw-r--r--  1 blyth  wheel  80 Apr 11 12:06 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/8/wy.npy
    8 -rw-r--r--  1 blyth  wheel  80 Apr 11 12:06 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/7/wy.npy
    8 -rw-r--r--  1 blyth  wheel  80 Apr 11 12:06 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/6/wy.npy
    8 -rw-r--r--  1 blyth  wheel  80 Apr 11 12:06 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/5/wy.npy
    8 -rw-r--r--  1 blyth  wheel  80 Apr 11 12:06 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/4/wy.npy
    8 -rw-r--r--  1 blyth  wheel  80 Apr 11 12:06 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/3/wy.npy
    8 -rw-r--r--  1 blyth  wheel  80 Apr 11 12:06 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/2/wy.npy
    8 -rw-r--r--  1 blyth  wheel  80 Apr 11 12:06 /tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/wy.npy

    epsilon:g4ok blyth$ ipython

    In [1]: a = np.load("/tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/wy.npy")                                                                                                                          
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
    <ipython-input-1-e7034ef9cfe7> in <module>
    ----> 1 a = np.load("/tmp/blyth/opticks/G4OKTest/evt/g4live/natural/1/wy.npy")

    ~/miniconda3/lib/python3.7/site-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding)
        438             else:
        439                 return format.read_array(fid, allow_pickle=allow_pickle,
    --> 440                                          pickle_kwargs=pickle_kwargs)
        441         else:
        442             # Try a pickle

    ~/miniconda3/lib/python3.7/site-packages/numpy/lib/format.py in read_array(fp, allow_pickle, pickle_kwargs)
        769             array = array.transpose()
        770         else:
    --> 771             array.shape = shape
        772 
        773     return array

    ValueError: cannot reshape array of size 0 into shape (5000,2,4)

    In [2]:                                    


okc::

    1983 void OpticksEvent::saveWayData()
    1984 {
    1985     NPY<float>* wy = getWayData();
    1986     if(wy) wy->save(m_pfx, "wy", m_typ,  m_tag, m_udet);
    1987 }
    1988 


    1816 void OpticksEvent::save()
    1817 {   
    1818     //std::raise(SIGINT); 
    1819     //const char* dir =  m_event_spec->getDir() ; 
    1820     const char* dir =  getDir() ;
    1821     LOG(info) << dir ;
    1822     
    1823     OK_PROFILE("_OpticksEvent::save");
    1824 
    1825     
    1826     LOG(LEVEL) 
    1827         << description("") << getShapeString()
    1828         << " dir " << dir
    1829         ;
    1830     
    1831     bool production = m_ok->isProduction() ;
    1832     
    1833     if(production)
    1834     {   
    1835         if(m_ok->hasOpt("savehit")) saveHitData();  // FOR production hit check
    1836         saveTimes();
    1837     }
    1838     else
    1839     {   
    1840         saveHitData();
    1841         saveHiyData();   
    1842         saveNopstepData();
    1843         saveGenstepData();
    1844         savePhotonData();
    1845         saveSourceData();
    1846         saveRecordData();
    1847         saveSequenceData();
    1848         saveDebugData();
    1849         saveWayData();
    1850         
    1851         //saveSeedData();
    1852         saveIndex();
    1853         
    1854         recordDigests();
    1855         saveDomains();
    1856         saveParameters();
    1857     }
    1858 

    0372 NPY<float>* OpticksEvent::getWayData() const
     373 {
     374     return m_way_data ;
     375 }

    0344 bool OpticksEvent::hasWayData() const
     345 {
     346     return m_way_data && m_way_data->hasData() ;
     347 }



The issue is how to handle an optional part of the OpticksEvent.
Try just not saving when no wy data::

    1983 void OpticksEvent::saveWayData()
    1984 {
    1985     NPY<float>* wy = getWayData();
    1986     if(wy && wy->hasData()) wy->save(m_pfx, "wy", m_typ,  m_tag, m_udet);
    1987 }   


