ckm-okg4-tmp-user-paths-unwieldy-in-geocache
===============================================


issue : 2nd executable event paths are unweildy
-------------------------------------------------

The below is relative to geocache::

    tmp/blyth/OKG4Test/evt/g4live/natural/1/gs.npy 

Change to below after asserting that exename ends with Test, to avoid
messing with rest of geocache::

    OKG4Test/evt/g4live/natural/1/gs.npy 



::

    [blyth@localhost 1]$ np.py source/evt/g4live/natural/1/
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/source/evt/g4live/natural/1
      source/evt/g4live/natural/1/report.txt : 39 
        source/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 
          source/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 
          source/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : a5dcecca8a61f6ef3e324edac8f36361 
        source/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : d7acfbeef40f01f422f9c4aec021dc17 
          source/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : bbf07d0da33758272b447ba44655decd 
          source/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 
          source/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c 
          source/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 
          source/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 
    source/evt/g4live/natural/1/20190527_153801/report.txt : 37 
    source/evt/g4live/natural/1/20190527_173616/report.txt : 37 
    source/evt/g4live/natural/1/20190527_200301/report.txt : 37 
    source/evt/g4live/natural/1/20190527_205604/report.txt : 37 
    source/evt/g4live/natural/1/20190527_211410/report.txt : 37 
    source/evt/g4live/natural/1/20190527_213651/report.txt : 37 
    source/evt/g4live/natural/1/20190527_215438/report.txt : 37 
    source/evt/g4live/natural/1/20190528_145542/report.txt : 38 
    source/evt/g4live/natural/1/20190528_145945/report.txt : 38 
    source/evt/g4live/natural/1/20190528_185729/report.txt : 39 
    source/evt/g4live/natural/1/20190529_152823/report.txt : 39 
    source/evt/g4live/natural/1/20190529_213220/report.txt : 39 
    source/evt/g4live/natural/1/20190529_220841/report.txt : 39 
    source/evt/g4live/natural/1/20190529_220911/report.txt : 39 


    [blyth@localhost 1]$ np.py tmp/blyth/OKG4Test/evt/g4live/natural/1
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/tmp/blyth/OKG4Test/evt/g4live/natural/1
    tmp/blyth/OKG4Test/evt/g4live/natural/1/report.txt : 39 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : a5dcecca8a61f6ef3e324edac8f36361 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : d7acfbeef40f01f422f9c4aec021dc17 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : bbf07d0da33758272b447ba44655decd 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/20190530_150253/report.txt : 39 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/20190530_151658/report.txt : 39 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/20190530_151715/report.txt : 39 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/20190530_194634/report.txt : 39 
    tmp/blyth/OKG4Test/evt/g4live/natural/1/20190530_195703/report.txt : 39 
    [blyth@localhost 1]$ 



Wow resource handling needs a sledgehammer
----------------------------------------------


After exploring quite a rabbit hole find::


    493 void BOpticksResource::setupViaKey()
    ...
    580     const char* user = SSys::username();
    581     m_srcevtbase = makeIdPathPath("source");
    582     m_res->addDir( "srcevtbase", m_srcevtbase );
    583 
    584     const char* exename = SAr::Instance->exename();
    585     m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath("tmp", user, exename ) ;
    586     ///  should this always be KeySource ???
    587     ///      NO : KeySource means that the current executable is same as the exename 
    588     ///           enshrined into the geocache : ie the geocache creator  
    589 
    590     m_res->addDir( "evtbase", m_evtbase );
    591 


Some of the rabbithole
-------------------------

::

    1700 void OpticksEvent::saveGenstepData()
    1701 {
    1702     // genstep were formally not saved as they exist already elsewhere,
    1703     // however recording the gs in use for posterity makes sense
    1704     // 
    1705     NPY<float>* gs = getGenstepData();
    1706     if(gs) gs->save("gs", m_typ,  m_tag, m_udet);
    1707 }

     730 template <typename T>
     731 void NPY<T>::save(const char* tfmt, const char* source, const char* tag, const char* det)
     732 {
     733     //std::string path_ = NPYBase::path(det, source, tag, tfmt );
     734     std::string path_ = BOpticksEvent::path(det, source, tag, tfmt );
     735     save(path_.c_str());
     736 }

::

    blyth@localhost optickscore]$ opticks-f OPTICKS_EVENT_BASE
    ./ana/nload.py:DEFAULT_BASE = "$OPTICKS_EVENT_BASE/evt"
    ./ana/ncensus.py:    c = Census("$OPTICKS_EVENT_BASE/evt")
    ./ana/base.py:        self.setdefault("OPTICKS_EVENT_BASE",      os.path.join(keydir, "source" ))
    ./ana/base.py:        self.setdefault("OPTICKS_EVENT_BASE",      os.path.expandvars("/tmp/$USER/opticks") )
    ./boostrap/tests/BFileTest.cc:    ss.push_back("$OPTICKS_EVENT_BASE/evt/dayabay/cerenkov/1") ; 
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE_NOTAG = "$OPTICKS_EVENT_BASE/evt/$1/$2" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE       = "$OPTICKS_EVENT_BASE/evt/$1/$2/$3" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:       LOG(debug) << "BOpticksEvent::directory_template OVERRIDE_EVENT_BASE replacing OPTICKS_EVENT_BASE with " << OVERRIDE_EVENT_BASE ; 
    ./boostrap/BOpticksEvent.cc:       boost::replace_first(deftmpl, "$OPTICKS_EVENT_BASE/evt", OVERRIDE_EVENT_BASE );
    ./boostrap/BFile.cc:    else if(strcmp(key,"OPTICKS_EVENT_BASE")==0) 
    ./boostrap/BFile.cc:        LOG(verbose) << "replacing $OPTICKS_EVENT_BASE  with " << evalue ; 
    [blyth@localhost opticks]$ 


::

    159 std::string BFile::ResolveKey( const char* key )
    160 {
    161 
    162     const char* envvar = SSys::getenvvar(key) ;
    163     std::string evalue ;
    164 
    165     if( IsAllowedEnvvar(key) )
    166     {
    167         if( envvar != NULL )
    168         {
    169             evalue = envvar ;
    170             LOG(verbose) << "replacing allowed envvar token " << key << " with value of tenvvar " << evalue ;
    171         }
    172         else
    173         {
    174             evalue = usertmpdir("/tmp","opticks", NULL);
    175             LOG(error) << "replacing allowed envvar token " << key << " with default value " << evalue << " as envvar not defined " ;
    176         }
    177     }
    178     else if(strcmp(key,"KEYDIR")==0 )
    179     {
    180         const char* idpath = BResource::Get("idpath") ;
    181         assert( idpath );
    182         evalue = idpath ;
    183         LOG(error) << "replacing $IDPATH with " << evalue ;
    184     }
    185     else if(strcmp(key,"DATADIR")==0 )
    186     {
    187         const char* datadir = BResource::Get("opticksdata_dir") ;
    188         assert( datadir );
    189         evalue = datadir ;
    190         LOG(error) << "replacing $DATADIR with " << evalue ;
    191     }
    192     else if(strcmp(key,"OPTICKS_EVENT_BASE")==0)
    193     {
    194         const char* evtbase = BResource::Get("evtbase") ;
    195         if( evtbase != NULL )
    196         {
    197             evalue = evtbase ;
    198         }
    199         else
    200         {
    201             //evalue = BResource::Get("tmpuser_dir") ; 
    202             evalue = usertmpdir("/tmp","opticks",NULL);
    203         }
    204         LOG(verbose) << "replacing $OPTICKS_EVENT_BASE  with " << evalue ;
    205     }
    206     else
    207     {
    208         evalue = key ;
    209     }
    210     return evalue ;
    211 }
    212 
    213 
    214 
    215 





