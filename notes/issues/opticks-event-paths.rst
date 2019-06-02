opticks-event-paths
====================


4-argument form of NPY save::

    1591 void OpticksEvent::saveDomains()
    1592 {
    1593     updateDomainsBuffer();
    1594 
    1595     NPY<float>* fdom = getFDomain();
    1596     if(fdom) fdom->save(fdom_, m_typ,  m_tag, m_udet);
    1597 
    1598     NPY<int>* idom = getIDomain();
    1599     if(idom) idom->save(idom_, m_typ,  m_tag, m_udet);
    1600 }

* 1st argument is the stem of the .npy name eg "fdom", "idom", "so", "ox", "ph", "no", "rx"

::

     730 template <typename T>
     731 void NPY<T>::save(const char* tfmt, const char* source, const char* tag, const char* det)
     732 {
     733     //std::string path_ = NPYBase::path(det, source, tag, tfmt );
     734     std::string path_ = BOpticksEvent::path(det, source, tag, tfmt );
     735     save(path_.c_str());
     736 }


See BOpticksEventTest::

    [BOpticksEvent::directory_@80]  top tboolean-box sub torch tag 1 anno NULL base (directory_template) $OPTICKS_EVENT_BASE/evt/$1/$2/$3


The root depends on OPTICKS_EVENT_BASE::

    [blyth@localhost tests]$ opticks-f OPTICKS_EVENT_BASE
    ./ana/nload.py:#DEFAULT_BASE = "$OPTICKS_EVENT_BASE/evt"
    ./ana/nload.py:DEFAULT_BASE = "$OPTICKS_EVENT_BASE/$0/evt"
    ./ana/ncensus.py:    c = Census("$OPTICKS_EVENT_BASE/source/evt")
    ./ana/base.py:        #self.setdefault("OPTICKS_EVENT_BASE",      os.path.join(keydir, "source" ))
    ./ana/base.py:        self.setdefault("OPTICKS_EVENT_BASE",       keydir )
    ./ana/base.py:        self.setdefault("OPTICKS_EVENT_BASE",      os.path.expandvars("/tmp/$USER/opticks") )
    ./boostrap/tests/BFileTest.cc:    ss.push_back("$OPTICKS_EVENT_BASE/evt/dayabay/cerenkov/1") ; 
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE_NOTAG = "$OPTICKS_EVENT_BASE/evt/$1/$2" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE       = "$OPTICKS_EVENT_BASE/evt/$1/$2/$3" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"
    ./boostrap/BOpticksEvent.cc:       LOG(debug) << "BOpticksEvent::directory_template OVERRIDE_EVENT_BASE replacing OPTICKS_EVENT_BASE with " << OVERRIDE_EVENT_BASE ; 
    ./boostrap/BOpticksEvent.cc:       boost::replace_first(deftmpl, "$OPTICKS_EVENT_BASE/evt", OVERRIDE_EVENT_BASE );
    ./boostrap/BFile.cc:    else if(strcmp(key,"OPTICKS_EVENT_BASE")==0) 
    ./boostrap/BFile.cc:        LOG(verbose) << "replacing $OPTICKS_EVENT_BASE  with " << evalue ; 
    [blyth@localhost opticks]$ 



OPTICKS_EVENT_BASE is not an allowed envvar, so it defaults to $TMP unless evtbase was set::

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
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
    203         }
    204         LOG(verbose) << "replacing $OPTICKS_EVENT_BASE  with " << evalue ;
    205     }
    206     else
    207     {
    208         evalue = key ;
    209     } 
    210     return evalue ;
    211 }



evtbase::

    [blyth@localhost opticks]$ opticks-f evtbase
    ./boostrap/BOpticksEvent.cc:srcevtbase
    ./boostrap/BOpticksEvent.cc:    const char* srcevtbase = BResource::Get("srcevtbase");   
    ./boostrap/BOpticksEvent.cc:    if( srcevtbase == NULL ) srcevtbase = BResource::Get("tmpuser_dir") ;   
    ./boostrap/BOpticksEvent.cc:    assert( srcevtbase ); 
    ./boostrap/BOpticksEvent.cc:    std::string path = BFile::FormPath(srcevtbase, "evt", det, typ, tag ); 
    ./boostrap/BOpticksResource.cc:    m_srcevtbase(NULL),
    ./boostrap/BOpticksResource.cc:    m_evtbase(NULL),
    ./boostrap/BOpticksResource.cc:    m_srcevtbase = makeIdPathPath("source"); 
    ./boostrap/BOpticksResource.cc:    m_res->addDir( "srcevtbase", m_srcevtbase ); 
    ./boostrap/BOpticksResource.cc:    //m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath("tmp", user, exename ) ;  
    ./boostrap/BOpticksResource.cc:    m_evtbase = isKeySource() ? strdup(m_srcevtbase) : makeIdPathPath(exename ) ;  
    ./boostrap/BOpticksResource.cc:    m_res->addDir( "evtbase", m_evtbase ); 
    ./boostrap/BOpticksResource.cc:  it writes its event and genstep into a distinctive "standard" directory (resource "srcevtbase") 
    ./boostrap/BOpticksResource.cc:  a relpath named after the executable (resource "evtbase")   
    ./boostrap/BOpticksResource.cc:srcevtbase 
    ./boostrap/BOpticksResource.cc:evtbase
    ./boostrap/BOpticksResource.cc://const char* BOpticksResource::getSrcEventBase() const { return m_srcevtbase ; } 
    ./boostrap/BOpticksResource.cc:const char* BOpticksResource::getEventBase() const { return m_evtbase ; } 
    ./boostrap/BOpticksResource.hh:        const char* m_srcevtbase ; 
    ./boostrap/BOpticksResource.hh:        const char* m_evtbase ; 
    ./boostrap/BFile.cc:        const char* evtbase = BResource::Get("evtbase") ; 
    ./boostrap/BFile.cc:        if( evtbase != NULL )
    ./boostrap/BFile.cc:            evalue = evtbase ; 
    [blyth@localhost opticks]$ 




Missing::

    /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMaterialLib/GPropertyLibMetadata.json


    [blyth@localhost 1]$ cat GMaterialLib/GPropertyLibMetadata.json
    {"abbrev":{"Air":"Ai","Glass":"Gl","Water":"Wa"}}

Created it with GMaterialLibTest:create_Meta


