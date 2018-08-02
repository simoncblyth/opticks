
surface_metadata_mechanics_review
=====================================


Metadata Mechanics for surfaces
-----------------------------------

What/where are the mechanics of metadata composition from the indiv GPropertyMap
into collective GPropertyLib ?


Where is the metadata persisted ? Only IDPATH/GSurfaceLib/GPropertyLibMetadata.json
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:1 blyth$ pwd
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1
    epsilon:1 blyth$ find . -name GPropertyLibMetadata.json
    ./GSurfaceLib/GPropertyLibMetadata.json

::

    epsilon:GSurfaceLib blyth$ opticks-pretty GPropertyLibMetadata.json
    {
        "ADVertiCableTraySurface": {
            "index": 18,
            "name": "__dd__Geometry__PoolDetails__PoolSurfacesAll__ADVertiCableTraySurface",
            "shortname": "ADVertiCableTraySurface",
            "sslv": "__dd__Geometry__PoolDetails__lvInnVertiCableTray0xbf28e40",
            "type": "skinsurface"
        },
        ...
        "ESRAirSurfaceBot": {
            "bpv1": "__dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458",
            "bpv2": "__dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08",
            "index": 1,
            "name": "__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot",
            "shortname": "ESRAirSurfaceBot",
            "type": "bordersurface"
        },


GPropertyMap
--------------

::

    099    public:
    100       // from metadata
    101       std::string getBPV1() const ;
    102       std::string getBPV2() const ;
    103       std::string getSSLV() const ;


Huh the getSSLV is never used?::

    epsilon:issues blyth$ opticks-find getSSLV
    ./ggeo/GPropertyMap.cc:std::string GPropertyMap<T>::getSSLV() const 
    ./ggeo/GPropertyMap.hh:      std::string getSSLV() const ; 



all property libs must implement createMeta 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:ggeo blyth$ opticks-find createMeta 
    ./ggeo/GBndLib.cc:NMeta* GBndLib::createMeta()
    ./ggeo/GPropertyLib.cc:    NMeta* meta = createMeta();
    ./ggeo/GScintillatorLib.cc:NMeta* GScintillatorLib::createMeta()
    ./ggeo/GSurfaceLib.cc:NMeta* GSurfaceLib::createMeta()
    ./ggeo/GSourceLib.cc:NMeta*  GSourceLib::createMeta()
    ./ggeo/GMaterialLib.cc:NMeta* GMaterialLib::createMeta()
    ./ggeo/GSurfaceLib.hh:       NMeta*      createMeta();
    ./ggeo/GScintillatorLib.hh:       NMeta*      createMeta();
    ./ggeo/GBndLib.hh:       NMeta*      createMeta();
    ./ggeo/GSourceLib.hh:       NMeta*      createMeta();
    ./ggeo/GMaterialLib.hh:       NMeta*      createMeta();

    ./ggeo/GPropertyLib.hh:        virtual NMeta*      createMeta() = 0;  



at GPropertyLib::close the metadata is collected and NMeta property on the lib is set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    398 void GPropertyLib::close()
    399 {
    ...
    408     sort();
    ...
    412     // create methods from sub-class specializations
    413 
    414     GItemList* names = createNames();
    415     NPY<float>* buf = createBuffer() ;
    416     NMeta* meta = createMeta();
    ...
    425     setNames(names);
    426     setBuffer(buf);
    427     setMeta(meta);
    428     setClosed();
    ...
    431 }

::

     619 /**
     620 NMeta* GSurfaceLib::createMeta()
     621 ---------------------------------
     622 
     623 Compose collective metadata keyed by surface shortnames
     624 with all metadata for each surface.
     625 
     626 **/
     627 
     628 NMeta* GSurfaceLib::createMeta()
     629 {
     630     NMeta* libmeta = new NMeta ;
     631     unsigned int ni = getNumSurfaces();
     632     for(unsigned int i=0 ; i < ni ; i++)
     633     {
     634         GPropertyMap<float>* surf = getSurface(i) ;
     635         const char* key = surf->getShortName() ;
     636         NMeta* surfmeta = surf->getMeta();
     637         assert( surfmeta );
     638         libmeta->setObj(key, surfmeta );
     639     }   
     640     return libmeta ;
     641 }   


Library metadata survives GPropertyLib::saveToCache GPropertyLib::loadFromCache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    463 void GPropertyLib::saveToCache()
    464 {
    ...
    471     if(m_buffer)
    472     {
    473         std::string dir = getCacheDir();
    474         std::string name = getBufferName();
    475         m_buffer->save(dir.c_str(), name.c_str());
    479             m_meta->save(dir.c_str(),  METANAME );
    485         saveNames(NULL);
    ...
    491 }

::

    500 void GPropertyLib::loadFromCache()
    501 {
    502     LOG(trace) << "GPropertyLib::loadFromCache" ;
    503 
    504     std::string dir = getCacheDir();
    505     std::string name = getBufferName();
    506 
    513     NPY<float>* buf = NPY<float>::load(dir.c_str(), name.c_str());
    ...
    523     NMeta* meta = NMeta::Load(dir.c_str(), METANAME ) ;
    524     assert( meta && "probably the geocache is an old version : lacking metadata : recreate geocache with -G option " );
    ...
    527     setBuffer(buf);
    528     setMeta(meta) ;
    529 
    530     GItemList* names = GItemList::load(m_resource->getIdPath(), m_type);
    531     setNames(names);
    532 
    533     import();
    534 }


Lib metadata gets split out into surfmeta and held back in reconstituted GPropertyMap in m_surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     779 /**
     780 GSurfaceLib::importForTex2d
     781 ------------------------------
     782 
     783 1. surfmeta gets pulled out of the collective libmeta and 
     784    set into the reconstituted GPropertyMap
     785 
     786 * observe : GOpticalSurface not reconstructed ?
     787 
     788 **/
     789 
     790 
     791 void GSurfaceLib::importForTex2d()
     792 {
     793     unsigned int ni = m_buffer->getShape(0); // surfaces
     802     for(unsigned int i=0 ; i < ni ; i++)
     803     {
     804         const char* key = m_names->getKey(i);
     809         GOpticalSurface* os = NULL ;  // huh : not reconstructed ?
     811         NMeta* surfmeta = m_meta ? m_meta->getObj(key) : NULL  ;
     813         const char* surftype = surfmeta ? AssignSurfaceType(surfmeta) : NULL ;
     831         GPropertyMap<float>* surf = new GPropertyMap<float>(key,i, surftype, os, surfmeta );
     ...
     845         m_surfaces.push_back(surf);
     846     }
     847 }
     848 


sslv 
------

::

    epsilon:1 blyth$ opticks-find sslv
    ./assimprap/AssimpGGeo.cc:        const char* sslv  = getStringProperty(mat, g4dae_skinsurface_volume );
    ./assimprap/AssimpGGeo.cc:        if( sslv )
    ./assimprap/AssimpGGeo.cc:                      << " sslv " << sslv 
    ./assimprap/AssimpGGeo.cc:            gss->setSkinSurface(sslv);
    ./assimprap/AssimpGGeo.cc:                gss_raw->setSkinSurface(sslv);
    ./assimprap/AssimpGGeo.cc:        free((void*)sslv);
    ./assimprap/AssimpGGeo.cc:        const char* sslv = gg->getCathodeLV(i);
    ./assimprap/AssimpGGeo.cc:                  << " sslv " << sslv 
    ./assimprap/AssimpGGeo.cc:        std::string name = BStr::trimPointerSuffixPrefix(sslv, NULL );
    ./assimprap/AssimpGGeo.cc:        gss->setSkinSurface(sslv);
    ./assimprap/AssimpGGeo.cc:            gss_raw->setSkinSurface(sslv);

    ##  AssimpGGeo takes the parsed G4DAE/COLLADA properties and sets them 
    ##  into GSkinSurface objects which are collected within GGeo/GSurfaceLib

    ./ggeo/GSurfaceLib.cc:const char* GSurfaceLib::SSLV             = "sslv" ;
    ./ggeo/GSurfaceLib.cc:void GSurfaceLib::addSkinSurface(GPropertyMap<float>* surf, const char* sslv_, bool direct )
    ./ggeo/GSurfaceLib.cc:    std::string sslv = sslv_ ;
    ./ggeo/GSurfaceLib.cc:    surf->setMetaKV(SSLV, sslv );
    ./ggeo/GSurfaceLib.cc:void GSurfaceLib::relocateBasisSkinSurface(const char* name, const char* sslv)
    ./ggeo/GSurfaceLib.cc:    addSkinSurface( surf, sslv, direct ); 
    ./ggeo/GPropertyMap.cc:    std::string sslv = m_meta->get<std::string>( GSurfaceLib::SSLV ) ;  
    ./ggeo/GPropertyMap.cc:    assert( !sslv.empty() );
    ./ggeo/GPropertyMap.cc:    return sslv ; 
    ./ggeo/GSurfaceLib.hh:        void addSkinSurface(GPropertyMap<float>* surf, const char* sslv_, bool direct );
    ./ggeo/GSurfaceLib.hh:        void relocateBasisSkinSurface(const char* name, const char* sslv);
    epsilon:opticks blyth$ 



