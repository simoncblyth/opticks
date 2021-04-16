enabledmergedmesh-not-working-anymore
========================================

::

   OpSnapTest --target 304632 --eye -1,-1,-1  --rtx 1 --cvd 1 --enabledmergedmesh 1


Formerly this controlled which solids got included into GPU geometry, 
but seems to do nothing now.


::

    1237    char enabledmergedmesh[256];
    1238    snprintf(enabledmergedmesh,256, "(former restrictmesh) Comma delimited string giving list of mesh indices to convert into OptiX geometry eg \"0,2,5\". Or blank for all. Default %s ", m_enab     ledmergedmesh.c_str() );
    1239    m_desc.add_options()
    1240        ("enabledmergedmesh",  boost::program_options::value<std::string>(&m_enabledmergedmesh), enabledmergedmesh );
    1241 

    2019 template <class Listener>
    2020 const std::string& OpticksCfg<Listener>::getEnabledMergedMesh() const
    2021 {
    2022     return m_enabledmergedmesh ;
    2023 }

    epsilon:optickscore blyth$ opticks-f getEnabledMergedMesh 
    ./optickscore/OpticksCfg.cc:const std::string& OpticksCfg<Listener>::getEnabledMergedMesh() const 
    ./optickscore/OpticksCfg.hh:     const std::string& getEnabledMergedMesh() const ; 
    ./optickscore/OpticksDbg.cc:   const std::string& enabledmm = m_cfg->getEnabledMergedMesh() ;
    epsilon:opticks blyth$ 

    046 class OKCORE_API OpticksDbg
     47 {      
     ...
     68        bool isX4PolySkip(unsigned lvIdx) const ;  
     69        bool isCSGSkipLV(unsigned lvIdx) const ;   // --csgskiplv
     70        bool isEnabledMergedMesh(unsigned mm) const ;
     71     public:

    epsilon:optickscore blyth$ opticks-f isEnabledMergedMesh 
    ./optickscore/OpticksDbg.hh:       bool isEnabledMergedMesh(unsigned mm) const ;
    ./optickscore/Opticks.hh:       bool isEnabledMergedMesh(unsigned mm) const ;
    ./optickscore/OpticksDbg.cc:bool OpticksDbg::isEnabledMergedMesh(unsigned mm) const 
    ./optickscore/Opticks.cc:bool Opticks::isEnabledMergedMesh(unsigned mm) const 
    ./optickscore/Opticks.cc:   return m_dbg->isEnabledMergedMesh(mm);
    epsilon:opticks blyth$ 

    0747 bool Opticks::isEnabledMergedMesh(unsigned mm) const
     748 {
     749    return m_dbg->isEnabledMergedMesh(mm);
     750 }


Seems it is not consulted::

    0295 void OGeo::convert()
     296 {
     297     m_geolib->dump("OGeo::convert");
     298 
     299     unsigned int nmm = m_geolib->getNumMergedMesh();
     300 
     301     LOG(info) << "[ nmm " << nmm ;
     302 
     303     if(m_verbosity > 1) m_geolib->dump("OGeo::convert GGeoLib" );
     304 
     305     for(unsigned i=0 ; i < nmm ; i++)
     306     {
     307         convertMergedMesh(i);
     308     }   
     309     
     310     m_top->setAcceleration( makeAcceleration(m_top_accel, false) );
     311     
     312     if(m_verbosity > 0) dumpStats();
     313     
     314     LOG(info) << "] nmm " << nmm  ;
     315 }   


Put this control back in.

