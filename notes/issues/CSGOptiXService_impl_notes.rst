CSGOptiXService_impl_notes
============================

Overview
---------

* nanobind works to interface from FastAPI python thru to Opticks/CSGOptiX 
  (notice no Geant4 dependency in the service : its not needed as pure optical)






TODO : replace CSGOptiX API used from G4CXOpticks with an equivalent API that is using the remote CSGOptiX via libcurl calls
------------------------------------------------------------------------------------------------------------------------------

What does CSGOptiX do that needs to be done in the client ?
Can that functionality be moved to sysrap/CSGFoundry ?

* DONE: hide all QSim usage within G4CXOpticks behind CSGOptiX, motivation
  is to concentrate GPU usage into the one object making switching that out
  for the non-GPU client simpler


How does G4CXOpticks use CSGOptiX ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* G4CXOpticks::setGeometry_ invokes::

     cx = CSGOptiX::Create(fd)


BUT CSGOptiX::Create does some Init needed on client ? WHERE TO DRAW GPU/npGPU line?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* GPU SEvt setup done by CSGOptiX::InitEvt also needed on client::


     235 void CSGOptiX::InitEvt( CSGFoundry* fd  )
     236 {
     237     SEvt* sev = SEvt::CreateOrReuse(SEvt::EGPU) ;
     238 
     239     sev->setGeo((SGeo*)fd);
     240 


     241     std::string* rms = SEvt::RunMetaString() ;
     242     assert(rms);
     243 
     244     bool stamp = false ;
     245     smeta::Collect(*rms, "CSGOptiX__InitEvt", stamp );
     246 }




TODO : realistic client with local hit formation
-------------------------------------------------

Client needs to:

1. use libcurl to shoot off the gensteps and receive hits 

* HMM: maybe rejig G4CXOpticks to use same API for local and remote
  simulation with compile time switch (very different dependencies)

* [START DEVELOPMENT WITH RUNTIME SWITCH FOR CONVENIENCE OF TESTING, 
   UNTIL TEST ON NON-GPU MACHINE]

Eventually must use compile time switch, maybe detected by CMake based
on available packages but need runtime switch to begin with.

* monolithic "client" and "service" together (depends on CUDA/OptiX/CSGOptiX)
* "client" (depends on libcurl)
 
  * HMM: maybe arrange G4CXOpticks to work with protocol fulfilled by both:

    * CSGOptiX/CSGOptiX (monolithic)
    * sysrap/s_CURL_CSGOptiX.h (client)


HMM: the client just needs to receive the remotely obtained hits into its SEvt, 
thence can directly use existing machinery.  The only difference is where the
simulation is done.::


     67 /**
     68 U4HitGet::FromEvt
     69 ------------------
     70  
     71 HMM: this is awkward for Opticks-as-a-service as do not want to return both global and local ?
     72 Better to define a composite hit array serialization that can be formed on the server,
     73 to avoid doubling the size of hit array that needs to be returned.
     74  
     75 The Opticks-client will be receiving global hit(sphoton) 
     76 from which it needs to do something like SEvt::getLocalHit
     77 to apply the appropriate transforms to get the local hit.
     78  
     79 **/
     80  
     81 inline void U4HitGet::FromEvt(U4Hit& hit, unsigned idx, int eidx )
     82 {
     83     sphoton global ;
     84     sphoton local ;
     85     
     86     SEvt* sev = SEvt::Get(eidx);
     87     sev->getHit( global, idx);    // gets *idx* item from the hit array
     88     
     89     sphit ht ;  // extra hit info, just 3 ints : iindex, sensor_identifier, sensor_index
     90     sev->getLocalHit( ht, local,  idx);
     91     
     92     ConvertFromPhoton(hit, global, local, ht );
     93 }   



    4831 void SEvt::getLocalHit(sphit& ht, sphoton& lp, unsigned idx) const
    4832 {
    4833     getHit(lp, idx);   // copy *idx* hit from NP array (starts global frame) into sphoton& lp struct of caller
    4834     int iindex = lp.iindex() ;
    4835 
    4836     const glm::tmat4x4<double>* tr = tree ? tree->get_iinst(iindex) : nullptr ;
    4837 
    4838     LOG_IF(fatal, tr == nullptr)
    4839          << " FAILED TO GET INSTANCE TRANSFORM : WHEN TESTING NEEDS SSim::Load NOT SSim::Create"
    4840          << " iindex " << iindex
    4841          << " tree " << ( tree ? "YES" : "NO " )
    4842          << " tree.desc_inst " << ( tree ? tree->desc_inst() : "-" )
    4843          ;
    4844     assert( tr );
    4845 
    4846     bool normalize = true ;
    4847     lp.transform( *tr, normalize );   // inplace transforms lp (pos, mom, pol) into local frame
    4848 
    4849     glm::tvec4<int64_t> col3 = {} ;
    4850     strid::Decode( *tr, col3 );
    4851 
    4852     ht.iindex = col3[0] ;
    4853     ht.sensor_identifier = col3[2] ;  // NB : NO "-1" HERE : SEE ABOVE COMMENT
    4854     ht.sensor_index = col3[3] ;
    4855 }




DONE : test clients using curl commandline or C++ libcurl to use remote API
-----------------------------------------------------------------------------

* ~/np/tests/np_curl_test/np_curl_test.sh


DONE : High Level simulate API to receive gensteps from python and return hits
-------------------------------------------------------------------------------

CSGOptiX/CSGOptiXService.h::

     76 inline NP* CSGOptiXService::simulate( NP* gs, int eventID )
     77 {
     78     if(level > 0) std::cout << "[CSGOptiXService::simulate gs " << ( gs ? gs->sstr() : "-" ) << "\n" ;
     79  
     80     NP* ht = cx->simulate(gs, eventID );
     81  
     82     if(level > 0) std::cout << "]CSGOptiXService::simulate ht " << ( ht ? ht->sstr() : "-" ) << "\n" ;
     83     return ht ;
     84 }


DONE : Python package nanobind bound to CSGOptiX : using a "shadow" _CSGOptiXService to take care of the nanobind interfacing
-------------------------------------------------------------------------------------------------------------------------------

CSGOptiX/opticks_CSGOptiX.cc::

     14 #include "CSGOptiXService.h"
     15  
     16 namespace nb = nanobind;
     17  
     18  
     19 struct _CSGOptiXService
     20 {
     21    int             level ;
     22    CSGOptiXService svc ;
     23  
     24    _CSGOptiXService();
     25    virtual ~_CSGOptiXService();
     26  
     27    nb::ndarray<nb::numpy> simulate( nb::ndarray<nb::numpy> _gs, int eventID ) ;
     28    nb::tuple    simulate_with_meta( nb::ndarray<nb::numpy> _gs, nb::str _gs_meta, int eventID ) ;
     29  
     30    std::string desc() const ;
     31 };
     ..
     96 // First argument is module name which must match the first arg to nanobind_add_module in CMakeLists.txt
     97 NB_MODULE(opticks_CSGOptiX, m)
     98 {
     99     m.doc() = "nanobind _CSGOptiXService ";
    100  
    101     nb::class_<_CSGOptiXService>(m, "_CSGOptiXService")
    102         .def(nb::init<>())
    103         .def("__repr__", &_CSGOptiXService::desc)
    104         .def("simulate", &_CSGOptiXService::simulate )
    105         .def("simulate_with_meta", &_CSGOptiXService::simulate_with_meta )
    106         ;
    107 }




CSGOptiX/tests/CSGOptiXService_FastAPI_test/CSGOptiXService_FastAPI_test.sh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


CSGOptiX/tests/CSGOptiXService_FastAPI_test/main.py : FastAPI endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




