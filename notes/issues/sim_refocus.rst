sim_refocus
=============

propagation : bring the bounce loop over from OptiXRap/cu/generate.cu to OptiX7Test.cu 
-----------------------------------------------------------------------------------------

* split off qgator ? or do in qsim ?    

* need bounce_max in param 

* most involved aspect is material prop access which needs an updated qstate.h 
  and fill_state equivalent to populate it via boundary lookups 
 
* actually moving the photon param is straightforward and can be done in mostly 
  static qsim methods adapted from OptiXRap/cu/propagate.h simply removing OptiX-isms 
  and adapting to new qstate

* for ease of development one of the most useful things is to 
  provide a route for CPU only testing 

  * currently qsim.h methods are GPU only 
  * needs mockup CPU equivalents for tex2D and  curandState
 
* for diving back into QUDARap the tests are the place to start





optixrap/cu/state.h:fill_state
---------------------------------

optical_buffer
~~~~~~~~~~~~~~~~


* boundary_lookup already brought over  bnd.npy being passed to CSGFoundry in CSG_GGeo_Convert::convertBndLib

::

     231 NP* GPropertyLib::getBuf() const
     232 {
     233     NP* buf = m_buffer ? m_buffer->spawn() : nullptr ;
     234     const std::vector<std::string>& names = getNameList();
     235     if(buf && names.size() > 0)
     236     {
     237         buf->set_meta(names);
     238     }
     239     return buf ;
     240 }


* need optical_buffer too like 

  * OBndLib::convert/OBndLib::makeBoundaryOptical 

* hmm just needs a GPU side buffer, no texture needed  




bnd domain range rejig
~~~~~~~~~~~~~~~~~~~~~~~~

bnd lookups failing for lack of domain metadata

Suspect the domain metadata is getting stomped on by boundary names::

    epsilon:tests blyth$ opticks-f domain_low
    ./ggeo/GBndLib.cc:    float domain_low = dom.x ; 
    ./ggeo/GBndLib.cc:    wav->setMeta("domain_low",   domain_low ); 
    ./qudarap/QBnd.cc:    domainX.f.x = dsrc->getMeta<float>("domain_low", "0" ); 
    ./qudarap/QBnd.cc:        << " domain_low " << std::fixed << std::setw(10) << std::setprecision(3) << domainX.f.x  
    epsilon:opticks blyth$ 

Avoid the stomping by adding set_names/get_names to NP





