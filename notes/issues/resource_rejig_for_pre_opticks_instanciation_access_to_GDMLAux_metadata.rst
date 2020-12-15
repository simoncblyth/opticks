resource_rejig_for_pre_opticks_instanciation_access_to_GDMLAux_metadata
=========================================================================

resource rejig steps
---------------------

1. DONE : make m_rsc BOpticksResource a constituent of OpticksResource (not base class) for flexibility 
2. DONE : lower level metadata handling in BMeta  
3. DONE : add nljson dependency to brap as a REQUIRED, adding nljson- external  
4. DONE : migrate all NMeta usage to BMeta 
5. make BOpticksResource always auto-boot using setupViaKey  
5. make npy dependency on YoctoGL optional for GLTF handling only 
6. clean up the OLD_RESOURCE blocks







