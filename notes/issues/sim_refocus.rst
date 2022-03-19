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



hmm need to hand over metadata between NPY and NP : NPYSpawnNPTest.cc to check addition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    1072     NPY<double>* wav = NPY<double>::make( ni, nj, nk, nl, nm) ;
    1073     wav->fill( GSurfaceLib::SURFACE_UNSET );
    1074 
    1075     float domain_low = dom.x ;
    1076     float domain_high = dom.y ;
    1077     float domain_step = dom.z ;
    1078     float domain_range = dom.w ;
    1079 
    1080     wav->setMeta("domain_low",   domain_low );
    1081     wav->setMeta("domain_high",  domain_high );
    1082     wav->setMeta("domain_step",  domain_step );
    1083     wav->setMeta("domain_range", domain_range );
    1084 
    1085 




Avoid the stomping by adding set_names/get_names to NP



Check with cosTheta=0.5f::

fill_state check optical
---------------------------

Check with cosTheta=-0.5f::

    In [28]: np.c_[np.arange(44), t.state[:,4].view(np.uint32), t.state_names ]                                                                                                                             
    Out[28]: 
    array([[0, 0, 0, 0, 0, 'Galactic///Galactic'],
           [1, 0, 0, 0, 0, 'Galactic///Rock'],
           [2, 0, 0, 0, 0, 'Rock///Galactic'],
           [3, 0, 0, 0, 0, 'Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air'],
           [4, 0, 0, 0, 0, 'Rock///Rock'],
           [5, 0, 0, 0, 0, 'Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air'],
           [6, 0, 0, 0, 0, 'Air///Steel'],
           [7, 0, 0, 0, 0, 'Air///Air'],
           [8, 0, 0, 0, 0, 'Air///LS'],
           [9, 0, 0, 0, 0, 'Air///Tyvek'],
           [10, 0, 0, 0, 0, 'Air///Aluminium'],
           [11, 0, 0, 0, 0, 'Aluminium///Adhesive'],
           [12, 0, 0, 0, 0, 'Adhesive///TiO2Coating'],
           [13, 0, 0, 0, 0, 'TiO2Coating///Scintillator'],
           [14, 0, 0, 0, 0, 'Rock///Tyvek'],
           [15, 0, 0, 0, 0, 'Tyvek//Implicit_RINDEX_NoRINDEX_pOuterWaterPool_pPoolLining/vetoWater'],
           [16, 0, 0, 0, 0, 'vetoWater///LatticedShellSteel'],
           [17, 16, 0, 3, 20, 'vetoWater/CDTyvekSurface//Tyvek'],
           [18, 0, 0, 0, 0, 'Tyvek//CDInnerTyvekSurface/Water'],
           [19, 0, 0, 0, 0, 'Water///Acrylic'],
           [20, 0, 0, 0, 0, 'Acrylic///LS'],
           [21, 0, 0, 0, 0, 'LS///Acrylic'],
           [22, 0, 0, 0, 0, 'LS///PE_PA'],
           [23, 17, 0, 3, 20, 'Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/StrutSteel'],
           [24, 18, 0, 3, 20, 'Water/Strut2AcrylicOpSurface/Strut2AcrylicOpSurface/StrutSteel'],
           [25, 0, 0, 0, 0, 'Water///Steel'],
           [26, 0, 0, 0, 0, 'Water///PE_PA'],
           [27, 0, 0, 0, 0, 'Water///Water'],
           [28, 0, 0, 0, 0, 'Water///AcrylicMask'],
           [29, 19, 0, 3, 20, 'Water/NNVTMaskOpticalSurface/NNVTMaskOpticalSurface/CDReflectorSteel'],
           [30, 0, 0, 0, 0, 'Water///Pyrex'],
           [31, 0, 0, 0, 0, 'Pyrex///Pyrex'],
           [32, 4, 0, 0, 100, 'Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum'],
           [33, 0, 0, 0, 0, 'Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum'],
           [34, 20, 0, 3, 20, 'Water/HamamatsuMaskOpticalSurface/HamamatsuMaskOpticalSurface/CDReflectorSteel'],
           [35, 7, 0, 0, 100, 'Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum'],
           [36, 0, 0, 0, 0, 'Pyrex//HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum'],
           [37, 10, 0, 0, 100, 'Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum'],
           [38, 0, 0, 0, 0, 'Pyrex//PMT_3inch_absorb_logsurf1/Vacuum'],
           [39, 0, 0, 0, 0, 'Water///LS'],
           [40, 21, 0, 3, 20, 'Water/Steel_surface/Steel_surface/Steel'],
           [41, 0, 0, 0, 0, 'vetoWater///Water'],
           [42, 15, 0, 0, 100, 'Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum'],
           [43, 0, 0, 0, 0, 'Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum']], dtype=object)


Check with cosTheta=0.5f::

    In [1]: np.c_[np.arange(44), t.state[:,4].view(np.uint32), t.state_names ]                                                                                                                              
    Out[1]: 
    array([[0, 0, 0, 0, 0, 'Galactic///Galactic'],
           [1, 0, 0, 0, 0, 'Galactic///Rock'],
           [2, 0, 0, 0, 0, 'Rock///Galactic'],
           [3, 22, 1, 1, 100, 'Rock//Implicit_RINDEX_NoRINDEX_pDomeAir_pDomeRock/Air'],
           [4, 0, 0, 0, 0, 'Rock///Rock'],
           [5, 23, 1, 1, 100, 'Rock//Implicit_RINDEX_NoRINDEX_pExpHall_pExpRockBox/Air'],
           [6, 0, 0, 0, 0, 'Air///Steel'],
           [7, 0, 0, 0, 0, 'Air///Air'],
           [8, 0, 0, 0, 0, 'Air///LS'],
           [9, 0, 0, 0, 0, 'Air///Tyvek'],
           [10, 0, 0, 0, 0, 'Air///Aluminium'],
           [11, 0, 0, 0, 0, 'Aluminium///Adhesive'],
           [12, 0, 0, 0, 0, 'Adhesive///TiO2Coating'],
           [13, 0, 0, 0, 0, 'TiO2Coating///Scintillator'],
           [14, 0, 0, 0, 0, 'Rock///Tyvek'],
           [15, 24, 1, 1, 100, 'Tyvek//Implicit_RINDEX_NoRINDEX_pOuterWaterPool_pPoolLining/vetoWater'],
           [16, 0, 0, 0, 0, 'vetoWater///LatticedShellSteel'],
           [17, 0, 0, 0, 0, 'vetoWater/CDTyvekSurface//Tyvek'],
           [18, 12, 0, 3, 20, 'Tyvek//CDInnerTyvekSurface/Water'],
           [19, 0, 0, 0, 0, 'Water///Acrylic'],
           [20, 0, 0, 0, 0, 'Acrylic///LS'],
           [21, 0, 0, 0, 0, 'LS///Acrylic'],
           [22, 0, 0, 0, 0, 'LS///PE_PA'],
           [23, 17, 0, 3, 20, 'Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/StrutSteel'],
           [24, 18, 0, 3, 20, 'Water/Strut2AcrylicOpSurface/Strut2AcrylicOpSurface/StrutSteel'],
           [25, 0, 0, 0, 0, 'Water///Steel'],
           [26, 0, 0, 0, 0, 'Water///PE_PA'],
           [27, 0, 0, 0, 0, 'Water///Water'],
           [28, 0, 0, 0, 0, 'Water///AcrylicMask'],
           [29, 19, 0, 3, 20, 'Water/NNVTMaskOpticalSurface/NNVTMaskOpticalSurface/CDReflectorSteel'],
           [30, 0, 0, 0, 0, 'Water///Pyrex'],
           [31, 0, 0, 0, 0, 'Pyrex///Pyrex'],
           [32, 2, 0, 0, 100, 'Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum'],
           [33, 3, 0, 1, 99, 'Pyrex//NNVTMCPPMT_PMT_20inch_mirror_logsurf1/Vacuum'],
           [34, 20, 0, 3, 20, 'Water/HamamatsuMaskOpticalSurface/HamamatsuMaskOpticalSurface/CDReflectorSteel'],
           [35, 5, 0, 0, 100, 'Pyrex/HamamatsuR12860_PMT_20inch_photocathode_logsurf2/HamamatsuR12860_PMT_20inch_photocathode_logsurf1/Vacuum'],
           [36, 6, 0, 1, 99, 'Pyrex//HamamatsuR12860_PMT_20inch_mirror_logsurf1/Vacuum'],
           [37, 8, 0, 0, 100, 'Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum'],
           [38, 9, 0, 0, 100, 'Pyrex//PMT_3inch_absorb_logsurf1/Vacuum'],
           [39, 0, 0, 0, 0, 'Water///LS'],
           [40, 21, 0, 3, 20, 'Water/Steel_surface/Steel_surface/Steel'],
           [41, 0, 0, 0, 0, 'vetoWater///Water'],
           [42, 13, 0, 0, 100, 'Pyrex/PMT_20inch_veto_photocathode_logsurf2/PMT_20inch_veto_photocathode_logsurf1/Vacuum'],
           [43, 14, 0, 1, 99, 'Pyrex//PMT_20inch_veto_mirror_logsurf1/Vacuum']], dtype=object)


