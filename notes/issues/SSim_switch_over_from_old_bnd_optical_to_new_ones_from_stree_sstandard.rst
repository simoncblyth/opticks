SSim_switch_over_from_old_bnd_optical_to_new_ones_from_stree_sstandard
=========================================================================


Currently using nasty mix of old and new workflows
-----------------------------------------------------

::

    2526 void GGeo::convertSim() const
    2527 {
    2528     SSim* sim = SSim::Get();
    2529     LOG_IF(fatal, sim == nullptr) << "SSim should have been created by CSGFoundry::CSGFoundry " ;
    2530     assert(sim);
    2531 
    2532     convertSim_BndLib(sim);
    2533     convertSim_ScintillatorLib(sim);
    2534     convertSim_Prop(sim);
    2535     convertSim_MultiFilm(sim);
    2536 }


Just now added the OLD prefix to make clear whats happening::

    2548 void GGeo::convertSim_BndLib(SSim* sim) const
    2549 {
    2550     LOG(LEVEL) << "[" ;
    2551     GBndLib* blib = getBndLib();
    2552 
    2553     bool can_create = blib->canCreateBuffer() ;
    2554     NP* oldbnd = nullptr ;
    2555     NP* oldoptical = nullptr ;
    2556 
    2557     if( can_create )
    2558     {
    2559         blib->createDynamicBuffers();
    2560         // hmm perhaps this is done already on loading now ?
    2561         oldbnd = blib->getBuf();
    2562 
    2563         LOG(LEVEL) << " oldbnd.desc " << oldbnd->desc() ;
    2564 
    2565         oldoptical = blib->getOpticalBuf();
    2566 
    2567         const std::vector<std::string>& bndnames = blib->getNameList();
    2568         oldbnd->set_names( bndnames );
    2569 
    2570         LOG(LEVEL) << " bnd.set_names " << bndnames.size() ;
    2571 
    2572 
    2573         sim->add(SSim::OLDBND, oldbnd );
    2574         sim->add(SSim::OLDOPTICAL, oldoptical );
    2575 
    2576         // OLD WORKFLOW ADDITION TO CHECK NEW WORKFLOW 
    2577         GMaterialLib* mlib = getMaterialLib();
    2578         GSurfaceLib*  slib = getSurfaceLib();
    2579         NP* oldmat = mlib->getBuf();
    2580         NP* oldsur = slib->getBuf();
    2581         sim->add(SSim::OLDMAT, oldmat );




update stree persisting to use NPFold
------------------------------------------

::

    In [14]: getattr(st, "mtname.txt")
    Out[14]: array([], dtype=int32)

    In [15]: getattr(st, "mtname.txt_names")
    Out[15]: 
    array(['Air', 'Rock', 'Galactic', 'Steel', 'LS', 'Tyvek', 'Scintillator', 'TiO2Coating', 'Adhesive', 'Aluminium', 'LatticedShellSteel', 'Acrylic', 'PE_PA', 'StrutSteel', 'AcrylicMask',
           'CDReflectorSteel', 'Vacuum', 'Pyrex', 'Water', 'vetoWater'], dtype='<U18')






