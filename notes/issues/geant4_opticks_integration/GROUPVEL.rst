GROUPVEL
==========

Approach
-----------

This is a longstanding issue of proagation time mismatch between Opticks and G4

Longterm solution is to export GROUPVEL property together with 
RINDEX and others in the G4DAE export.  Not prepared to go there
yet though, so need a shortcut way to get this property into the
Opticks boundary texture.


See Also
---------

* ana/groupvel.py 
* ggeo/GProperty<T>::make_GROUPVEL


Review
--------

* http://www.hep.man.ac.uk/u/roger/PHYS10302/lecture15.pdf
* http://web.ift.uib.no/AMOS/PHYS261/opticsPDF/Examples_solutions_phys263.pdf

::
                
    .
          c          w  dn           c           
    vg = --- (  1 +  -- ---  )   ~  --- (  1 +   ?  )
          n          n  dw           n              


     d logn      dn   1  
     ------ =   ---  --- 
      dw         dw   n


     d logw      dw   1             dn/n       dn   w
     ------ =   ---  ---    ->     -----  =    ---  -
      dn         dn   w            d logw       dw   n


     c          dn / n 
    --- ( 1 +   ---    )
     n          d logw


     c          dn  
     -   +   c  ---
     n          dlogw




                c         
    vg =  ---------------        # angular freq proportional to E for light     
            n + E dn/dE

    G4 using this energy domain approach approximating the dispersion part E dn/dE as shown below

                c                  n1 - n0         n1 - n0               dn        dn    dE          
    vg =  -----------       ds = ------------  =  ------------     ~   ------  =  ---- ------- =  E dn/dE 
           nn +  ds               log(E1/E0)      log E1 - log E0      d(logE)     dE   dlogE        
  



Now get G4 warnings when run without groupvel option
-------------------------------------------------------

::

    634   accuracy = theVelocityChange/c_light - 1.0;
    635   if (accuracy > accuracyForWarning) {
    636     itsOKforVelocity = false;
    637     nError += 1;
    638     exitWithError = exitWithError ||  (accuracy > accuracyForException);
    639 #ifdef G4VERBOSE
    640     if (nError < maxError) {
    641       G4cout << "  G4ParticleChange::CheckIt    : ";
    642       G4cout << "the velocity is greater than c_light  !!" << G4endl;
    643       G4cout << "  Velocity:  " << theVelocityChange/c_light  <<G4endl;
    644       G4cout << aTrack.GetDefinition()->GetParticleName()
    645          << " E=" << aTrack.GetKineticEnergy()/MeV
    646          << " pos=" << aTrack.GetPosition().x()/m
    647          << ", " << aTrack.GetPosition().y()/m
    648          << ", " << aTrack.GetPosition().z()/m
    649          <<G4endl;
    650     }
    651 #endif
    652   }



    2016-11-10 17:03:42.091 INFO  [373895] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
      G4ParticleChange::CheckIt    : the velocity is greater than c_light  !!
      Velocity:  1.00069
    opticalphoton E=2.88335e-06 pos=1.18776, -0.130221, 2.74632
          -----------------------------------------------
            G4ParticleChange Information  
          -----------------------------------------------
            # of 2ndaries       :                    0
          -----------------------------------------------
            Energy Deposit (MeV):                    0
            Non-ionizing Energy Deposit (MeV):                    0
            Track Status        :                Alive
            True Path Length (mm) :                3e+03
            Stepping Control      :                    0
        First Step In the voulme  : 
        Last Step In the voulme  : 
            Mass (GeV)   :                    0
            Charge (eplus)   :                    0
            MagneticMoment   :                    0
                    :  =                    0*[e hbar]/[2 m]
            Position - x (mm)   :             1.19e+03
            Position - y (mm)   :                 -130
            Position - z (mm)   :             2.75e+03
            Time (ns)           :                 9.98
            Proper Time (ns)    :                    0
            Momentum Direct - x :                0.397
            Momentum Direct - y :              -0.0435
            Momentum Direct - z :                0.917
            Kinetic Energy (MeV):             2.88e-06
            Velocity  (/c):                    1
            Polarization - x    :                0.918
            Polarization - y    :               0.0188
            Polarization - z    :               -0.396
      G4ParticleChange::CheckIt    : the velocity is greater than c_light  !!
      Velocity:  1.00069
    opticalphoton E=2.88335e-06 pos=1.18776, -0.130221, 2.74632
          -----------------------------------------------

::

    254 ///////////////////
    255 G4double G4Track::CalculateVelocityForOpticalPhoton() const
    256 ///////////////////
    257 {
    258    
    259   G4double velocity = c_light ;
    260  
    261 
    262   G4Material* mat=0;
    263   G4bool update_groupvel = false;
    264   if ( fpStep !=0  ){
    265     mat= this->GetMaterial();         //   Fix for repeated volumes
    266   }else{
    267     if (fpTouchable!=0){
    268       mat=fpTouchable->GetVolume()->GetLogicalVolume()->GetMaterial();
    269     }
    270   }
    271   // check if previous step is in the same volume
    272     //  and get new GROUPVELOCITY table if necessary 
    273   if ((mat != 0) && ((mat != prev_mat)||(groupvel==0))) {
    274     groupvel = 0;
    275     if(mat->GetMaterialPropertiesTable() != 0)
    276       groupvel = mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    277     update_groupvel = true;
    278   }
    279   prev_mat = mat;
    280  
    281   if  (groupvel != 0 ) {
    282     // light velocity = c/(rindex+d(rindex)/d(log(E_phot)))
    283     // values stored in GROUPVEL material properties vector
    284     velocity =  prev_velocity;
    285    
    286     // check if momentum is same as in the previous step
    287     //  and calculate group velocity if necessary 
    288     G4double current_momentum = fpDynamicParticle->GetTotalMomentum();
    289     if( update_groupvel || (current_momentum != prev_momentum) ) {
    290       velocity =
    291     groupvel->Value(current_momentum);
    292       prev_velocity = velocity;
    293       prev_momentum = current_momentum;
    294     }
    295   }  
    296  
    297   return velocity ;
    298 }







Opticks GROUPVEL
------------------

::

    simon:cfg4 blyth$ opticks-find GROUPVEL 
    ./cfg4/CPropLib.cc: GROUPVEL kludge causing "generational" confusion
    ./cfg4/CPropLib.cc:             LOG(info) << "CPropLib::makeMaterialPropertiesTable applying GROUPVEL kludge" ; 
    ./cfg4/CPropLib.cc:             addProperty(mpt, "GROUPVEL", prop );
    ./cfg4/CPropLib.cc:    bool groupvel = strcmp(lkey, "GROUPVEL") == 0 ; 
    ./cfg4/CTraverser.cc:const char* CTraverser::GROUPVEL = "GROUPVEL" ; 
    ./cfg4/CTraverser.cc:    // First get of GROUPVEL property creates it 
    ./cfg4/CTraverser.cc:            G4MaterialPropertyVector* gv = mpt->GetProperty(GROUPVEL);  
    ./cfg4/tests/CInterpolationTest.cc:    const char* mkeys_1 = "GROUPVEL,,," ;
    ./ggeo/GGeoTestConfig.cc:const char* GGeoTestConfig::GROUPVEL_ = "groupvel"; 
    ./ggeo/GGeoTestConfig.cc:    else if(strcmp(k,GROUPVEL_)==0)   arg = GROUPVEL ; 
    ./ggeo/GGeoTestConfig.cc:        case GROUPVEL       : setGroupvel(s)       ;break;
    ./ggeo/GMaterialLib.cc:"group_velocity:GROUPVEL,"
    ./cfg4/CTraverser.hh:        static const char* GROUPVEL ; 
    ./ggeo/GGeoTestConfig.hh:                      GROUPVEL,
    ./ggeo/GGeoTestConfig.hh:       static const char* GROUPVEL_ ; 
    simon:opticks blyth$ 



G4 GROUPVEL
--------------

::

    simon:geant4_10_02_p01 blyth$ find source -name '*.*' -exec grep -H GROUPVEL {} \;
    source/materials/include/G4MaterialPropertiesTable.hh:// Updated:     2005-05-12 add SetGROUPVEL() by P. Gumplinger
    source/materials/include/G4MaterialPropertiesTable.hh:    G4MaterialPropertyVector* SetGROUPVEL();
    source/materials/include/G4MaterialPropertiesTable.icc:  //2- So we have a data race if two threads access the same element (GROUPVEL)
    source/materials/include/G4MaterialPropertiesTable.icc:  //   at the bottom of the code, one thread in SetGROUPVEL(), and the other here
    source/materials/include/G4MaterialPropertiesTable.icc:  //3- SetGROUPVEL() is protected by a mutex that ensures that only
    source/materials/include/G4MaterialPropertiesTable.icc:  //   the same problematic key (GROUPVEL) the mutex will be used.
    source/materials/include/G4MaterialPropertiesTable.icc:  //5- As soon as a thread acquires the mutex in SetGROUPVEL it checks again
    source/materials/include/G4MaterialPropertiesTable.icc:  //   if the map has GROUPVEL key, if so returns immediately.
    source/materials/include/G4MaterialPropertiesTable.icc:  //   group velocity only once even if two threads enter SetGROUPVEL together
    source/materials/include/G4MaterialPropertiesTable.icc:  if (G4String(key) == "GROUPVEL") return SetGROUPVEL();
    source/materials/src/G4MaterialPropertiesTable.cc:// Updated:     2005-05-12 add SetGROUPVEL(), courtesy of
    source/materials/src/G4MaterialPropertiesTable.cc:G4MaterialPropertyVector* G4MaterialPropertiesTable::SetGROUPVEL()
    source/materials/src/G4MaterialPropertiesTable.cc:  // check if "GROUPVEL" already exists
    source/materials/src/G4MaterialPropertiesTable.cc:  itr = MPT.find("GROUPVEL");
    source/materials/src/G4MaterialPropertiesTable.cc:  // add GROUPVEL vector
    source/materials/src/G4MaterialPropertiesTable.cc:  // fill GROUPVEL vector using RINDEX values
    source/materials/src/G4MaterialPropertiesTable.cc:    G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    source/materials/src/G4MaterialPropertiesTable.cc:      G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    source/materials/src/G4MaterialPropertiesTable.cc:        G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    source/materials/src/G4MaterialPropertiesTable.cc:  this->AddProperty( "GROUPVEL", groupvel );
    source/processes/optical/src/G4OpBoundaryProcess.cc:           Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    source/track/src/G4Track.cc:    //  and get new GROUPVELOCITY table if necessary 
    source/track/src/G4Track.cc:      groupvel = mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    source/track/src/G4Track.cc:    // values stored in GROUPVEL material properties vector
    simon:geant4_10_02_p01 blyth$ 




G4Track.cc::

    ///
    ///  GROUPVEL  material property lookup just like RINDEX
    ///            the peculiarity is that the property is 
    ///            derived from RINDEX at first access by special casing in GetProperty
    ///

    317    // cached values for CalculateVelocity  
    318    mutable G4Material*               prev_mat;
    319    mutable G4MaterialPropertyVector* groupvel;
    320    mutable G4double                  prev_velocity;
    321    mutable G4double                  prev_momentum;
    322 


    254 ///////////////////
    255 G4double G4Track::CalculateVelocityForOpticalPhoton() const
    256 ///////////////////
    257 {
    258 
    259   G4double velocity = c_light ;
    260 
    261 
    262   G4Material* mat=0;
    263   G4bool update_groupvel = false;
    264   if ( fpStep !=0  ){
    265     mat= this->GetMaterial();         //   Fix for repeated volumes
    266   }else{
    267     if (fpTouchable!=0){
    268       mat=fpTouchable->GetVolume()->GetLogicalVolume()->GetMaterial();
    269     }
    270   }
    271   // check if previous step is in the same volume
    272     //  and get new GROUPVELOCITY table if necessary 
    273   if ((mat != 0) && ((mat != prev_mat)||(groupvel==0))) {
    274     groupvel = 0;
    275     if(mat->GetMaterialPropertiesTable() != 0)
    276       groupvel = mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    277     update_groupvel = true;
    278   }
    279   prev_mat = mat;
    280 
    281   if  (groupvel != 0 ) {
    282     // light velocity = c/(rindex+d(rindex)/d(log(E_phot)))
    283     // values stored in GROUPVEL material properties vector
    284     velocity =  prev_velocity;
    285 
    286     // check if momentum is same as in the previous step
    287     //  and calculate group velocity if necessary 
    288     G4double current_momentum = fpDynamicParticle->GetTotalMomentum();
    289     if( update_groupvel || (current_momentum != prev_momentum) ) {
    290       velocity =
    291     groupvel->Value(current_momentum);
    292       prev_velocity = velocity;
    293       prev_momentum = current_momentum;
    294     }
    295   }
    296 
    297   return velocity ;
    298 }



/usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc::

     529 
     530         aParticleChange.ProposeMomentumDirection(NewMomentum);
     531         aParticleChange.ProposePolarization(NewPolarization);
     532 
     533         if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
     534            G4MaterialPropertyVector* groupvel =
     535            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
     536            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
     537            aParticleChange.ProposeVelocity(finalVelocity);
     538         }
     ///
     ///     such velocity setting not in DsG4OpBoundaryProcess
     ///
     539 
     540         if ( theStatus == Detection ) InvokeSD(pStep);
     541 
     542         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     543 }
     544 
     545 void G4OpBoundaryProcess::BoundaryProcessVerbose() const
     546 {




source/materials/include/G4MaterialPropertiesTable.icc::

    115 inline G4MaterialPropertyVector*
    116 G4MaterialPropertiesTable::GetProperty(const char *key)
    117 {
    118   // Returns a Material Property Vector corresponding to a key
    119 
    120   //Important Note for MT. adotti 17 Feb 2016
    121   //In previous implementation the following line was at the bottom of the
    122   //function causing a rare race-condition.
    123   //Moving this line here from the bottom solves the problem because:
    124   //1- Map is accessed only via operator[] (to insert) and find() (to search),
    125   //   and these are thread safe if done on separate elements.
    126   //   See notes on data-races at:
    127   //   http://www.cplusplus.com/reference/map/map/operator%5B%5D/
    128   //   http://www.cplusplus.com/reference/map/map/find/
    129   //2- So we have a data race if two threads access the same element (GROUPVEL)
    130   //   one in read and one in write mode. This was happening with the line
    131   //   at the bottom of the code, one thread in SetGROUPVEL(), and the other here
    132   //3- SetGROUPVEL() is protected by a mutex that ensures that only
    133   //   one thread at the time will execute its code
    134   //4- The if() statement guarantees that only if two threads are searching
    135   //   the same problematic key (GROUPVEL) the mutex will be used.
    136   //   Different keys do not lock (good for performances)
    137   //5- As soon as a thread acquires the mutex in SetGROUPVEL it checks again
    138   //   if the map has GROUPVEL key, if so returns immediately.
    139   //   This "double check" allows to execute the heavy code to calculate
    140   //   group velocity only once even if two threads enter SetGROUPVEL together
    141   if (G4String(key) == "GROUPVEL") return SetGROUPVEL();
    142 
    143   MPTiterator i;
    144   i = MPT.find(G4String(key));
    145   if ( i != MPT.end() ) return i->second;
    146   return NULL;
    147 }

    /// computing a GROUPVEL property vector at first access cause lots of hassle, 
    /// given that RINDEX is constant, should just up front compute GROUPVEL for 
    /// all materials before any event handling happens




::

    119 G4MaterialPropertyVector* G4MaterialPropertiesTable::SetGROUPVEL()
    120 {
    ...
    141   G4MaterialPropertyVector* groupvel = new G4MaterialPropertyVector();
    142 
    146   G4double E0 = rindex->Energy(0);
    147   G4double n0 = (*rindex)[0];
    154   
    160   G4double E1 = rindex->Energy(1);
    161   G4double n1 = (*rindex)[1];
    168 
    169   G4double vg;
    173   vg = c_light/(n0+(n1-n0)/std::log(E1/E0));
    174 
          //   before the loop
          //            E0 = Energy(0)   E1 = Energy(1)      Energy(0) n[0]
          //

    177   if((vg<0) || (vg>c_light/n0))  { vg = c_light/n0; }
    178 
    179   groupvel->InsertValues( E0, vg );
    180 
    184   for (size_t i = 2; i < rindex->GetVectorLength(); i++)
    185   {
    186        vg = c_light/( 0.5*(n0+n1)+(n1-n0)/std::log(E1/E0));

            /// 
            /// note the sleight of hand the same (n1-n0)/std::log(E1/E0) is used for 1st 2 values
            ///

    187 
    190        if((vg<0) || (vg>c_light/(0.5*(n0+n1))))  { vg = c_light/(0.5*(n0+n1)); }

              // at this point in the loop
              //
              // i = 2,    E0 = Energy(0) E1 = Energy(1)    (Energy(0)+Energy(1))/2   // 1st pass using pre-loop settings
              // i = 3,    E0 = Energy(1) E1 = Energy(2)    (Energy(1)+Energy(2))/2   // 2nd pass E0,n0,E1,n1 shunted   
              // i = 4,    E0 = Energy(2) E1 = Energy(3)    (Energy(2)+Energy(3))/2   // 3rd pass E0,n0,E1,n1 shunted   
              //  ....
              // i = N-1   E0 = Energy(N-3)  E1 = Energy(N-2)   (Energy(N-3)+Energy(N-2))/2  


    191        groupvel->InsertValues( 0.5*(E0+E1), vg );
    195        E0 = E1;
    196        n0 = n1;
    197        E1 = rindex->Energy(i);
    198        n1 = (*rindex)[i];
    205   }
    ///
    ///       after the loop 
    ///       "i = N"      E0 = Energy(N-2)   E1 = Energy(N-1)         Energy(N-1)
    ///
    ///     hmmm a difference of bins is needed, but in order not to loose a bin
    ///     a tricky manoever is used of using the 1st and last bin and 
    ///     the average of the body bins
    ///     which means the first bin is half width, and last is 1.5 width
    ///
    ///         0  +  1  +  2  +  3  +  4  +  5        <--- 6 original values
    ///         |    /     /     /     /      |
    ///         |   /     /     /     /       |
    ///         0  1     2     3     4        5        <--- still 6 
    ///
    ///  
    ///
    206 
    209   vg = c_light/(n1+(n1-n0)/std::log(E1/E0));
    213   if((vg<0) || (vg>c_light/n1))  { vg = c_light/n1; }
    214   groupvel->InsertValues( E1, vg );
    ... 
    220   
    221   this->AddProperty( "GROUPVEL", groupvel );
    222   
    223   return groupvel;
    224 }

    ///
    ///           Argh... my domain checking cannot to be working...
    ///           this is sticking values midway in energy 
    ///
    ///           Opticks material texture requires fixed domain raster... 
    ///           so either interpolate to get that or adjust the calc ???
    ///


::

   ml = np.load("GMaterialLib.npy")
   wl = np.linspace(60,820,39)
   ri = ml[0,0,:,0]

   c_light = 299.792

   w0 = wl[:-1]
   w1 = wl[1:]

   n0 = ri[:-1]
   n1 = ri[1:]

    In [41]: c_light/(n0 + (n1-n0)/np.log(w1/w0))    # douple flip for e to w, one for reciprocal, one for order ???
    Out[41]: 
    array([ 206.2411,  206.2411,  206.2411,  106.2719,  114.2525, -652.0324,  125.2658,  210.3417,  215.9234,  221.809 ,  228.0242,  234.5973,  207.5104,  209.0361,  210.5849,  212.1565,  213.7514,
            207.991 ,  206.1923,  205.4333,  205.883 ,  206.8385,  207.5627,  208.0809,  206.0739,  205.295 ,  205.4116,  205.5404,  205.7735,  206.0065,  206.2412,  205.3909,  204.2895,  204.3864,
            204.4841,  204.5806,  204.6679,  202.8225])









