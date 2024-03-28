Geant4_UseGivenVelocity_after_refraction_is_there_a_better_way_than_the_kludge_fix
===================================================================================


Related
---------

* :doc:`Geant4_using_GROUPVEL_from_wrong_initial_material_after_refraction.rst`
* :doc:`geant4_opticks_integration/GROUPVEL.rst`

* https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/Geant4_using_GROUPVEL_from_wrong_initial_material_after_refraction.rst
* https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/geant4_opticks_integration/GROUPVEL.rst


U4 Kludge fix is
------------------


::

    inline bool U4Track::IsOptical(const G4Track* track)
    {
        G4ParticleDefinition* particle = track->GetDefinition(); 
        return particle == G4OpticalPhoton::OpticalPhotonDefinition() ; 
    }




    bool is_optical = track->GetDefinition() == G4OpticalPhoton::OpticalPhotonDefinition() ; 
    if(is_optical) const_cast<G4Track*>(track)->UseGivenVelocity(true) ;    



   



::

     330 void U4Recorder::PreUserTrackingAction(const G4Track* track){  LOG(LEVEL) ; if(U4Track::IsOptical(track)) PreUserTrackingAction_Optical(track); }

     375 void U4Recorder::PreUserTrackingAction_Optical(const G4Track* track)
     376 {
     377     LOG(LEVEL) << "[" ;
     378 
     379     G4Track* _track = const_cast<G4Track*>(track) ;
     380     _track->UseGivenVelocity(true); // notes/issues/Geant4_using_GROUPVEL_from_wrong_initial_material_after_refraction.rst
     381 


This is needed to pickup the change at tail of [G4/C4]OpBoundaryProcess::PostStepDoIt::

     542         if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
     543            G4MaterialPropertyVector* groupvel =
     544            Material2->GetMaterialPropertiesTable()->GetProperty(kGROUPVEL);
     545            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
     546            aParticleChange.ProposeVelocity(finalVelocity);
     547         }
     548 
     549         if ( theStatus == Detection && fInvokeSD ) InvokeSD(pStep);
     550 
     551         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     552 }




G4Track ctor
-------------

::

    057 G4Track::G4Track(G4DynamicParticle* apValueDynamicParticle,
     58                  G4double aValueTime,
     59                  const G4ThreeVector& aValuePosition)
     60 ///////////////////////////////////////////////////////////
     61   : fCurrentStepNumber(0),    fPosition(aValuePosition),
     62     fGlobalTime(aValueTime),  fLocalTime(0.),
     63     fTrackLength(0.),
     64     fParentID(0),             fTrackID(0),
     65     fVelocity(c_light),
     66     fpDynamicParticle(apValueDynamicParticle),
     67     fTrackStatus(fAlive),
     68     fBelowThreshold(false),   fGoodForTracking(false),
     69     fStepLength(0.0),         fWeight(1.0),
     70     fpStep(0),
     71     fVtxKineticEnergy(0.0),
     72     fpLVAtVertex(0),          fpCreatorProcess(0),
     73     fCreatorModelIndex(-1),
     74     fpUserInformation(0),
     75     prev_mat(0),  groupvel(0),
     76     prev_velocity(0.0), prev_momentum(0.0),
     77     is_OpticalPhoton(false),
     78     useGivenVelocity(false),
     79     fpAuxiliaryTrackInformationMap(0)
     80 {  
     81   static G4ThreadLocal G4bool isFirstTime = true;
     82   static G4ThreadLocal G4ParticleDefinition* fOpticalPhoton =0;
     83   if ( isFirstTime ) {      
     84     isFirstTime = false;
     85     // set  fOpticalPhoton
     86     fOpticalPhoton = G4ParticleTable::GetParticleTable()->FindParticle("opticalphoton");
     87   }
     88   // check if the particle type is Optical Photon
     89   is_OpticalPhoton = (fpDynamicParticle->GetDefinition() == fOpticalPhoton);
     90 
     91   if (velTable ==0 ) velTable = G4VelocityTable::GetVelocityTable();
     92    
     93   fVelocity = CalculateVelocity();
     94 
     95 }  



    222 G4double G4Track::CalculateVelocity() const
    223 ///////////////////
    224 {
    225   if (useGivenVelocity) return fVelocity;
    226 
    227   G4double velocity = c_light ;
    228 
    229   G4double mass = fpDynamicParticle->GetMass();
    230 
    231   // special case for photons
    232   if ( is_OpticalPhoton ) return CalculateVelocityForOpticalPhoton();
    233 
    234   // particles other than optical photon
    235   if (mass<DBL_MIN) {
    236     // Zero Mass
    237     velocity = c_light;
    238   } else {
    239     G4double T = (fpDynamicParticle->GetKineticEnergy())/mass;
    240     if (T > GetMaxTOfVelocityTable()) {
    241       velocity = c_light;
    242     } else if (T<DBL_MIN) {
    243       velocity =0.;
    244     } else if (T<GetMinTOfVelocityTable()) {
    245       velocity = c_light*std::sqrt(T*(T+2.))/(T+1.0);
    246     } else {
    247       velocity = velTable->Value(T);
    248     }
    249    
    250   }
    251   return velocity ;
    252 }


Hmm so even without the touchable can access material via fpStep::

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




What about calling UseGivenVelocity at generation in the modified G4Scintillation and G4Cerenkov ?
-----------------------------------------------------------------------------------------------------

That might work, G4Track::CalculateVelocityForOpticalPhoton is public but it depends on material GROUPVEL 
being present. But that should be present for material with RINDEX::

    247 G4MaterialPropertyVector* G4MaterialPropertiesTable::AddProperty(
    248                                             const char *key,
    249                                             G4double   *PhotonEnergies,
    250                                             G4double   *PropertyValues,
    251                                             G4int      NumEntries)
    252 {
    ...
    268   // if key is RINDEX, we calculate GROUPVEL - 
    269   // contribution from Tao Lin (IHEP, the JUNO experiment) 
    270   if (k=="RINDEX") {
    271       CalculateGROUPVEL();
    272   }
    273 
    274   return mpv;
    275 }
     

    
::

    365 
    366       G4ThreeVector aSecondaryPosition = x0 + rand * aStep.GetDeltaPosition();
    367 
    368       G4Track* aSecondaryTrack =
    369                new G4Track(aCerenkovPhoton,aSecondaryTime,aSecondaryPosition);

    /// HMM the ctor runs without the touchable, so that means no material, no groupvel ? 
    

    370 
    371       aSecondaryTrack->SetTouchableHandle(
    372                                aStep.GetPreStepPoint()->GetTouchableHandle());
    373 
    374       aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    375 

Following source/processes/solidstate/phonon/src/G4VPhononProcess.cc could try::

    G4double velocity = aSecondaryTrack->CalculateVelocityForOpticalPhoton() ; 
    aSecondaryTrack->SetVelocity( velocity ); 
    aSecondaryTrack->UseGivenVelocity(true) ; 


TODO : Check the velocity of secondary track optical photon just after construction
------------------------------------------------------------------------------------



Plot thickens
----------------

* :google:`geant4 forum UseGivenVelocity`

* https://geant4-forum.web.cern.ch/t/optical-photons-wrong-velocity-after-a-reflection/6303
* https://bugzilla-geant4.kek.jp/show_bug.cgi?id=2438

* https://geant4.kek.jp/lxr/source/processes/optical/src/G4OpBoundaryProcess.cc#L189




Check velocity in rainbow test
--------------------------------

::

    ~/o/g4cx/tests/G4CXTest_raindrop_CPU.sh


    w = b.q_startswith("TO BT BT SA")   

First step is on x axis::

    r = e.f.record[sel_p,sel_r]   

    In [29]: r.shape
    Out[29]: (100000, 4, 4, 4)

    In [21]: r[:,1,0,:3] - r[:,0,0,:3]
    Out[21]: 
    array([[46.382,  0.   ,  0.   ],
           [30.038,  0.   ,  0.   ],
           [61.019,  0.   ,  0.   ],
           [35.257,  0.   ,  0.   ],
           [42.761,  0.   ,  0.   ],
           ...,
           [34.53 ,  0.   ,  0.   ],
           [70.81 ,  0.   ,  0.   ],
           [41.801,  0.   ,  0.   ],
           [33.255,  0.   ,  0.   ],
           [38.751,  0.   ,  0.   ]], dtype=float32)


Get the distance in 3D way::

    In [27]: np.sqrt(np.sum( (r[:,1,0,:3] - r[:,0,0,:3])*(r[:,1,0,:3] - r[:,0,0,:3]), axis=1 ))
    Out[27]: array([46.382, 30.038, 61.019, 35.257, 42.761, ..., 34.53 , 70.81 , 41.801, 33.255, 38.751], dtype=float32)

    In [28]: np.sqrt(np.sum( (r[:,1,0,:3] - r[:,0,0,:3])*(r[:,1,0,:3] - r[:,0,0,:3]), axis=1 )) / (r[:,1,0,3] - r[:,0,0,3])
    Out[28]: array([299.792, 299.792, 299.792, 299.792, 299.792, ..., 299.792, 299.792, 299.792, 299.792, 299.792], dtype=float32)


    In [30]: speed_ = lambda r,i:np.sqrt(np.sum( (r[:,i+1,0,:3]-r[:,i,0,:3])*(r[:,i+1,0,:3]-r[:,i,0,:3]), axis=1))/(r[:,i+1,0,3]-r[:,i,0,3])
    In [31]: speed_(r,0)
    Out[31]: array([299.792, 299.792, 299.792, 299.792, 299.792, ..., 299.792, 299.792, 299.792, 299.792, 299.792], dtype=float32)

    In [32]: speed_(r,1)
    Out[32]: array([224.901, 224.901, 224.901, 224.901, 224.901, ..., 224.901, 224.901, 224.901, 224.901, 224.901], dtype=float32)

    In [33]: speed_(r,2)
    Out[33]: array([299.792, 299.792, 299.792, 299.792, 299.792, ..., 299.793, 299.792, 299.792, 299.792, 299.793], dtype=float32)


::

    PICK=A MODE=3 SELECT="TO BT BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BT : 299.792 299.792 
    speed min/max for : 1 -> 2 : BT -> BR : 224.901 224.901 
    speed min/max for : 2 -> 3 : BR -> BT : 224.900 224.901 
    speed min/max for : 3 -> 4 : BT -> SA : 299.792 299.793 
    _pos.shape (45166, 3) 

    PICK=B MODE=3 SELECT="TO BT BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BT : 299.792 299.792 
    speed min/max for : 1 -> 2 : BT -> BR : 224.901 224.901 
    speed min/max for : 2 -> 3 : BR -> BT : 224.901 224.901 
    speed min/max for : 3 -> 4 : BT -> SA : 299.792 299.793 
    _pos.shape (45, 3) 

    PICK=B MODE=3 SELECT="TO BT BR BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BT : 299.792 299.792 
    speed min/max for : 1 -> 2 : BT -> BR : 224.901 224.901 
    speed min/max for : 2 -> 3 : BR -> BR : 224.901 224.901 
    speed min/max for : 3 -> 4 : BR -> BT : 224.901 224.901 
    speed min/max for : 4 -> 5 : BT -> SA : 299.792 299.793 
    _pos.shape (3, 3) 


    PICK=A MODE=3 SELECT="TO BT BR BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BT : 299.792 299.792 
    speed min/max for : 1 -> 2 : BT -> BR : 224.901 224.901 
    speed min/max for : 2 -> 3 : BR -> BR : 224.900 224.901 
    speed min/max for : 3 -> 4 : BR -> BT : 224.900 224.901 
    speed min/max for : 4 -> 5 : BT -> SA : 299.792 299.793 
    _pos.shape (5476, 3) 

    PICK=A MODE=3 SELECT="TO BT BR BR BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BT : 299.792 299.792 
    speed min/max for : 1 -> 2 : BT -> BR : 224.901 224.901 
    speed min/max for : 2 -> 3 : BR -> BR : 224.900 224.901 
    speed min/max for : 3 -> 4 : BR -> BR : 224.900 224.901 
    speed min/max for : 4 -> 5 : BR -> BT : 224.900 224.901 
    speed min/max for : 5 -> 6 : BT -> SA : 299.792 299.793 
    _pos.shape (1360, 3) 

    PICK=A MODE=3 SELECT="TO BT BR BR BR BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BT : 299.792 299.792 
    speed min/max for : 1 -> 2 : BT -> BR : 224.901 224.901 
    speed min/max for : 2 -> 3 : BR -> BR : 224.900 224.901 
    speed min/max for : 3 -> 4 : BR -> BR : 224.900 224.901 
    speed min/max for : 4 -> 5 : BR -> BR : 224.900 224.901 
    speed min/max for : 5 -> 6 : BR -> BT : 224.900 224.901 
    speed min/max for : 6 -> 7 : BT -> SA : 299.792 299.793 
    _pos.shape (536, 3) 



* https://www.quora.com/Why-does-total-internal-reflection-happen-inside-a-raindrop-Why-not-refraction

* Seem raindrop reflections can never TIR due to the geometry... need to generate light from inside the drop


