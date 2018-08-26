strategy_for_Cerenkov_Scintillation_alignment
================================================

* for context see :doc:`OKG4Test_direct_two_executable_shakedown`

Realisation of need for strategy redirection came from comparing 
the simstreams (call locations of flat for ckm-- and ckm-okg4).  

Thoughts
-----------

Why was I trying to start from primaries ? Rather than gensteps ?

* because are trying to duplicate the G4 running too, and G4 cannot 
  start from gensteps 

  * yes G4 cannot start from Gensteps : but it can start from the photons that the gensteps yield 
 
    * **THIS IS THE CORE PROBLEM** are missing: CCerenkovSource (primary generator class)
      

  * original motivation is to try to align Cerenkov/Scintillation generation of photons  

  * need to find another approach, how to cleanroom isolate Cerenkov generation ?
    but Cerenkov needs the em energy loss to do anything  

    * how does the Cerenkov process interface to learn of the energy loss ? 
      Can I mock that up somehow  ? basically I need a clean environment 
      without a lot of physics calling flat  

* SO : need to focus first on matching the G4 runs, start by 
  collecting G4 hits into an OpticksEvent : not with full machinery 
  need a simple way, as want to do from 1st and 2nd executable  

  * just have a collector for hits too, hmm will need to do for 
    entire collection in reverse : so do at collection level ?
    then theres an intermediary hit class inbetween : better to 
    do in ProcessHits just like `CWriter::writePhoton(const G4StepPoint* point )`


Validating the C+S sources ?
-------------------------------

* simply compare the photons and hits from the natural full physics case
  with the cleanroom case booted from gensteps 
    

What is really needed ?
--------------------------

0. run full physics Geant4 : persist geometry and gensteps (Cerenkov and Scintillation) to geocache
1. run minimal physics Geant4 (cleanroom environment) : load geometry and gensteps 

   * generate some photons from them and propagate them
   * this is treating C+S processes as generators of photons, so they should not
     be included into the physics : just reuse the photon generation loop converting 
     the "secondaries" into "primaries"   

2. run Opticks : load geometry and gensteps 

   * generate some photons from them and propagate them 

3. compare 
   

* no hope of aligning 0 with anything other than another 0
* it may be possible to align 1 and 2 however 

* recall gensteps are the "stack" midway thru the Cerenkov and Scintillation processes 
  so getting a hacked version of these to generate photons from loaded gensteps should
  be straightforward 



Turning S+C into genstep consuming photon generators is entirely doable : but what then
------------------------------------------------------------------------------------------

Basically are extracting photon secondary formation from the C+S processes photon loops 
and turning them into generators (or Sources in Opticks lingo).

*  CCerenkovSource
*  CScintillationSource

Then this is something that can run with optical physics only : which 
is amenable to random alignment with Opticks which is doing the same thing
when running from gensteps. 


DONE : first cut CCerenkovGenerator + tests/CCerenkovGeneratorTest 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* standalone CCerenkovGeneratorTest executable

  * (requires loading full hub just to get at RINDEX for materials)
  * loads gensteps
  * generates and saves photons

* issues : with old gensteps had to force a materialIndex 

  * TODO: avoid by using new gensteps (from ckm)
  * TODO: wavelength units off by 1e6 : review unit conversions thru the chain


Opticks revolves around the gensteps watershed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    168 G4VParticleChange*
    169 G4Cerenkov::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    ...
    265   G4double MeanNumberOfPhotons1 =
    266                      GetAverageNumberOfPhotons(charge,beta1,aMaterial,Rindex);
    267   G4double MeanNumberOfPhotons2 =
    268                      GetAverageNumberOfPhotons(charge,beta2,aMaterial,Rindex);
    269 

    /////////  Gensteps persist the stack here   //////////  

    270   for (G4int i = 0; i < fNumPhotons; i++) {
    271 
    272       // Determine photon energy
    273 
    274       G4double rand;
    275       G4double sampledEnergy, sampledRI;
    276       G4double cosTheta, sin2Theta;
    277 
    278       // sample an energy
    279 
    280       do {
    281          rand = G4UniformRand();
    282          sampledEnergy = Pmin + rand * dp;
    283          sampledRI = Rindex->Value(sampledEnergy);


DONE : CAlignEngine : Multiple Random streams, with a cursor for each  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* also added simstream logging into the idPath 


Apply CAlignEngine to CerenkovMinimal+G4Opticks 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Ordering of generation and propagation is different in Geant4 and in 
Opticks.  Opticks generates and propagates each photon using a contiguous 
stream of random numbers, wheras Geant4 interleaves generation and propagation::

    genstep0 - generate
    genstep0 - propagate  
        preTrack()  : needs to find a record_id as switch streams to it  
        postTrack()
        preTrack()
        postTrack()
        ...
       
    genstep1 - generate   
    genstep1 - propagate
    ...

Because of this have to do some gymnastics to get the Geant4 photon
RNG consumption to stay on the correct aligned stream.

       
* DONE : Added photon count collection to CCollector, renamed CGenstepCollector, for simple high level totals 
  of the photons that came before : this enables generation alone to stay aligned : but propagation 
  must also be contolled 

The above should be sufficient for generation, but need some more fpr propagationL

1. switch to aligned stream of the TrackID at preTrack
2. switch to non-aligned at postTrack 

   * TO CHECK : DOES THAT COVER ALL THE RNG CONSUMPTION OF THE PROPAGATION ?
     iWITHOUT ANY EXTRAS 

   * although there are other particles in the full physics Geant4 being propagated
     they should not interfere ? if the optical bracketing is complete ?



Try keeping generation using the appropriate streams like this::

    296 #ifdef WITH_OPTICKS
    297     unsigned opticks_photon_offset = 0 ;
    298     {
    299         const G4ParticleDefinition* definition = aParticle->GetDefinition();
    300         G4ThreeVector deltaPosition = aStep.GetDeltaPosition();
    301         G4int materialIndex = aMaterial->GetIndex();
    302         LOG(verbose) << dp ;
    303 
    304         opticks_photon_offset = G4Opticks::GetOpticks()->getNumPhotons();
    305         // total photons from all gensteps collected before this one
    306         // within this OpticksEvent (potentially crossing multiple G4Event) 
    307 
    308         G4Opticks::GetOpticks()->collectCerenkovStep(
    309                0,                  // 0     id:zero means use cerenkov step count 
    310                aTrack.GetTrackID(),
    311                materialIndex,
    312                NumPhotons,
    ...
    339     }
    340 #endif
    341
    342     for (G4int i = 0; i < NumPhotons; i++) {
    343 
    344         // Determine photon energy
    345 #ifdef WITH_OPTICKS
    346         unsigned record_id = opticks_photon_offset+i ;
    347         G4Opticks::GetOpticks()->setAlignIndex(record_id);
    348 #endif
    349 
    ...        the generation   ....
    456 
    457 #ifdef WITH_OPTICKS
    458         aSecondaryTrack->SetTrackID( record_id ); 
    459         G4Opticks::GetOpticks()->setAlignIndex(-1);
    460 #endif
    461 
    462     
    463     }


I recall trying to use TrackID before and getting stomped upon by G4, so use CTrackInfo :
to try to keep propagation in the grooves::

     67 void Ctx::setTrackOptical(const G4Track* track)
     68 {
     69     const_cast<G4Track*>(track)->UseGivenVelocity(true);
     70     
     71 #ifdef WITH_OPTICKS 
     72     CTrackInfo* info=dynamic_cast<CTrackInfo*>(track->GetUserInformation());
     73     assert(info) ;
     74     _record_id = info->photon_record_id ;
     75     G4Opticks::GetOpticks()->setAlignIndex(_record_id);
     76 #endif
     77 }   
     78 
     79 void Ctx::postTrackOptical(const G4Track* track)
     80 {   
     81 #ifdef WITH_OPTICKS
     82     CTrackInfo* info=dynamic_cast<CTrackInfo*>(track->GetUserInformation());
     83     assert(info) ; 
     84     assert( _record_id == info->photon_record_id ) ;
     85     G4Opticks::GetOpticks()->setAlignIndex(-1);
     86 #endif
     87 }




m_ctx._record_id used in CRandomEngine::preTrack 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Back then had a fixed number of photons per event, so could form 
a _record_id from _event_id and _photon_id (which is same as _track_id when 
no reemission to worry about).

::

    239 void CG4Ctx::setTrackOptical() // invoked by CG4Ctx::setTrack
    240 {
    241     LOG(debug) << "CTrackingAction::setTrack setting UseGivenVelocity for optical " ;
    242 
    243     _track->UseGivenVelocity(true);
    244 
    245     // NB without this BoundaryProcess proposed velocity to get correct GROUPVEL for material after refraction 
    246     //    are trumpled by G4Track::CalculateVelocity 
    247 
    248     _primary_id = CTrack::PrimaryPhotonID(_track) ;    // layed down in trackinfo by custom Scintillation process
    249     _photon_id = _primary_id >= 0 ? _primary_id : _track_id ;
    250     _reemtrack = _primary_id >= 0 ? true        : false ;
    251 
    252      // retaining original photon_id from prior to reemission effects the continuation
    253     _record_id = _photons_per_g4event*_event_id + _photon_id ;
    254     _record_fraction = double(_record_id)/double(_record_max) ;
    255 


Now with a fixed number of gensteps per event, i need to record in ctx
gensteps together with their photon counts.


CTrackInfo ? might be handy for debug, but an expensive way 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


CRandomEngine::preTrack  
~~~~~~~~~~~~~~~~~~~~~~~~~

Tis troublesome to have to modify all optical processes



How to use the simstream
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

      16 12  CerenkovMinimal                     0x0000000100032dad G4::beamOn(int)                                                                                      + 45
      17 13  CerenkovMinimal                     0x0000000100032c57 G4::G4(int)                                                                                          + 1015
      18 14  CerenkovMinimal                     0x0000000100032ddb G4::G4(int)                                                                                          + 27
      19 15  CerenkovMinimal                     0x0000000100011886 main + 550
      20 16  libdyld.dylib                       0x00007fff7acac015 start + 1
      21 (0     :0   ) 0.740219   :      0x1000212e0     + 2784 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      22 (0     :1   ) 0.438451   :      0x1000213bd     + 3005 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      23 (0     :2   ) 0.517013   :      0x10002141c     + 3100 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      24 (0     :3   ) 0.156989   :      0x1000216e4     + 3812 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      25 (0     :4   ) 0.071368   :      0x100021754     + 3924 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      26 (1     :0   ) 0.920994   :      0x1000212e0     + 2784 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      27 (1     :1   ) 0.460364   :      0x1000213bd     + 3005 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      28 (1     :2   ) 0.333464   :      0x10002141c     + 3100 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      29 (1     :3   ) 0.372520   :      0x1000216e4     + 3812 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      30 (1     :4   ) 0.489602   :      0x100021754     + 3924 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      31 (2     :0   ) 0.039020   :      0x1000212e0     + 2784 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      32 (2     :1   ) 0.250215   :      0x1000213bd     + 3005 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      33 (2     :2   ) 0.184484   :      0x10002141c     + 3100 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      34 (2     :3   ) 0.962422   :      0x1000216e4     + 3812 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      35 (2     :4   ) 0.520555   :      0x100021754     + 3924 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      36 (3     :0   ) 0.968963   :      0x1000212e0     + 2784 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
      37 (3     :1   ) 0.494743   :      0x1000213bd     + 3005 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)


Either directly use the addresses to jump to the file and line in lldb, 
or use relative offsets togther with the first address for jumping around 
within one symbol::

    epsilon:~ blyth$ ckm-addr2line  0x1000212e0-2784+3005
    (lldb) target create "/usr/local/opticks/lib/CerenkovMinimal"
    Current executable set to '/usr/local/opticks/lib/CerenkovMinimal' (x86_64).
    (lldb) source list -a 0x1000212e0-2784+3005
    /usr/local/opticks/lib/CerenkovMinimal`L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&) + 2986 at /Users/blyth/opticks/examples/Geant4/CerenkovMinimal/L4Cerenkov.cc:364
       353 			G4double cosTheta, sin2Theta;
       354 			
       355 			// sample an energy
       356 	
       357 			do {
       358 				rand = G4UniformRand();	
       359 				sampledEnergy = Pmin + rand * dp; 
       360 				sampledRI = Rindex->Value(sampledEnergy);
       361 				cosTheta = BetaInverse / sampledRI;  
       362 	
       363 				sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
    -> 364 				rand = G4UniformRand();	
       365 	
       366 			  // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
       367 			} while (rand*maxSin2 > sin2Theta);
       368 	
       369 			// Generate random position of photon on cone surface 
    epsilon:~ blyth$ 




Hmm would be good to know how much for generation and propagation separately::

    In [17]: c = np.load("/usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/CAlignEngine.npy")

    In [18]: c
    Out[18]: array([15, 15,  9, ...,  0,  0,  0], dtype=int32)

    In [19]: c[:100]
    Out[19]: 
    array([ 15,  15,   9,  19,   9,   9,   9,   9,  15,   9,   9,   9,   9,   9,  88,   9,   9,  11,   9,  15,  15,   9,  15,   9,  17,  11,  15,   9,  13,   9,  11,   9,   9,   9,   9,   9,   9,   9,
             9,   9,  13,   9,  15,  19,  21,   9,   9,   9,   9,  17,  13,  11,   9,  32,   9,   9,   9,   9, 344,   9,   9,  11,  13,   9,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0], dtype=int32)

    In [20]: 



Can see by grepping the simstream::

    epsilon:1 blyth$ grep "(0" CAlignEngine.log
    (0     :0   ) 0.740219   :      0x1000212e0     + 2784 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    (0     :1   ) 0.438451   :      0x1000213bd     + 3005 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    (0     :2   ) 0.517013   :      0x10002141c     + 3100 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    (0     :3   ) 0.156989   :      0x1000216e4     + 3812 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    (0     :4   ) 0.071368   :      0x100021754     + 3924 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    (0     :5   ) 0.462508   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    (0     :6   ) 0.227643   :      0x103b9c62c     + 7276 G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)
    (0     :7   ) 0.329358   :      0x103ba3b87       + 39 G4OpBoundaryProcess::G4BooleanRand(double) const
    (0     :8   ) 0.144065   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    (0     :9   ) 0.187799   :      0x103b9c62c     + 7276 G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)
    (0     :10  ) 0.915383   :      0x103ba3b87       + 39 G4OpBoundaryProcess::G4BooleanRand(double) const
    (0     :11  ) 0.540125   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    (0     :12  ) 0.974661   :      0x103b9c62c     + 7276 G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)
    (0     :13  ) 0.547469   :      0x103ba3b87       + 39 G4OpBoundaryProcess::G4BooleanRand(double) const
    (0     :14  ) 0.653160   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    epsilon:1 blyth$ 


::

    epsilon:1 blyth$ grep "(14 " CAlignEngine.log
    (14    :0   ) 0.681419   :      0x1000212e0     + 2784 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    (14    :1   ) 0.051981   :      0x1000213bd     + 3005 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    (14    :2   ) 0.907418   :      0x10002141c     + 3100 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    (14    :3   ) 0.050762   :      0x1000216e4     + 3812 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    (14    :4   ) 0.455413   :      0x100021754     + 3924 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    (14    :5   ) 0.384523   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    (14    :6   ) 0.295749   :      0x103b9c62c     + 7276 G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)
    (14    :7   ) 0.775048   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    (14    :8   ) 0.466141   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    (14    :9   ) 0.568090   :      0x103b9c62c     + 7276 G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)
    (14    :10  ) 0.477616   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    (14    :11  ) 0.929151   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    (14    :12  ) 0.326689   :      0x103b9c62c     + 7276 G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)
    (14    :13  ) 0.421148   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    (14    :14  ) 0.967082   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    (14    :15  ) 0.047660   :      0x103b9c62c     + 7276 G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)
    (14    :16  ) 0.068004   :      0x103ba3b87       + 39 G4OpBoundaryProcess::G4BooleanRand(double) const
    (14    :17  ) 0.064567   :      0x103b9642a       + 42 G4VProcess::ResetNumberOfInteractionLengthLeft()
    (14    :18  ) 0.426426   :      0x103b9c62c     + 7276 G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)
    (14    :19  ) 0.554089   :      0x103ba3b87       + 39 G4OpBoundaryProcess::G4BooleanRand(double) const

::

    In [20]: a[14]
    Out[20]: 
    array([[0.6814, 0.052 , 0.9074, 0.0508, 0.4554, 0.3845, 0.2957, 0.775 , 0.4661, 0.5681, 0.4776, 0.9292, 0.3267, 0.4211, 0.9671, 0.0477],
           [0.068 , 0.0646, 0.4264, 0.5541, 0.3078, 0.6465, 0.8975, 0.0655, 0.3716, 0.6215, 0.0535, 0.6389, 0.7884, 0.39  , 0.2253, 0.6899],
           [0.66  , 0.6058, 0.9699, 0.2572, 0.7936, 0.9252, 0.4559, 0.026 , 0.5386, 0.6192, 0.4679, 0.5474, 0.4873, 0.7793, 0.7539, 0.2975],
           [0.8542, 0.7306, 0.9052, 0.0072, 0.1194, 0.5093, 0.9403, 0.3871, 0.5629, 0.6254, 0.1167, 0.1175, 0.7874, 0.9329, 0.4942, 0.3054],
           [0.4878, 0.7517, 0.947 , 0.6053, 0.1629, 0.078 , 0.4845, 0.8413, 0.6961, 0.7894, 0.3104, 0.1364, 0.2848, 0.385 , 0.7814, 0.543 ],



