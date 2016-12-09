OKG4 G4GUN Shakedown
======================

::

    OKG4Test --g4gun --compute --save


OKG4 running does both:

* G4 simulation/propagation with Opticks style recording 
* OK propagation using gensteps collected from G4

The gensteps specify position and number of photons, 
so same photon count between them is a given.  

The history and material differences are all bugs to fix. 


Red Gensteps everywhere issue
------------------------------

::
  
   tg4gun-
   tg4gun-t 



FIXED issue: G4Gun ignoring gunconfig argument and using default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:optickscore blyth$ opticks-find g4gunconfig
    ./bin/ggv.bash:       --g4gun --g4gundbg --g4gunconfig "$(join _ ${g4gun_config[@]})" \
    ./tests/tg4gun.bash:#       --g4gun --g4gundbg --g4gunconfig "$(join _ ${g4gun_config[@]})" \
    ./tests/tg4gun.bash:       --g4gun --g4gundbg --g4gunconfig "$(join _ ${g4gun_config[@]})" \
    ./optickscore/OpticksCfg.cc:       m_g4gunconfig(""),
    ./optickscore/OpticksCfg.cc:       ("g4gunconfig",   boost::program_options::value<std::string>(&m_g4gunconfig), "g4gun configuration" );
    ./optickscore/OpticksCfg.cc:    return m_g4gunconfig ;
    ./optickscore/OpticksCfg.hh:     std::string m_g4gunconfig ;
    ./.hg/last-message.txt:find that tg4gun currently ignoring g4gunconfig argument and using defaults
    simon:opticks blyth$ 
    simon:opticks blyth$ 
    simon:opticks blyth$ vi optickscore/OpticksCfg.cc
    simon:opticks blyth$ opticks-find getG4GunConfig
    ./cfg4/CGenerator.cc:    std::string gunconfig = m_hub->getG4GunConfig() ;
    ./optickscore/OpticksCfg.cc:const std::string& OpticksCfg<Listener>::getG4GunConfig()
    ./opticksgeo/OpticksHub.cc:std::string OpticksHub::getG4GunConfig()
    ./optickscore/OpticksCfg.hh:     const std::string& getG4GunConfig();
    ./opticksgeo/OpticksHub.hh:       std::string    getG4GunConfig();
    simon:opticks blyth$ 





Genstep visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The formerly considered arbitrary param.x should actually be 1 (current default is 25) 
as vdir is actually DeltaPosition.

With param.x 25 get crazy red viz, but even with the correct 1 the scintillation gensteps mostly
hide the photon propagation. And this is with scintillation yield dialed down. 

* TODO: record style time input genstep visualization ? almost certainly would need to 
  be used as an alternative to photon record propagation


Lack of scintillation propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Position of propagation suggests its just original cerenkov. 

* TODO: interface for live genstep selection (CK or SI) during gun running 
  and modulo scaledown 


Genstep examination
~~~~~~~~~~~~~~~~~~~~~

* half scint gs have zero photons, due to scnt loop with 
  all Num used up on first turn, DONE: added condition at collection to skip these

* gs is mixed CK, SI : simpler CK in more mature state producing
  observed upwards propagation


g4gun.py min/avg/max of genstep position/time::

    array([[ -20059.457 ,  -18087.8789,  -16097.7148],
           [-801680.75  , -799696.    , -797717.5   ],
           [  -9092.    ,   -5589.7075,   -2110.8208],
           [      0.1   ,       4.5868,     974.2122]], dtype=float32)


Note suspicious doubled up 0/1 for scintillation, probably the scintillation imp got abandoned
when dived into validation comparisons using tconcentric::

    In [11]: evt.gs[:100,0].view(np.int32)   #  sid/parentId/materialIndex/numPhotons 
    Out[11]: 
    A()sliced
    A([[  1,   1,  95, 102],
           [  2,   1,  95,   0],
           [  2,   1,  95,   1],
           [  2, 103,  95,   0],
           [  2, 103,  95,   1],
           [  2, 102,  95,   0],
           [  2, 102,  95,   1],
           [  2, 100,  95,   0],
           [  2, 100,  95,   1],
           [  2,  94,  95,   0],
           [  2,  94,  95,   1],
           [  2,  93,  95,   0],
           [  2,  93,  95,   1],
           [  2,  91,  95,   0],
           [  2,  91,  95,   1],
    ...
    In [13]: evt.gs[-10:,0].view(np.int32)
    Out[13]: 
    A()sliced
    A([[     2, 479475,     95,      0],
           [     2, 479475,     95,      1],
           [     2, 479382,     95,      0],
           [     2, 479382,     95,      1],
           [     2, 479477,     95,      0],
           [     2, 479477,     95,      1],
           [     2, 479381,     95,      0],
           [     2, 479381,     95,      1],
           [     2, 479377,     95,      0],
           [     2, 479377,     95,      1]], dtype=int32)




genstep rendering uses p2l Rdr, currently the only user of the p2l shaders::

     448 void Scene::initRenderers()
     449 {
     450     LOG(debug) << "Scene::initRenderers "
     451               << " shader_dir " << m_shader_dir
     452               << " shader_incl_path " << m_shader_incl_path
     453                ;
     454 
     455     assert(m_shader_dir);
     456 
     457     m_device = new Device();
     458 
     459     m_colors = new Colors(m_device);
     460 
     461     m_global_renderer = new Renderer("nrm", m_shader_dir, m_shader_incl_path );
     462     m_globalvec_renderer = new Renderer("nrmvec", m_shader_dir, m_shader_incl_path );
     463     m_raytrace_renderer = new Renderer("tex", m_shader_dir, m_shader_incl_path );
     464 
     465    // small array of instance renderers to handle multiple assemblies of repeats 
     466     for( unsigned int i=0 ; i < MAX_INSTANCE_RENDERER ; i++)
     467     {
     468         m_instance_mode[i] = false ;
     469         m_instance_renderer[i] = new Renderer("inrm", m_shader_dir, m_shader_incl_path );
     470         m_instance_renderer[i]->setInstanced();
     471 
     472         m_bbox_mode[i] = false ;
     473         m_bbox_renderer[i] = new Renderer("inrm", m_shader_dir, m_shader_incl_path );
     474         m_bbox_renderer[i]->setInstanced();
     475         m_bbox_renderer[i]->setWireframe(false);  // wireframe is much slower than filled
     476     }
     477 
     478     //LOG(info) << "Scene::init geometry_renderer ctor DONE";
     479 
     480     m_axis_renderer = new Rdr(m_device, "axis", m_shader_dir, m_shader_incl_path );
     481 
     482     m_genstep_renderer = new Rdr(m_device, "p2l", m_shader_dir, m_shader_incl_path);

::

    simon:ok blyth$ opticks-find p2l
    ./externals/optix.bash:* p2l: genstep
    ./oglrap/oglrap.bash:  and p2l (point to line) geometry shader based on my ancient one
    ./oglrap/Scene.cc:    m_genstep_renderer = new Rdr(m_device, "p2l", m_shader_dir, m_shader_incl_path);
    ./oglrap/Scene.cc:    m_genstep_renderer = new Rdr(m_device, "p2l", m_shader_dir, m_shader_incl_path);
    simon:opticks blyth$ 


Looks like just need to form an attribute to grab the steplength which 
can then scale the mom direction by instead of using arbitray Param.x.

Nope, the vdir is actually absolute delta position so it duplicates the 
info in the step length.



oglrap/gl/p2l/vert.glsl::

     01 #version 400
      2 
      3 // p2l passthrough to geometry shader
      4 
      5 uniform mat4 ModelViewProjection ;
      6 uniform mat4 ModelView ;
      7 
      8 layout(location = 0) in vec4 vpos ;
      9 layout(location = 1) in vec4 vdir ;
     10 
     11 out vec3 colour;
     12 out vec3 direction ;
     13 
     14 
     15 void main ()
     16 {
     17     colour = vec3(1.0,0.0,0.0) ;
     18     direction = vdir.xyz ;
     19     gl_Position = vec4( vpos.xyz, 1.0);
     20 }   
     21 

oglrap/gl/p2l/geom.glsl::

     01 #version 400
      2 
      3 uniform mat4 ModelViewProjection ;
      4 uniform vec4 Param ;
      5 in vec3 colour[];
      6 in vec3 direction[];
      7 
      8 // https://www.opengl.org/wiki/Geometry_Shader
      9 
     10 layout (points) in;
     11 layout (line_strip, max_vertices = 2) out;
     12 
     13 out vec3 fcolour ;
     14 
     15 
     16 void main ()
     17 {
     18     gl_Position = ModelViewProjection * gl_in[0].gl_Position ;
     19     fcolour = colour[0] ;
     20     EmitVertex();
     21 
     22     gl_Position = ModelViewProjection * ( gl_in[0].gl_Position + Param.x*vec4(direction[0], 0.) ) ;
     23     fcolour = colour[0] ;
     24     EmitVertex();
     25 
     26     EndPrimitive();
     27 
     28 }



tg4gun.py examine gensteps shows vdir to actually be non-normalized DeltaPosition::

    In [10]: gs[:100,(1,2)]
    Out[10]: 
    A()sliced
    A([[[ -18079.4531, -799699.4375,   -6606.    ,       0.1   ],
            [      0.    ,       0.    ,       0.7653,       0.7653]],

           [[ -18079.4531, -799699.4375,   -6606.    ,       0.1   ],
            [      0.    ,       0.    ,       0.7653,       0.7653]],

           [[ -18079.4531, -799699.4375,   -6605.9136,       0.1003],
            [      0.    ,      -0.    ,       0.    ,       0.    ]],

           [[ -18079.4531, -799699.4375,   -6605.3418,       0.1022],
            [   -231.3343,      -5.7752,     209.7892,     312.3466]],

           [[ -18079.4531, -799699.4375,   -6605.7944,       0.1007],
            [     -0.    ,       0.0002,       0.0001,       0.0002]],

           [[ -18079.4531, -799699.4375,   -6605.9741,       0.1001],
            [   -103.424 ,     -85.0688,     120.7377,     180.3076]],

           [[ -18079.4531, -799699.4375,   -6605.3564,       0.1022],
            [     -0.0001,      -0.    ,       0.0001,       0.0001]],

           [[ -18079.4531, -799699.4375,   -6605.8066,       0.1006],
            [     -0.0014,      -0.0007,       0.0015,       0.0022]],

           [[ -18079.4531, -799699.4375,   -6605.8101,       0.1006],
            [      0.0002,       0.0001,       0.0002,       0.0003]],

           [[ -18079.4531, -799699.4375,   -6605.3013,       0.1023],
            [     -0.    ,       0.    ,       0.    ,       0.0001]],

           [[ -18079.4531, -799699.4375,   -6605.3013,       0.1023],
            [     50.4503,      95.0544,     -79.5832,     133.8434]],




Comparing length of the DeltaPosition with the stepLength shows several 100 
deviations, most of them are Cerenkov steps.::

    In [18]: df = np.sqrt(np.sum(gs[:,2,:3]*gs[:,2,:3], axis=1)) - gs[:,2,3] 

    In [19]: df
    A([-0., -0., -0., ...,  0.,  0.,  0.], dtype=float32)

    In [20]: df.min()
    A(-0.41480427980422974, dtype=float32)

    In [21]: df.max()
    A(0.000244140625, dtype=float32)


    In [37]: np.count_nonzero(df < -0.01)
    Out[37]: 424

    In [38]: np.count_nonzero(df > 0.01)
    Out[38]: 0

    In [39]: np.count_nonzero(df < 0.01)
    Out[39]: 174845


    In [24]: gs[:,2][df < -0.01]
    Out[24]: 
    A()sliced
    A([[ 0.2876, -0.6129,  0.516 ,  0.9698],
           [ 0.0663,  0.5022,  0.6936,  1.1281],
           [ 0.0663,  0.5022,  0.6936,  1.1281],
           ..., 
           [-0.3708,  0.272 ,  0.731 ,  1.0429],
           [-0.2866,  0.1007,  0.3472,  0.5684],
           [-0.0442,  0.2691,  0.0583,  0.4641]], dtype=float32)

    In [25]: gs[:,2][df > -0.01]
    A([[  0.    ,   0.    ,   0.7653,   0.7653],
           [  0.    ,   0.    ,   0.7653,   0.7653],
           [  0.    ,  -0.    ,   0.    ,   0.    ],
           ..., 
           [ 11.8779,   7.8823,   3.869 ,  14.771 ],
           [ -0.0207,   0.0142,   0.0077,   0.0263],
           [ -0.0024,   0.0008,   0.0012,   0.0028]], dtype=float32)


::

    321     // OPTICKS STEP COLLECTION : STEALING THE STACK
    322     {
    323         const G4ParticleDefinition* definition = aParticle->GetDefinition();
    324         G4ThreeVector deltaPosition = aStep.GetDeltaPosition();
    325         G4int materialIndex = aMaterial->GetIndex();
    326         CCollector::Instance()->collectCerenkovStep(
    327 
    328                0,                  // 0     id:zero means use cerenkov step count 
    329                aTrack.GetTrackID(),
    330                materialIndex,
    331                NumPhotons,
    332 
    333                x0.x(),                // 1
    334                x0.y(),
    335                x0.z(),
    336                t0,
    337 
    338                deltaPosition.x(),     // 2
    339                deltaPosition.y(),
    340                deltaPosition.z(),
    341                aStep.GetStepLength(),
    342 


::

     625             // OPTICKS STEP COLLECTION : STEALING THE STACK
     626             if(Num > 0)
     627             {
     628                 const G4ParticleDefinition* definition = aParticle->GetDefinition();
     629                 G4ThreeVector deltaPosition = aStep.GetDeltaPosition();
     630                 CCollector::Instance()->collectScintillationStep(
     631 
     632                        0,                  // 0     id:zero means use scintillation step count 
     633                        aTrack.GetTrackID(),
     634                        materialIndex,
     635                        Num,
     636 
     637                        x0.x(),                // 1
     638                        x0.y(),
     639                        x0.z(),
     640                        t0,
     641 
     642                        deltaPosition.x(),     // 2
     643                        deltaPosition.y(),
     644                        deltaPosition.z(),
     645                        aStep.GetStepLength(),



Collecting the stepLength within Scintillation/Cerenkov processes 
results in relationship between deltaPosition and stepLength that in some cases 
(400 out of 175000) us not as would expect. But this is only a fraction of a mm difference
so can probably ignore it.

g4-cls G4Step::

    106    // step length
    107    G4double GetStepLength() const;
    108    void SetStepLength(G4double value);
    109     // Before the end of the AlongStepDoIt loop,StepLength keeps
    110     // the initial value which is determined by the shortest geometrical Step
    111     // proposed by a physics process. After finishing the AlongStepDoIt,
    112     // it will be set equal to 'StepLength' in G4Step. 
    113 

    186 // Member data
    187    G4StepPoint* fpPreStepPoint;
    188    G4StepPoint* fpPostStepPoint;
    189    G4double fStepLength;
    190      // Step length which may be updated at each invocation of 
    191      // AlongStepDoIt and PostStepDoIt


    063 inline
     64  G4double G4Step::GetStepLength() const
     65  {
     66    return fStepLength;
     67  }
     68 
     69 inline
     70  void G4Step::SetStepLength(G4double value)
     71  {
     72    fStepLength = value;
     73  }
     74 
     75 inline
     76  G4ThreeVector G4Step::GetDeltaPosition() const
     77  {
     78    return fpPostStepPoint->GetPosition()
     79             - fpPreStepPoint->GetPosition();
     80  }


::

    simon:geant4_opticks_integration blyth$ g4-cc SetStepLength 
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/biasing/importance/src/G4ImportanceProcess.cc:  fGhostStep->SetStepLength(step.GetStepLength());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/biasing/importance/src/G4WeightCutOffProcess.cc:  fGhostStep->SetStepLength(step.GetStepLength());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/biasing/importance/src/G4WeightWindowProcess.cc:  fGhostStep->SetStepLength(step.GetStepLength());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:    fpTrack->SetStepLength(fpState->fPhysicalStep);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:    fpStep->SetStepLength(fpState->fPhysicalStep);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:  fpStep->SetStepLength(0.);  //the particle has stopped
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:  fpTrack->SetStepLength(0.);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/scoring/src/G4ParallelWorldProcess.cc:  fGhostStep->SetStepLength(step.GetStepLength());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/scoring/src/G4ParallelWorldProcess.cc:    fpHyperStep->SetStepLength(step.GetStepLength());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/scoring/src/G4ParallelWorldScoringProcess.cc:  fGhostStep->SetStepLength(step.GetStepLength());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/scoring/src/G4ScoreSplittingProcess.cc:        fSplitStep->SetStepLength(stepLength);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/scoring/src/G4ScoreSplittingProcess.cc:  fSplitStep->SetStepLength(step.GetStepLength());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForGamma.cc:  pStep->SetStepLength( 0.0 );
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForMSC.cc:  pStep->SetStepLength(theTrueStepLength);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4ParticleChangeForTransport.cc:  //pStep->SetStepLength( theTrueStepLength );
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/track/src/G4VParticleChange.cc:  pStep->SetStepLength( theTrueStepLength );
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/tracking/src/G4SteppingManager.cc:     fStep->SetStepLength( PhysicalStep );
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/tracking/src/G4SteppingManager.cc:     fTrack->SetStepLength( PhysicalStep );
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/tracking/src/G4SteppingManager2.cc:   fStep->SetStepLength( 0. );  //the particle has stopped
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/tracking/src/G4SteppingManager2.cc:   fTrack->SetStepLength( 0. );
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/visualization/RayTracer/src/G4RayTrajectory.cc:  trajectoryPoint->SetStepLength(aStep->GetStepLength());
    simon:geant4_opticks_integration blyth$ 


g4-cls G4VParticleChange::

    ### but this Propose not used in cfg4 

    145   public: // with description
    146     //---- the following methods are for TruePathLength ----
    147     G4double GetTrueStepLength() const;
    148     void  ProposeTrueStepLength(G4double truePathLength);
    149     //  Get/Propose theTrueStepLength
    150 


g4-cls G4SteppingManager::

    179      // Find minimum Step length demanded by active disc./cont. processes
    180      DefinePhysicalStepLength();
    181 
    182      // Store the Step length (geometrical length) to G4Step and G4Track
    183      fStep->SetStepLength( PhysicalStep );
    184      fTrack->SetStepLength( PhysicalStep );
    185      G4double GeomStepLength = PhysicalStep;
    186 
    187      // Store StepStatus to PostStepPoint
    188      fStep->GetPostStepPoint()->SetStepStatus( fStepStatus );
    189 
    190      // Invoke AlongStepDoIt 
    191      InvokeAlongStepDoItProcs();
    192 
    193      // Update track by taking into account all changes by AlongStepDoIt
    194      fStep->UpdateTrack();
    195 
    196      // Update safety after invocation of all AlongStepDoIts
    197      endpointSafOrigin= fPostStepPoint->GetPosition();
    198 //     endpointSafety=  std::max( proposedSafety - GeomStepLength, 0.);
    199      endpointSafety=  std::max( proposedSafety - GeomStepLength, kCarTolerance);
    200 
    201      fStep->GetPostStepPoint()->SetSafety( endpointSafety );
    202 
    203 #ifdef G4VERBOSE
    204                          // !!!!! Verbose
    205            if(verboseLevel>0) fVerbose->AlongStepDoItAllDone();
    206 #endif
    207 
    208      // Invoke PostStepDoIt
    209      InvokePostStepDoItProcs();



::

    simon:opticksnpy blyth$ g4-cc ProposeTrue 
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITTransportation.cc:  // fParticleChange.ProposeTrueStepLength(geometryStepLength) ;
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/management/src/G4ITTransportation.cc:  fParticleChange.ProposeTrueStepLength(track.GetStepLength());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/dna/processes/src/G4DNABrownianTransportation.cc:  fParticleChange.ProposeTrueStepLength(track.GetStepLength());
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/electromagnetic/utils/src/G4VMultipleScattering.cc:  fParticleChange.ProposeTrueStepLength(tPathLength);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/transportation/src/G4CoupledTransportation.cc:  fParticleChange.ProposeTrueStepLength(geometryStepLength) ;
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/transportation/src/G4CoupledTransportation.cc:  //fParticleChange. ProposeTrueStepLength( track.GetStepLength() ) ;
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/transportation/src/G4Transportation.cc:  fParticleChange.ProposeTrueStepLength(geometryStepLength) ;
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/transportation/src/G4Transportation.cc:  //fParticleChange. ProposeTrueStepLength( track.GetStepLength() ) ;
    simon:opticksnpy blyth$ 












Tao commit changing Scintillaton and Cerenkov
-----------------------------------------------

* https://bitbucket.org/simoncblyth/opticks/commits/55879cfc0aea49d57227bcb23a2ac92f01355082 


Debug running
---------------

::

    2016-11-29 21:13:16.803 INFO  [37041] [*DsG4Cerenkov::PostStepDoIt@460]  ParentID 1
    Process 8288 stopped
    * thread #1: tid = 0x90b1, 0x000000010708de44 libG4global.dylib`G4PhysicsVector::Value(this=0x0000000000000000, theEnergy=<unavailable>, lastIdx=0x00007fff5fbfd6d8) const + 4 at G4PhysicsVector.cc:501, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x10)
        frame #0: 0x000000010708de44 libG4global.dylib`G4PhysicsVector::Value(this=0x0000000000000000, theEnergy=<unavailable>, lastIdx=0x00007fff5fbfd6d8) const + 4 at G4PhysicsVector.cc:501
       498  G4double G4PhysicsVector::Value(G4double theEnergy, size_t& lastIdx) const
       499  {
       500    G4double y;
    -> 501    if(theEnergy <= edgeMin) {
       502      lastIdx = 0; 
       503      y = dataVector[0]; 
       504    } else if(theEnergy >= edgeMax) { 
    (lldb) bt
    * thread #1: tid = 0x90b1, 0x000000010708de44 libG4global.dylib`G4PhysicsVector::Value(this=0x0000000000000000, theEnergy=<unavailable>, lastIdx=0x00007fff5fbfd6d8) const + 4 at G4PhysicsVector.cc:501, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x10)
      * frame #0: 0x000000010708de44 libG4global.dylib`G4PhysicsVector::Value(this=0x0000000000000000, theEnergy=<unavailable>, lastIdx=0x00007fff5fbfd6d8) const + 4 at G4PhysicsVector.cc:501
        frame #1: 0x0000000103e1364b libcfg4.dylib`G4PhysicsVector::Value(this=0x0000000000000000, theEnergy=0.000018830823148420297) const + 43 at G4PhysicsVector.icc:249
        frame #2: 0x0000000103e33cc1 libcfg4.dylib`DsG4Cerenkov::GetPoolPmtQe(this=0x000000010b04d330, energy=0.000018830823148420297) const + 241 at DsG4Cerenkov.cc:842
        frame #3: 0x0000000103e32a13 libcfg4.dylib`DsG4Cerenkov::PostStepDoIt(this=0x000000010b04d330, aTrack=0x0000000135e8ef00, aStep=0x000000010c2547c0) + 3267 at DsG4Cerenkov.cc:347
        frame #4: 0x0000000104c88e2b libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x000000010c254630, np=<unavailable>) + 59 at G4SteppingManager2.cc:530
        frame #5: 0x0000000104c88d2b libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x000000010c254630) + 139 at G4SteppingManager2.cc:502
        frame #6: 0x0000000104c86909 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000010c254630) + 825 at G4SteppingManager.cc:209
        frame #7: 0x0000000104c90771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000010c2545f0, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #8: 0x0000000104be8727 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000010c254560, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #9: 0x0000000104b6a611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010c145d00, i_event=0) + 49 at G4RunManager.cc:399
        frame #10: 0x0000000104b6a4db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010c145d00, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #11: 0x0000000104b69913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010c145d00, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #12: 0x0000000103ef0433 libcfg4.dylib`CG4::propagate(this=0x000000010c145c40) + 1667 at CG4.cc:342
        frame #13: 0x0000000103fdf546 libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfed90) + 566 at OKG4Mgr.cc:86
        frame #14: 0x00000001000139ca OKG4Test`main(argc=2, argv=0x00007fff5fbfee78) + 1498 at OKG4Test.cc:57
        frame #15: 0x00007fff8ab4b5fd libdyld.dylib`start + 1
        frame #16: 0x00007fff8ab4b5fd libdyld.dylib`start + 1



Peculiarities
---------------

* scintillation dialed down by material override in cfg4
  but this will have same effect on G4 and OK 


Known sources of difference
----------------------------

* G4 is using stock (not DYB) scintillation but Opticks scintillation 
  not updated to handle stock gensteps   
  (this result is the MI)


::


    tokg4.py --src g4gun 

      A:seqhis_ana    1:dayabay 
                  41        0.354         492589       [2 ] CK AB
                   3        0.352         489714       [1 ] MI
             8cccc51        0.026          36768       [7 ] CK RE BT BT BT BT SA
                 451        0.025          34271       [3 ] CK RE AB
          cccbccccc1        0.019          26612       [10] CK BT BT BT BT BT BR BT BT BT
          cccccccc51        0.015          20339       [10] CK RE BT BT BT BT BT BT BT BT
            8cccc551        0.012          16259       [8 ] CK RE RE BT BT BT BT SA
                4551        0.010          14281       [4 ] CK RE RE AB
          ccbccccc51        0.008          11194       [10] CK RE BT BT BT BT BT BR BT BT
          ccccccc551        0.006           8303       [10] CK RE RE BT BT BT BT BT BT BT
           8cccc5551        0.005           7498       [9 ] CK RE RE RE BT BT BT BT SA
                 4c1        0.005           6533       [3 ] CK BT AB
               45551        0.004           6196       [5 ] CK RE RE RE AB
            4ccccc51        0.004           6007       [8 ] CK RE BT BT BT BT BT AB
          cbccccc551        0.004           5890       [10] CK RE RE BT BT BT BT BT BR BT
               4cc51        0.004           5550       [5 ] CK RE BT BT AB
          cccccc5551        0.004           4915       [10] CK RE RE RE BT BT BT BT BT BT
          cacccccc51        0.003           4779       [10] CK RE BT BT BT BT BT BT SR BT
           8cccccc51        0.003           4191       [9 ] CK RE BT BT BT BT BT BT SA
              4ccc51        0.003           4137       [6 ] CK RE BT BT BT AB
                         1392904         1.00 
       B:seqhis_ana   -1:dayabay 
                  4f        0.837        1166339       [2 ] GN AB
                 4cf        0.094         130309       [3 ] GN BT AB
          cccbcccccf        0.021          28980       [10] GN BT BT BT BT BT BR BT BT BT
                 4bf        0.007           9402       [3 ] GN BR AB
          bbbbbbbbbf        0.004           5184       [10] GN BR BR BR BR BR BR BR BR BR
                4ccf        0.003           4226       [4 ] GN BT BT AB
                 40f        0.002           3381       [3 ] GN ?0? AB
          ccbccccccf        0.002           2936       [10] GN BT BT BT BT BT BT BR BT BT
          ccbcccc0cf        0.002           2288       [10] GN BT ?0? BT BT BT BT BR BT BT
               4cccf        0.001           1879       [5 ] GN BT BT BT AB
          c00b00cccf        0.001           1669       [10] GN BT BT BT ?0? ?0? BR ?0? ?0? BT
             4cccccf        0.001           1585       [7 ] GN BT BT BT BT BT AB
            b00cc0cf        0.001           1498       [8 ] GN BT ?0? BT BT ?0? ?0? BR
          bcccbcccbf        0.001           1335       [10] GN BR BT BT BT BR BT BT BT BR
            8ccccccf        0.001           1260       [8 ] GN BT BT BT BT BT BT SA
          ccccbccccf        0.001           1116       [10] GN BT BT BT BT BR BT BT BT BT
          cbcccc0ccf        0.001            986       [10] GN BT BT ?0? BT BT BT BT BR BT
          bccccccccf        0.001            952       [10] GN BT BT BT BT BT BT BT BT BR
          cccbccbccf        0.001            914       [10] GN BT BT BR BT BT BR BT BT BT
              4ccccf        0.001            767       [6 ] GN BT BT BT BT AB
                         1392904         1.00 
       A:seqmat_ana    1:dayabay 
                   0        0.352         489714       [1 ] ?0?
                  11        0.233         323915       [2 ] Gd Gd
                  22        0.063          88210       [2 ] LS LS
             4432311        0.024          33745       [7 ] Gd Gd Ac LS Ac MO MO
                 111        0.021          29143       [3 ] Gd Gd Gd
                  44        0.020          28252       [2 ] MO MO
                  33        0.020          28178       [2 ] Ac Ac
                  ff        0.016          22849       [2 ] Ai Ai
          3343343231        0.015          21303       [10] Gd Ac LS Ac MO Ac Ac MO Ac Ac
            44323111        0.012          16966       [8 ] Gd Gd Gd Ac LS Ac MO MO
                1111        0.009          13196       [4 ] Gd Gd Gd Gd
          4433432311        0.006           8987       [10] Gd Gd Ac LS Ac MO Ac Ac MO MO
           443231111        0.006           8181       [9 ] Gd Gd Gd Gd Ac LS Ac MO MO
          4432311111        0.005           6275       [10] Gd Gd Gd Gd Gd Ac LS Ac MO MO
               11111        0.004           6007       [5 ] Gd Gd Gd Gd Gd
          fff3432311        0.003           4573       [10] Gd Gd Ac LS Ac MO Ac Ai Ai Ai
          3334323111        0.003           4443       [10] Gd Gd Gd Ac LS Ac MO Ac Ac Ac
          3343231111        0.003           3595       [10] Gd Gd Gd Gd Ac LS Ac MO Ac Ac
                   6        0.003           3563       [1 ] Iw
            aa332311        0.002           3450       [8 ] Gd Gd Ac LS Ac Ac ES ES
                         1392904         1.00 
       B:seqmat_ana   -1:dayabay 
                  11        0.837        1166374       [2 ] Gd Gd
                 111        0.103         143334       [3 ] Gd Gd Gd
          1111111111        0.046          63409       [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Gd
                1111        0.004           6236       [4 ] Gd Gd Gd Gd
               11111        0.003           3749       [5 ] Gd Gd Gd Gd Gd
            11111111        0.002           3447       [8 ] Gd Gd Gd Gd Gd Gd Gd Gd
             1111111        0.002           2367       [7 ] Gd Gd Gd Gd Gd Gd Gd
              111111        0.002           2107       [6 ] Gd Gd Gd Gd Gd Gd
           111111111        0.001           1881       [9 ] Gd Gd Gd Gd Gd Gd Gd Gd Gd
                         1392904         1.00 



