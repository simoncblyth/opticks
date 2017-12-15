RNG_seq_off_by_one
===================


Dirty Half Dozen
-----------------


::

    In [1]: ab.maligned
    Out[1]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [2]: ab.dumpline(ab.maligned)
          0   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          1   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          2  14510 :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          3  49786 :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
          4  69653 :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
          5  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 



Launch
--------

::

    tboolean-;tboolean-box --okg4 --align --mask 1230  --pindex 0 --pindexlog -DD   
    tboolean-;tboolean-box --okg4 --align --mask 9041  --pindex 0 --pindexlog -DD   
    tboolean-;tboolean-box --okg4 --align --mask 14510 --pindex 0 --pindexlog -DD   
    tboolean-;tboolean-box --okg4 --align --mask 49786 --pindex 0 --pindexlog -DD   
    tboolean-;tboolean-box --okg4 --align --mask 69653 --pindex 0 --pindexlog -DD   
    tboolean-;tboolean-box --okg4 --align --mask 77962 --pindex 0 --pindexlog -DD   


    ucf.py 9041


Location misaligns
-------------------


::

    tboolean-;tboolean-box --okg4 --align --mask 1230  --pindex 0 --pindexlog -DD  --dbgnojump


    [  6]                                       OpAbsorption :     0.438019574 :    : 0.438019574 : 0.438019574 : 1 

    2017-12-15 14:35:12.704 INFO  [730136] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-15 14:35:12.704 ERROR [730136] [CRandomEngine::poststep@236] CRandomEngine::poststep _noZeroSteps 1 backseq -3 --dbgnojump YES
    2017-12-15 14:35:12.704 FATAL [730136] [CRandomEngine::poststep@244] CRandomEngine::poststep rewind inhibited by option: --dbgnojump 
    flatExit: mrk:-# crfc:    8 df:1.10290554e-10 u_g4:0.714031577 u_ok:0.714031577 loc_g4:          OpBoundary loc_ok:          OpRayleigh  : lucf : 29    
    rayleigh_scatter_align p.direction (0 0 -1)
    rayleigh_scatter_align p.polarization (-0 1 -0)
    rayleigh_scatter_align.do u_OpRayleigh:0.714031577
     [  7]                                         OpRayleigh :     0.714031577 :    : 0.714031577 : 0.714031577 : 3 

    Process 51835 stopped





    tboolean-;tboolean-box --okg4 --align --mask 1230  --pindex 0 --pindexlog -DD 


    flatExit: mrk:-- crfc:    4 df:3.70178332e-11 u_g4:0.938713491 u_ok:0.938713491 loc_g4:OpBoundary_DiDiTransCoeff loc_ok:OpBoundary_DiDiTransCoeff  : lucf : 29    
    propagate_at_boundary  u_OpBoundary_DiDiTransCoeff:0.938713491  reflect:1   TransCoeff:   0.93847  c2c2:    1.0000 tir:0  pos (  -37.8785    11.8230  -100.0000)
     [  3]                          OpBoundary_DiDiTransCoeff :     0.938713491 :    : 0.938713491 : 0.938713491 : 1 


    //                     opticks.ana.loc.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[1] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                      /TransCoeff :  0.938471  
    //                                                                              /_u :  0.938713  
    //                                                                       /_transmit : False 

    //                   opticks.ana.loc.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[1] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                       .theStatus : (DsG4OpBoundaryProcessStatus) theStatus = FresnelReflection 
    flatExit: mrk:-- crfc:    5 df:3.44848594e-11 u_g4:0.753801465 u_ok:0.753801465 loc_g4:          OpBoundary loc_ok:          OpBoundary  : lucf : 29    
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:1
    propagate_to_boundary  u_OpBoundary:0.753801465 speed:299.79245
     [  4]                                         OpBoundary :     0.753801465 :    : 0.753801465 : 0.753801465 : 2 

    flatExit: mrk:-- crfc:    6 df:4.58282523e-10 u_g4:0.999846756 u_ok:0.999846756 loc_g4:          OpRayleigh loc_ok:          OpRayleigh  : lucf : 29    
    propagate_to_boundary  u_OpRayleigh:0.999846756   scattering_length(s.material1.z):1000000 scattering_distance:153.25528
     [  5]                                         OpRayleigh :     0.999846756 :    : 0.999846756 : 0.999846756 : 1 

    flatExit: mrk:-- crfc:    7 df:3.11492943e-10 u_g4:0.438019574 u_ok:0.438019574 loc_g4:        OpAbsorption loc_ok:        OpAbsorption  : lucf : 29    
    propagate_to_boundary  u_OpAbsorption:0.438019574   absorption_length(s.material1.y):10000000 absorption_distance:8254917
     [  6]                                       OpAbsorption :     0.438019574 :    : 0.438019574 : 0.438019574 : 1 

    2017-12-15 14:29:48.568 INFO  [727965] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-15 14:29:48.568 ERROR [727965] [CRandomEngine::poststep@236] CRandomEngine::poststep _noZeroSteps 1 backseq -3 --dbgnojump NO
    flatExit: mrk:*# crfc:    8 df:0.039769888 u_g4:0.753801465 u_ok:0.714031577 loc_g4:          OpBoundary loc_ok:          OpRayleigh  : lucf : 29    
    rayleigh_scatter_align p.direction (0 0 -1)
    rayleigh_scatter_align p.polarization (-0 1 -0)
    rayleigh_scatter_align.do u_OpRayleigh:0.714031577
     [  7]                                         OpRayleigh :     0.714031577 :    : 0.714031577 : 0.714031577 : 3 


    * OpBoundary is 1st consumption of the step



       1230 : /tmp/blyth/opticks/ox_1230.log  
     [  0]                                         OpBoundary :   0.00111702492 :    : 0.001117025 : 0.001117025 : 3 
     [  1]                                         OpRayleigh :      0.50264734 :    : 0.502647340 : 0.502647340 : 1 
     [  2]                                       OpAbsorption :     0.601504147 :    : 0.601504147 : 0.601504147 : 1 
     [  3]                          OpBoundary_DiDiTransCoeff :     0.938713491 :    : 0.938713491 : 0.938713491 : 1 

     [  4]                                         OpBoundary :    *0.753801465* :    : 0.753801465 : 0.753801465 : 2 
     [  5]                                         OpRayleigh :     0.999846756 :    : 0.999846756 : 0.999846756 : 1 
     [  6]                                       OpAbsorption :     0.438019574 :    : 0.438019574 : 0.438019574 : 1 

     [  7]                                         OpRayleigh :    *0.714031577* :    : 0.714031577 : 0.714031577 : 3 
     [  8]                                         OpRayleigh :     0.330403954 :    : 0.330403954 : 0.330403954 : 1 
     [  9]                                         OpRayleigh :     0.570741653 :    : 0.570741653 : 0.570741653 : 1 
     [ 10]                                         OpRayleigh :     0.375908673 :    : 0.375908673 : 0.375908673 : 1 
     [ 11]                                         OpRayleigh :      0.78497833 :    : 0.784978330 : 0.784978330 : 1 

     [ 12]                                         OpBoundary :     0.892654359 :    : 0.892654359 : 0.892654359 : 6 
     [ 13]                                         OpRayleigh :     0.441063195 :    : 0.441063195 : 0.441063195 : 1 
     [ 14]                                       OpAbsorption :     0.773742437 :    : 0.773742437 : 0.773742437 : 1 
     [ 15]                          OpBoundary_DiDiTransCoeff :     0.556839108 :    : 0.556839108 : 0.556839108 : 1 




What could go wrong with the rewind ?
----------------------------------------

* hmm why not seeing the burnt flatExit calls


::

    196 double CRandomEngine::flat()
    197 {
    198     if(!m_internal) m_location = CurrentProcessName();
    199     assert( m_current_record_flat_count < m_curand_nv );
    200     m_flat =  _flat() ;
    201     m_current_record_flat_count++ ; 
    202     m_current_step_flat_count++ ; 
    203     return m_flat ;   // (*lldb*) flatExit
    204 }   


    228 void CRandomEngine::poststep()
    229 {
    230     if(m_ctx._noZeroSteps > 0)
    231     {
    232         int backseq = -m_current_step_flat_count ;
    233         bool dbgnojump = m_ok->isDbgNoJump() ;
    234 
    235         LOG(error) << "CRandomEngine::poststep"
    236                    << " _noZeroSteps " << m_ctx._noZeroSteps
    237                    << " backseq " << backseq
    238                    << " --dbgnojump " << ( dbgnojump ? "YES" : "NO" )
    239                    ;
    240 
    241         if( dbgnojump )
    242         {
    243             LOG(fatal) << "CRandomEngine::poststep rewind inhibited by option: --dbgnojump " ;
    244         }
    245         else
    246         {
    247             jump(backseq);
    248         }
    249     }
    250 
    251     m_current_step_flat_count = 0 ;
    252 
    253     if( m_locseq )
    254     {
    255         m_locseq->poststep();
    256         LOG(info) << CProcessManager::Desc(m_ctx._process_manager) ;
    257     }
    258 }




Full unmasked run into tag 2
-------------------------------

For access to some non-maligned photons that scatter, do a full run into tag 2

::

    tboolean-;TBOOLEAN_TAG=2 tboolean-box --okg4 --align 
    tboolean-;TBOOLEAN_TAG=2 tboolean-box-ip


    In [1]: ab.maligned
    Out[1]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [2]: ab.dum
    ab.dump      ab.dumpline  

    In [2]: ab.dumpline(ab.maligned)
          0   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          1   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          2  14510 :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          3  49786 :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
          4  69653 :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
          5  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 


::

    In [1]: ab.aselhis = "TO BT SC BT SA"

    In [2]: ab.a.where
    Out[2]: array([ 4608, 17968, 61921, 86722, 91760, 93259, 94773])

    In [3]: ab.b.where
    Out[3]: array([ 4608, 17968, 61921, 86722, 91760, 93259, 94773])

    In [4]: ab.dumpline(ab.a.where)
          0   4608 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          1  17968 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          2  61921 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          3  86722 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          4  91760 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          5  93259 :                                     TO BT SC BT SA                                     TO BT SC BT SA 
          6  94773 :                                     TO BT SC BT SA                                     TO BT SC BT SA 


::

    tboolean-;tboolean-box --okg4 --align --mask 4608 --pindex 0 --pindexlog -DD 






Try blanket inhibiting the jump --dbgnojump
-----------------------------------------------

::

    tboolean-;tboolean-box --okg4 --align --mask 1230  --pindex 0 --pindexlog -DD --dbgnojump   
    tboolean-;tboolean-box --okg4 --align --mask 9041  --pindex 0 --pindexlog -DD --dbgnojump   
    tboolean-;tboolean-box --okg4 --align --mask 14510 --pindex 0 --pindexlog -DD --dbgnojump   
    tboolean-;tboolean-box --okg4 --align --mask 49786 --pindex 0 --pindexlog -DD --dbgnojump   
    tboolean-;tboolean-box --okg4 --align --mask 69653 --pindex 0 --pindexlog -DD --dbgnojump   
    tboolean-;tboolean-box --okg4 --align --mask 77962 --pindex 0 --pindexlog -DD --dbgnojump   


Switching off the rewind with --dbgnojump keeps the RNG seq aligned, but get different 
seqhis-tories.  Need procName alignment checking too.




Review Rewinding
------------------

Rewinding noted in :doc:`BR_PhysicalStep_zero_misalignment`

::

    Smouldering evidence : PhysicalStep-zero/StepTooSmall results in RNG mis-alignment 
    ------------------------------------------------------------------------------------

    Some G4 technicality yields zero step at BR, that means the lucky scatter 
    throw that Opticks saw was not seen by G4 : as the sequence gets out of alignment.


Zero steps result in G4 burning an entire steps RNGs compared to Opticks.  
The solution was to jump back in the sequence on the G4 side.
However for the misaligned six (the 3~4 studied) all appear to have an improper
jump back.


::

    231 void CRandomEngine::poststep()
    232 {
    233     if(m_ctx._noZeroSteps > 0)
    234     {
    235         int backseq = -m_current_step_flat_count ;
    236         LOG(error) << "CRandomEngine::poststep"
    237                    << " _noZeroSteps " << m_ctx._noZeroSteps
    238                    << " backseq " << backseq
    239                    ;
    240         jump(backseq);
    241     }
    242 
    243     m_current_step_flat_count = 0 ;
    244 
    245     if( m_locseq )
    246     {
    247         m_locseq->poststep();
    248         LOG(info) << CProcessManager::Desc(m_ctx._process_manager) ;
    249     }
    250 }


Review POstStep ClearNumberOfInteractionLengthLeft
------------------------------------------------------

At the end of everystep the RNG for AB and SC are cleared, in order to 
force G4VProcess::ResetNumberOfInteractionLengthLeft for every step, as
that is how Opticks works with AB and SC RNG consumption at every "propagate_to_boundary".

See :doc:`stepping_process_review`

::

     59 /*
     60 
     61      95 void G4VProcess::ResetNumberOfInteractionLengthLeft()
     62      96 {
     63      97   theNumberOfInteractionLengthLeft =  -std::log( G4UniformRand() );
     64      98   theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft;
     65      99 }
     66 
     67 */
     68 
     69 
     70 void CProcessManager::ClearNumberOfInteractionLengthLeft(G4ProcessManager* proMgr, const G4Track& aTrack, const G4Step& aStep)
     71 {
     72     G4ProcessVector* pl = proMgr->GetProcessList() ;
     73     G4int n = pl->entries() ;
     74 
     75     for(int i=0 ; i < n ; i++)
     76     {
     77         G4VProcess* p = (*pl)[i] ;
     78         const G4String& name = p->GetProcessName() ;
     79         bool is_ab = name.compare("OpAbsorption") == 0 ;
     80         bool is_sc = name.compare("OpRayleigh") == 0 ;
     81         //bool is_bd = name.compare("OpBoundary") == 0 ;
     82         if( is_ab || is_sc )
     83         {
     84             G4VDiscreteProcess* dp = dynamic_cast<G4VDiscreteProcess*>(p) ;
     85             assert(dp);   // Transportation not discrete
     86             dp->G4VDiscreteProcess::PostStepDoIt( aTrack, aStep );
     87             // devious way to invoke the protected ClearNumberOfInteractionLengthLeft via G4VDiscreteProcess::PostStepDoIt
     88         }
     89     }
     90 }








Who gets ahead on consumption ?
----------------------------------

::

   LOOKS LIKE AN UN-NEEDED -3 REWIND CAUSES THE MIS-ALIGN, 

   HMM SOME ZERO STEPS DONT NEED REWIND ?

   PERHAPS A ZERO STEP FOLLOWING A STEP IN WHICH THE BOUNDARY PROCESS WINS SHOULD NOT REWIND ?
 



69653 
~~~~~~~

::

    tboolean-;tboolean-box --okg4 --align --mask 69653 --pindex 0 --pindexlog -DD 



    curi:69653 
       69653 : /tmp/blyth/opticks/ox_69653.log  
     [  0]                                      boundary_burn :    0.0819766819 :    : 0.081976682 : 0.081976682 : 3 
     [  1]                                         scattering :     0.490069658 :    : 0.490069658 : 0.490069658 : 1 
     [  2]                                         absorption :     0.800361693 :    : 0.800361693 : 0.800361693 : 1 
     [  3]                                            reflect :      0.50900209 :    : 0.509002090 : 0.509002090 : 1 
     [  4]                                      boundary_burn :     0.793467045 :    : 0.793467045 : 0.793467045 : 2 
     [  5]                                         scattering :     0.999958992 :    : 0.999958992 : 0.999958992 : 1 
     [  6]                                         absorption :     0.475769788 :    : 0.475769788 : 0.475769788 : 1 
     [  7]                                               rsa0 :     0.416864127 :    : 0.416864127 : 0.416864127 : 3 
     [  8]                                               rsa1 :     0.186498553 :    : 0.186498553 : 0.186498553 : 1 
     [  9]                                               rsa2 :     0.985090375 :    : 0.985090375 : 0.985090375 : 1 
     [ 10]                                               rsa3 :    0.0522525758 :    : 0.052252576 : 0.052252576 : 1 
     [ 11]                                               rsa4 :     0.308176816 :    : 0.308176816 : 0.308176816 : 1 
     [ 12]                                      boundary_burn :     0.471794218 :    : 0.471794218 : 0.471794218 : 6 
     [ 13]                                         scattering :     0.792557418 :    : 0.792557418 : 0.792557418 : 1 
     [ 14]                                         absorption :      0.47266078 :    : 0.472660780 : 0.472660780 : 1 
     [ 15]                                            reflect :    *0.160018712* :    : 0.160018712 : 0.160018712 : 1 
     [ 16]                                      boundary_burn :     0.539000034 :    : 0.539000034 : 0.539000034 : 2 
     [ 17]                                         scattering :     0.493351549 :    : 0.493351549 : 0.493351549 : 1 
     [ 18]                                         absorption :    *0.831078768* :    : 0.831078768 : 0.831078768 : 1 
     [ 19]                                            reflect :     0.995906353 :    : 0.995906353 : 0.995906353 : 1 
     [ 20]                                      boundary_burn :     0.828557372 :    : 0.828557372 : 0.828557372 : 2 
     [ 21]                                         scattering :     0.159997851 :    : 0.159997851 : 0.159997851 : 1 





     [ 13]                                         scattering :     0.792557418 :    : 0.792557418 : 0.792557418 : 1 

    flatExit: mrk:   crfc:   15 df:4.69970729e-11 flat:0.47266078  ufval:0.47266078 :        OpAbsorption; : lufc : 29    
    propagate_to_boundary  u_absorption:0.47266078   absorption_length(s.material1.y):1000000 absorption_distance:749377.312
     [ 14]                                         absorption :      0.47266078 :    : 0.472660780 : 0.472660780 : 1 


    //                  opticks.ana.loc.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[19] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                       .theStatus : (DsG4OpBoundaryProcessStatus) theStatus = TotalInternalReflection 
    flatExit: mrk:   crfc:   16 df:2.82180779e-10 flat:*0.160018712*  ufval:0.160018712 :          OpBoundary; : lufc : 29    
    propagate_at_boundary  u_reflect:    0.160018712  reflect:1   TransCoeff:   0.00000  c2c2:   -1.2761 tir:1  pos (  133.7670    10.0854  -100.0000)
     [ 15]                                            reflect :     0.160018712 :    : 0.160018712 : 0.160018712 : 1 

    flatExit: mrk:   crfc:   17 df:3.32275429e-10 flat:0.539000034  ufval:0.539000034 :          OpRayleigh; : lufc : 29    
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:3
    propagate_to_boundary  u_boundary_burn:0.539000034 speed:165.028061
     [ 16]                                      boundary_burn :     0.539000034 :    : 0.539000034 : 0.539000034 : 2 

    flatExit: mrk:   crfc:   18 df:8.98590091e-11 flat:0.493351549  ufval:0.493351549 :        OpAbsorption; : lufc : 29    
    propagate_to_boundary  u_scattering:0.493351549   scattering_length(s.material1.z):1000000 scattering_distance:706533.25
     [ 17]                                         scattering :     0.493351549 :    : 0.493351549 : 0.493351549 : 1 

    2017-12-15 11:21:33.840 INFO  [650846] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-15 11:21:33.840 ERROR [650846] [CRandomEngine::poststep@236] CRandomEngine::poststep _noZeroSteps 1 backseq -3


    flatExit: mrk:** crfc:   19 df:0.671060056 flat:0.160018712  ufval:0.831078768 :          OpBoundary; : lufc : 29    
    propagate_to_boundary  u_absorption:0.831078768   absorption_length(s.material1.y):1000000 absorption_distance:185030.703
     [ 18]                                         absorption :     0.831078768 :    : 0.831078768 : 0.831078768 : 1 

    Process 27386 stopped
    * thread #1: tid = 0x9ee5e, 0x00000001044e063a libcfg4.dylib`CRandomEngine::flat(this=0x00000001100ca580) + 1082 at CRandomEngine.cc:206, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x00000001044e063a libcfg4.dylib`CRandomEngine::flat(this=0x00000001100ca580) + 1082 at CRandomEngine.cc:206
       203      //if(m_alignlevel > 1 || m_ctx._print) dumpFlat() ; 
       204      m_current_record_flat_count++ ; 
       205      m_current_step_flat_count++ ; 
    -> 206      return m_flat ;   // (*lldb*) flatExit
       207  }
       208  
       209  




77962
~~~~~~~~

::

    tboolean-;tboolean-box --okg4 --align --mask 77962 --pindex 0 --pindexlog -DD   


       77962 : /tmp/blyth/opticks/ox_77962.log  
     [  0]                                      boundary_burn :     0.587307692 :    : 0.587307692 : 0.587307692 : 3 
     [  1]                                         scattering :     0.367523879 :    : 0.367523879 : 0.367523879 : 1 
     [  2]                                         absorption :     0.368657529 :    : 0.368657529 : 0.368657529 : 1 
     [  3]                                            reflect :     0.883359611 :    : 0.883359611 : 0.883359611 : 1 
     [  4]                                      boundary_burn :     0.716171503 :    : 0.716171503 : 0.716171503 : 2 
     [  5]                                         scattering :    0.0115878591 :    : 0.011587859 : 0.011587859 : 1 
     [  6]                                         absorption :     0.265672505 :    : 0.265672505 : 0.265672505 : 1 
     [  7]                                            reflect :     0.959501982 :    : 0.959501982 : 0.959501982 : 1 
     [  8]                                      boundary_burn :    *0.974827707* :    : 0.974827707 : 0.974827707 : 2 
     [  9]                                         scattering :     0.999853075 :    : 0.999853075 : 0.999853075 : 1 
     [ 10]                                         absorption :     0.882926166 :    : 0.882926166 : 0.882926166 : 1 
     [ 11]                                               rsa0 :    *0.0676458701* :    : 0.067645870 : 0.067645870 : 3 
     [ 12]                                               rsa1 :     0.712023914 :    : 0.712023914 : 0.712023914 : 1 
     [ 13]                                               rsa2 :     0.388658017 :    : 0.388658017 : 0.388658017 : 1 
     [ 14]                                               rsa3 :     0.792805254 :    : 0.792805254 : 0.792805254 : 1 



    flatExit: mrk:   crfc:    8 df:2.64770539e-10 flat:0.959501982  ufval:0.959501982 :                      : lufc : 34    
    propagate_at_boundary  u_reflect:    0.959501982  reflect:1   TransCoeff:   0.93847  c2c2:    1.0000 tir:0  pos (  -29.0273    37.6855   100.0000)
     [  7]                                            reflect :     0.959501982 :    : 0.959501982 : 0.959501982 : 1 


    //                    opticks.ana.loc.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[25] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                      /TransCoeff :  0.938471  
    //                                                                              /_u :  0.959502  
    //                                                                       /_transmit : False 

    //                  opticks.ana.loc.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[19] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                       .theStatus : (DsG4OpBoundaryProcessStatus) theStatus = FresnelReflection 
    flatExit: mrk:   crfc:    9 df:1.86187732e-10 flat:*0.974827707*  ufval:0.974827707 :          OpBoundary; : lufc : 34    
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:2
    propagate_to_boundary  u_boundary_burn:0.974827707 speed:165.028061
     [  8]                                      boundary_burn :     0.974827707 :    : 0.974827707 : 0.974827707 : 2 

    flatExit: mrk:   crfc:   10 df:4.49371318e-10 flat:0.999853075  ufval:0.999853075 :          OpRayleigh; : lufc : 34    
    propagate_to_boundary  u_scattering:0.999853075   scattering_length(s.material1.z):1000000 scattering_distance:146.936249
     [  9]                                         scattering :     0.999853075 :    : 0.999853075 : 0.999853075 : 1 

    flatExit: mrk:   crfc:   11 df:5.75867132e-11 flat:0.882926166  ufval:0.882926166 :        OpAbsorption; : lufc : 34    
    propagate_to_boundary  u_absorption:0.882926166   absorption_length(s.material1.y):1000000 absorption_distance:124513.695
     [ 10]                                         absorption :     0.882926166 :    : 0.882926166 : 0.882926166 : 1 

    2017-12-15 11:16:26.480 INFO  [649101] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-15 11:16:26.480 ERROR [649101] [CRandomEngine::poststep@236] CRandomEngine::poststep _noZeroSteps 1 backseq -3


    flatExit: mrk:** crfc:   12 df:0.907181837 flat:0.974827707  ufval:*0.0676458701* :          OpBoundary; : lufc : 34    
    rayleigh_scatter_align p.direction (-0 -0 -1)
    rayleigh_scatter_align p.polarization (0 -1 0)
    rayleigh_scatter_align.do u_rsa0:0.0676458701
     [ 11]                                               rsa0 :    0.0676458701 :    : 0.067645870 : 0.067645870 : 3 

    Process 27097 stopped
    * thread #1: tid = 0x9e78d, 0x00000001044e063a libcfg4.dylib`CRandomEngine::flat(this=0x000000010f602e80) + 1082 at CRandomEngine.cc:206, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x00000001044e063a libcfg4.dylib`CRandomEngine::flat(this=0x000000010f602e80) + 1082 at CRandomEngine.cc:206
       203      //if(m_alignlevel > 1 || m_ctx._print) dumpFlat() ; 
       204      m_current_record_flat_count++ ; 
       205      m_current_step_flat_count++ ; 
    -> 206      return m_flat ;   // (*lldb*) flatExit
       207  }
       208  
       209  





???
~~~~~~


::


    .[ 10]                                               rsa2 :     0.775209486 :    : 0.775209486 : 0.775209486 : 1 
     [ 11]                                               rsa3 :     0.222410366 :    : 0.222410366 : 0.222410366 : 1 
     [ 12]                                               rsa4 :     0.434931546 :    : 0.434931546 : 0.434931546 : 1 
     [ 13]                                      boundary_burn :     0.971410215 :    : 0.971410215 : 0.971410215 : 6 
     [ 14]                                         scattering :     0.980197608 :    : 0.980197608 : 0.980197608 : 1 
     [ 15]                                         absorption :     0.124794453 :    : 0.124794453 : 0.124794453 : 1 
     [ 16]                                            reflect :      0.83465904 :    : 0.834659040 : 0.834659040 : 1 
     [ 17]                                      boundary_burn :     0.153918192 :    : 0.153918192 : 0.153918192 : 2 
     [ 18]                                         scattering :     0.400545776 :    : 0.400545776 : 0.400545776 : 1 
     [ 19]                                         absorption :     0.705055475 :    : 0.705055475 : 0.705055475 : 1 
     [ 20]                                            reflect :    *0.443446934*:    : 0.443446934 : 0.443446934 : 1   TIR
     [ 21]                                      boundary_burn :     0.806965649 :    : 0.806965649 : 0.806965649 : 2 
     [ 22]                                         scattering :     0.994345605 :    : 0.994345605 : 0.994345605 : 1 
     [ 23]                                         absorption :    *0.889802396*:    : 0.889802396 : 0.889802396 : 1 
     [ 24]                                            reflect :     0.970076799 :    : 0.970076799 : 0.970076799 : 1 
     [ 25]                                      boundary_burn :    0.0610740669 :    : 0.061074067 : 0.061074067 : 2 
     [ 26]                                         scattering :     0.410069585 :    : 0.410069585 : 0.410069585 : 1 



    //                  opticks.ana.loc.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[19] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                       .theStatus : (DsG4OpBoundaryProcessStatus) theStatus = TotalInternalReflection 
    flatExit: mrk:   crfc:   21 df:2.23175034e-10 flat:*0.443446934*  ufval:0.443446934 :          OpBoundary; : lufc : 34    
    propagate_at_boundary  u_reflect:    0.443446934  reflect:1   TransCoeff:   0.00000  c2c2:   -1.3720 tir:1  pos (   26.3642  -150.0000    98.5117)
     [ 20]                                            reflect :     0.443446934 :    : 0.443446934 : 0.443446934 : 1 

    flatExit: mrk:   crfc:   22 df:1.27960198e-10 flat:0.806965649  ufval:0.806965649 :          OpRayleigh; : lufc : 34    
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:3
    propagate_to_boundary  u_boundary_burn:0.806965649 speed:165.028061
     [ 21]                                      boundary_burn :     0.806965649 :    : 0.806965649 : 0.806965649 : 2 

    flatExit: mrk:   crfc:   23 df:3.73382547e-10 flat:0.994345605  ufval:0.994345605 :        OpAbsorption; : lufc : 34    
    propagate_to_boundary  u_scattering:0.994345605   scattering_length(s.material1.z):1000000 scattering_distance:5670.44141
     [ 22]                                         scattering :     0.994345605 :    : 0.994345605 : 0.994345605 : 1 

    2017-12-15 11:01:17.063 INFO  [644860] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-15 11:01:17.063 ERROR [644860] [CRandomEngine::poststep@236] CRandomEngine::poststep _noZeroSteps 1 backseq -3

    flatExit: mrk:** crfc:   24 df:0.446355462 flat:*0.443446934*  ufval:0.889802396 :          OpBoundary; : lufc : 34    
    propagate_to_boundary  u_absorption:0.889802396   absorption_length(s.material1.y):1000000 absorption_distance:116755.867
     [ 23]                                         absorption :     0.889802396 :    : 0.889802396 : 0.889802396 : 1 

    Process 26523 stopped
    * thread #1: tid = 0x9d6fc, 0x00000001044e063a libcfg4.dylib`CRandomEngine::flat(this=0x000000010fc04b20) + 1082 at CRandomEngine.cc:206, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x00000001044e063a libcfg4.dylib`CRandomEngine::flat(this=0x000000010fc04b20) + 1082 at CRandomEngine.cc:206
       203      //if(m_alignlevel > 1 || m_ctx._print) dumpFlat() ; 
       204      m_current_record_flat_count++ ; 
       205      m_current_step_flat_count++ ; 
    -> 206      return m_flat ;   // (*lldb*) flatExit
       207  }
       208  
       209  




::


    tboolean-;tboolean-box --okg4 --align --mask 9041 --pindex 0 --pindexlog -DD 


    .[ 21]                                      boundary_burn :    *0.885444343*:    : 0.885444343 : 0.885444343 : 2 
     [ 22]                                         scattering :     0.554676592 :    : 0.554676592 : 0.554676592 : 1 
     [ 23]                                         absorption :     0.302562296 :    : 0.302562296 : 0.302562296 : 1  still together
     [ 24]                                            reflect :    *0.530730784* :    : 0.530730784 : 0.530730784 : 1 
     [ 25]                                      boundary_burn :      0.68599081 :    : 0.685990810 : 0.685990810 : 2 
     [ 26]                                         scattering :     0.601776481 :    : 0.601776481 : 0.601776481 : 1 
     [ 27]                                         absorption :     0.215921149 :    : 0.215921149 : 0.215921149 : 1 


     [ 20]                                            reflect :     0.921632886 :    : 0.921632886 : 0.921632886 : 1 


    //                    opticks.ana.loc.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[25] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                      /TransCoeff :  0.901669  
    //                                                                              /_u :  0.921633  
    //                                                                       /_transmit : False 

    //                  opticks.ana.loc.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[19] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (type-error type-error type-error)  
    //                                                                     .NewMomentum :  (type-error type-error type-error)  
    //                                                                       .theStatus : (DsG4OpBoundaryProcessStatus) theStatus = FresnelReflection 
    flatExit: mrk:   crfc:   22 df:9.0057406e-11 flat:*0.885444343*  ufval:0.885444343 :          OpBoundary; : lufc : 42    
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:3
    propagate_to_boundary  u_boundary_burn:0.885444343 speed:165.028061
     [ 21]                                      boundary_burn :     0.885444343 :    : 0.885444343 : 0.885444343 : 2 

    flatExit: mrk:   crfc:   23 df:3.50006135e-10 flat:0.554676592  ufval:0.554676592 :          OpRayleigh; : lufc : 42    
    propagate_to_boundary  u_scattering:0.554676592   scattering_length(s.material1.z):1000000 scattering_distance:589370.062
     [ 22]                                         scattering :     0.554676592 :    : 0.554676592 : 0.554676592 : 1 

    flatExit: mrk:   crfc:   24 df:3.90533439e-10 flat:0.302562296  ufval:0.302562296 :        OpAbsorption; : lufc : 42    
    propagate_to_boundary  u_absorption:0.302562296   absorption_length(s.material1.y):1000000 absorption_distance:1195468.12
     [ 23]                                         absorption :     0.302562296 :    : 0.302562296 : 0.302562296 : 1 

    2017-12-15 10:46:01.548 INFO  [639881] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-15 10:46:01.548 ERROR [639881] [CRandomEngine::poststep@236] CRandomEngine::poststep _noZeroSteps 1 backseq -3

               LOOKS LIKE AN UN-NEEDED -3 REWIND CAUSES THE MIS-ALIGN, 

               HMM SOME ZERO STEPS DONT NEED REWIND ?

               PERHAPS A ZERO STEP FOLLOWING A STEP IN WHICH THE BOUNDARY PROCESS WINS SHOULD NOT REWIND ?
               

    flatExit: mrk:** crfc:   25 df:0.354713559 flat:*0.885444343*  ufval:0.530730784 :          OpBoundary; : lufc : 42    
    propagate_at_boundary  u_reflect:    0.530730784  reflect:1   TransCoeff:   0.00000  c2c2:   -1.4179 tir:1  pos (   54.0247    85.2057  -100.0000)
     [ 24]                                            reflect :     0.530730784 :    : 0.530730784 : 0.530730784 : 1 

    Process 25885 stopped
    * thread #1: tid = 0x9c389, 0x00000001044e06da libcfg4.dylib`CRandomEngine::flat(this=0x0000000110856110) + 1082 at CRandomEngine.cc:206, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x00000001044e06da libcfg4.dylib`CRandomEngine::flat(this=0x0000000110856110) + 1082 at CRandomEngine.cc:206
       203      //if(m_alignlevel > 1 || m_ctx._print) dumpFlat() ; 
       204      m_current_record_flat_count++ ; 
       205      m_current_step_flat_count++ ; 
    -> 206      return m_flat ;   // (*lldb*) flatExit
       207  }





Debugging Idea
----------------

* common logging format for both simulations, so can just diff it 


Auto-interleave ?
-------------------

Redirect OptiX/CUDA logging to file ?
---------------------------------------

* https://stackoverflow.com/questions/21238303/redirecting-cuda-printf-to-a-c-stream

::

    simon:opticks blyth$ opticks-find rdbuf
    ./openmeshrap/MTool.cc:         cout_redirect out_(coutbuf.rdbuf());
    ./openmeshrap/MTool.cc:         cerr_redirect err_(cerrbuf.rdbuf()); 
    ./boostrap/BDirect.hh:        : old( std::cout.rdbuf( new_buffer ) ) 
    ./boostrap/BDirect.hh:        std::cout.rdbuf( old );
    ./boostrap/BDirect.hh:        : old( std::cerr.rdbuf( new_buffer ) ) 
    ./boostrap/BDirect.hh:        std::cerr.rdbuf( old );
    simon:opticks blyth$ 





First look at the 6 maligned
--------------------------------


::

    In [1]: ab.maligned
    Out[1]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [2]: ab.dumpline(ab.maligned)
          0   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          1   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          2  14510 :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          3  49786 :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
          4  69653 :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
          5  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 


::

    In [20]: ab.dumpline(range(1220,1240))
          0   1220 :                                        TO BT BT SA                                        TO BT BT SA 
          1   1221 :                                        TO BT BT SA                                        TO BT BT SA 
          2   1222 :                                        TO BT BT SA                                        TO BT BT SA 
          3   1223 :                                        TO BT BT SA                                        TO BT BT SA 
          4   1224 :                                        TO BT BT SA                                        TO BT BT SA 
          5   1225 :                                        TO BT BT SA                                        TO BT BT SA 
          6   1226 :                                        TO BT BT SA                                        TO BT BT SA 
          7   1227 :                                        TO BT BT SA                                        TO BT BT SA 
          8   1228 :                                        TO BT BT SA                                        TO BT BT SA 
          9   1229 :                                        TO BT BT SA                                        TO BT BT SA 
         10   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
         11   1231 :                                        TO BT BT SA                                        TO BT BT SA 
         12   1232 :                                        TO BT BT SA                                        TO BT BT SA 
         13   1233 :                                        TO BT BT SA                                        TO BT BT SA 
         14   1234 :                                        TO BT BT SA                                        TO BT BT SA 
         15   1235 :                                        TO BT BT SA                                        TO BT BT SA 
         16   1236 :                                        TO BT BT SA                                        TO BT BT SA 
         17   1237 :                                        TO BT BT SA                                        TO BT BT SA 
         18   1238 :                                        TO BT BT SA                                        TO BT BT SA 
         19   1239 :                                           TO BR SA                                           TO BR SA 




1230 : could be reflectivity edge

::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -DD   




::

    In [9]: ab.recline([1230,1230])
    Out[9]: '   1230   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA '


    In [18]: a.rpolw_(slice(0,8))[1230]
    Out[18]: 
    A()sliced
    A([    [ 0.    , -1.    ,  0.    , -0.1575],    TO
           [ 0.    ,  1.    ,  0.    , -0.1575],    BR
           [-0.1969, -0.9528, -0.2283, -0.1575],    SC
           [-0.685 , -0.7165,  0.1417, -0.1575],    BT
           [-0.685 ,  0.7165, -0.1417, -0.1575],    BR
           [-0.1732,  0.9528,  0.252 , -0.1575],
           [-0.1732,  0.9528,  0.252 , -0.1575],
           [-1.    , -1.    , -1.    , -1.    ]], dtype=float32)

    In [19]: b.rpolw_(slice(0,8))[1230]
    Out[19]: 
    A()sliced
    A([    [ 0.    , -1.    ,  0.    , -0.1575],   TO
           [ 0.    ,  1.    ,  0.    , -0.1575],   BR
           [-0.1969, -0.9528, -0.2283, -0.1575],   SC
           [-0.685 , -0.7165,  0.1417, -0.1575],   BT
           [-0.685 ,  0.7165, -0.1417, -0.1575],   BR
           [-0.315 ,  0.9449, -0.0551, -0.1575],
           [-0.3307,  0.937 , -0.1024, -0.1575],
           [-0.3307,  0.937 , -0.1024, -0.1575]], dtype=float32)





Maligned Six
---------------

::

    In [1]: ab.maligned
    Out[1]: array([ 1230,  9041, 14510, 49786, 69653, 77962])

    In [2]: ab.dumpline(ab.maligned)
          0   1230 :                               TO BR SC BT BR BT SA                            TO BR SC BT BR BR BT SA 
          1   9041 :                         TO BT SC BR BR BR BR BT SA                               TO BT SC BR BR BT SA 
          2  14510 :                               TO SC BT BR BR BT SA                                  TO SC BT BR BT SA 
          3  49786 :                         TO BT BT SC BT BR BR BT SA                            TO BT BT SC BT BR BT SA 
          4  69653 :                               TO BT SC BR BR BT SA                                  TO BT SC BR BT SA 
          5  77962 :                               TO BT BR SC BR BT SA                            TO BT BR SC BR BR BT SA 



Manually interleaving RNG consumption logging for 1230.

::

    tboolean-;tboolean-box --okg4 --align --mask 1230 --pindex 0 -DD    



    2017-12-12 19:03:34.161 INFO  [146287] [CInputPhotonSource::GeneratePrimaryVertex@163] CInputPhotonSource::GeneratePrimaryVertex n 1
    2017-12-12 19:03:34.161 ERROR [146287] [CRandomEngine::pretrack@258] CRandomEngine::pretrack record_id:  ctx.record_id 0 index 1230 mask.size 1
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[0] :    1   1  :  0.00111702  :  OpBoundary;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[1] :    2   2  :  0.502647  :  OpRayleigh;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[2] :    3   3  :  0.601504  :  OpAbsorption;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[3] :    4   4  :  0.938713  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   

    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[0] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (  -0.000   -0.000    1.000)  
    //                                                                     .NewMomentum :  (   0.000    0.000    0.000)  
    //                                                                      /TransCoeff :  0.938471  
    //                                                                              /_u :  0.938713  
    //                                                                       /_transmit : False 
    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[0] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (   0.000    0.000   -1.000)  
    //                                                                     .NewMomentum :  (   0.000    0.000   -1.000)  


    2017-12-12 19:03:35.820 ERROR [146287] [OPropagator::launch@183] LAUNCH NOW
    generate photon_id 0 
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_boundary_burn:  0.00111702492 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.5026473403   scattering_length(s.material1.z):        1000000 scattering_distance:    687866.4375 
    propagate_to_boundary  u_absorption:   0.6015041471   absorption_length(s.material1.y):       10000000 absorption_distance:      5083218.5 
    propagate_at_boundary  u_reflect:       0.93871  reflect:1   TransCoeff:   0.93847 






    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[4] :    5   1  :  0.753801  :  OpBoundary;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[5] :    6   2  :  0.999847  :  OpRayleigh;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[6] :    7   3  :  0.43802  :  OpAbsorption;   

    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:1 
    propagate_to_boundary  u_boundary_burn:    0.753801465 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.9998467565   scattering_length(s.material1.z):        1000000 scattering_distance:    153.2552795 
    propagate_to_boundary  u_absorption:   0.4380195737   absorption_length(s.material1.y):       10000000 absorption_distance:        8254917 



    2017-12-12 19:03:34.663 INFO  [146287] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-12 19:03:34.663 ERROR [146287] [CRandomEngine::poststep@230] CRandomEngine::poststep _noZeroSteps 1 backseq -3
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[7] :    8   1  :  0.753801  :  OpBoundary;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[8] :    9   2  :  0.999847  :  OpRayleigh;   
    //                              opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[9] :   10   3  :  0.43802  :  OpAbsorption;   


    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[10] :   11   4  :  0.714032  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[11] :   12   5  :  0.330404  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[12] :   13   6  :  0.570742  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[13] :   14   7  :  0.375909  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[14] :   15   8  :  0.784978  :  OpRayleigh;   

    rayleigh_scatter_align p.direction (0 0 -1) 
    rayleigh_scatter_align p.polarization (-0 1 -0) 
    rayleigh_scatter_align.do u0:0.714032 u1:0.330404 u2:0.570742 u3:0.375909 u4:0.784978 
    rayleigh_scatter_align.do constant        (0.301043) 
    rayleigh_scatter_align.do newDirection    (0.632086 -0.301043 0.714032) 
    rayleigh_scatter_align.do newPolarization (-0.199541 -0.953611 -0.225411) 
    rayleigh_scatter_align.do doCosTheta -0.953611 doCosTheta2 0.909373   looping 0   


    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[15] :   16   1  :  0.892654  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[16] :   17   2  :  0.441063  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[17] :   18   3  :  0.773742  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[18] :   19   4  :  0.556839  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   


    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:2 
    propagate_to_boundary  u_boundary_burn:   0.8926543593 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.4410631955   scattering_length(s.material1.z):        1000000 scattering_distance:     818567.125 
    propagate_to_boundary  u_absorption:   0.7737424374   absorption_length(s.material1.y):       10000000 absorption_distance:     2565162.25 
    propagate_at_boundary  u_reflect:       0.55684  reflect:0   TransCoeff:   0.88430 


    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[1] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (   0.632   -0.301    0.714)  
    //                                                                     .NewMomentum :  (   0.000    0.000   -1.000)  
    //                                                                      /TransCoeff :  0.884304  
    //                                                                              /_u :  0.556839  
    //                                                                       /_transmit : True 

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[1] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (   0.381   -0.181    0.907)  
    //                                                                     .NewMomentum :  (   0.381   -0.181    0.907)  







    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[19] :   20   1  :  0.775349  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[20] :   21   2  :  0.752141  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[21] :   22   3  :  0.412002  :  OpAbsorption;   



    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:3 
    propagate_to_boundary  u_boundary_burn:    0.775349319 speed:    165.0280609 
    propagate_to_boundary  u_scattering:   0.7521412373   scattering_length(s.material1.z):        1000000 scattering_distance:    284831.1562 
    propagate_to_boundary  u_absorption:   0.4120023847   absorption_length(s.material1.y):        1000000 absorption_distance:     886726.125 
    propagate_at_boundary  u_reflect:       0.28246  reflect:1   TransCoeff:   0.00000  c2c2:   -1.3552 tir:1  pos (  150.0000   -77.6576    24.3052)   
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ WHATS THIS ??? DOES TIR CONSUME DIFFERENT ?



    In [7]: a.rpost_(slice(0,8))[1230]
    Out[7]: 
    A()sliced
    A([    [ -37.8781,   11.8231, -449.8989,    0.2002],    TO 
           [ -37.8781,   11.8231,  -99.9944,    1.3672],    BR   0
           [ -37.8781,   11.8231, -253.2548,    1.8781],    SC   1
           [  97.7921,  -52.7844,  -99.9944,    2.5941],    BT   2

           [ 149.9984,  -77.6556,   24.307 ,    3.4248],    BR   3   (point before was TIR)

           [ 118.2039,  -92.7959,   99.9944,    3.9308],   *BT*      << OK/G4 BT/BR
           [-191.6203, -240.3581,  449.9952,    5.566 ],   *SA*
           [   0.    ,    0.    ,    0.    ,    0.    ]])


    In [8]: b.rpost_(slice(0,8))[1230]
    Out[8]: 
    A()sliced
    A([    [ -37.8781,   11.8231, -449.8989,    0.2002],   TO
           [ -37.8781,   11.8231,  -99.9944,    1.3672],   BR 
           [ -37.8781,   11.8231, -253.2548,    1.8781],   SC
           [  97.7921,  -52.7844,  -99.9944,    2.5941],   BT
           [ 149.9984,  -77.6556,   24.307 ,    3.4248],   BR
           [ 118.2039,  -92.7959,   99.9944,    3.9308],  *BR* 
           [  34.2032, -132.8074,  -99.9944,    5.2675],  *BT*
           [-275.6348, -280.3696, -449.9952,    6.9027]]) *SA* 







    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:4 
    propagate_to_boundary  u_boundary_burn:   0.4324976802 speed:    165.0280609 
    propagate_to_boundary  u_scattering:   0.9078488946   scattering_length(s.material1.z):        1000000 scattering_distance:    96677.32812 
    propagate_to_boundary  u_absorption:   0.9121392369   absorption_length(s.material1.y):        1000000 absorption_distance:      91962.625 
    propagate_at_boundary  u_reflect:       0.20181  reflect:0   TransCoeff:   0.88556  c2c2:    0.5098 tir:0  pos (  118.2061   -92.8001   100.0000)   
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:5 
    propagate_to_boundary  u_boundary_burn:   0.7953493595 speed:      299.79245 
    propagate_to_boundary  u_scattering:   0.4842039943   scattering_length(s.material1.z):        1000000 scattering_distance:         725249 
    propagate_to_boundary  u_absorption:  0.09354860336   absorption_length(s.material1.y):       10000000 absorption_distance:       23692742 
    propagate_at_surface   u_surface:       0.7505 
    propagate_at_surface   u_surface_burn:       0.9462 
    2017-12-12 19:32:41.223 ERROR [157506] [OPropagator::launch@185] LAUNCH DONE




















    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[2] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.381   -0.181    0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181    0.907)  
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[22] :   23   1  :  0.282463  :  OpBoundary;    <<< off-by-1
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[23] :   24   2  :  0.432498  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[24] :   25   3  :  0.907849  :  OpAbsorption;   

    2017-12-12 19:03:34.795 INFO  [146287] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-12 19:03:34.795 ERROR [146287] [CRandomEngine::poststep@230] CRandomEngine::poststep _noZeroSteps 1 backseq -3

    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[25] :   26   1  :  0.282463  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[26] :   27   2  :  0.432498  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[27] :   28   3  :  0.907849  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[28] :   29   4  :  0.912139  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   

    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[2] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (  -0.381   -0.181    0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181    0.907)  
    //                                                                      /TransCoeff :  0.885559  
    //                                                                              /_u :  0.912139  
    //                                                                       /_transmit : False 

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[3] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.381   -0.181   -0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181   -0.907)  
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[29] :   30   1  :  0.201809  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[30] :   31   2  :  0.795349  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[31] :   32   3  :  0.484204  :  OpAbsorption;   
    2017-12-12 19:03:34.855 INFO  [146287] [CSteppingAction::setStep@132]  noZeroSteps 1 severity 0 ctx  record_id 0 event_id 0 track_id 0 photon_id 0 parent_id -1 primary_id -2 reemtrack 0
    2017-12-12 19:03:34.855 ERROR [146287] [CRandomEngine::poststep@230] CRandomEngine::poststep _noZeroSteps 1 backseq -3
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[32] :   33   1  :  0.201809  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[33] :   34   2  :  0.795349  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[34] :   35   3  :  0.484204  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[35] :   36   4  :  0.0935486  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025   

    //                opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_DiDiTransCoeff_.[3] : DiDiTransCoeff 
    //                                                                             this : DsG4OpBoundaryProcess_cc_DiDiTransCoeff 
    //                                                                     .OldMomentum :  (  -0.381   -0.181   -0.907)  
    //                                                                     .NewMomentum :  (  -0.381   -0.181   -0.907)  
    //                                                                      /TransCoeff :  0.874921  
    //                                                                              /_u :  0.0935486  
    //                                                                       /_transmit : True 

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[4] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.632   -0.301   -0.714)  
    //                                                                     .NewMomentum :  (  -0.632   -0.301   -0.714)  
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[36] :   37   1  :  0.750533  :  OpBoundary;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[37] :   38   2  :  0.946246  :  OpRayleigh;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[38] :   39   3  :  0.357591  :  OpAbsorption;   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[39] :   40   4  :  0.166174  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655   
    //                             opticks.ana.cfg4lldb.CRandomEngine_cc_flatExit_.[40] :   41   5  :  0.628917  :  OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1242   

    //              opticks.ana.cfg4lldb.DsG4OpBoundaryProcess_cc_ExitPostStepDoIt_.[5] : ExitPostStepDoIt 
    //                                                                             this : DsG4OpBoundaryProcess_cc_ExitPostStepDoIt 
    //                                                                     .OldMomentum :  (  -0.632   -0.301   -0.714)  
    //                                                                     .NewMomentum :  (  -0.632   -0.301   -0.714)  
    2017-12-12 19:03:34.926 INFO  [146287] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1

