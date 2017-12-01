photon-polarization-testauto-SR
==================================


TODO : extend --reflectcheat to SR ?
-----------------------------------------


DONE : non-normal incidence testing of SR via emitconfig.diffuse 
-------------------------------------------------------------------------------

* implemented using NRngDiffuse


::

    simon:opticksdata blyth$ tboolean-;tboolean-box-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    [2017-12-01 18:21:16,510] p90506 {/Users/blyth/opticks/ana/tboolean.py:27} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython False 
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171201-1753 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171201-1753 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/1 =  0.00  (pval:1.000 prob:0.000)  
    0000               8d    390951    390951             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] TO SA
    0001              8ad    209049    209049             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] TO SR SA
    .                             600000    600000         0.00/1 =  0.00  (pval:1.000 prob:0.000)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/1 =  0.00  (pval:1.000 prob:0.000)  
    0000             1080    390951    390951             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] TO|SA
    0001             1280    209049    209049             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] TO|SR|SA
    .                             600000    600000         0.00/1 =  0.00  (pval:1.000 prob:0.000)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/1 =  0.00  (pval:1.000 prob:0.000)  
    0000               12    390951    390951             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] Vm Rk
    0001              122    209049    209049             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] Vm Vm Rk
    .                             600000    600000         0.00/1 =  0.00  (pval:1.000 prob:0.000)  
    ab.a.metadata                  /tmp/blyth/opticks/evt/tboolean-box/torch/1 e0e532a9912d085f0cc73ddb9c6177df 08caf4a1cccdbf2f340247097a1fa206  600000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0 {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.013763847773702764, 0.013763847773702764] 
     0000            :                          TO SA :  390951   390951  :    390951 3127608/   1460: 0.000  mx/mn/av 0.01376/     0/6.164e-06  eps:0.0002    
     0001            :                       TO SR SA :  209049   209049  :    209049 2508588/    392: 0.000  mx/mn/av 0.01376/     0/1.841e-06  eps:0.0002    
    rpol_dv maxdvmax:0.00787401199341 maxdv:[0.007874011993408203, 0.0] 
     0000            :                          TO SA :  390951   390951  :    390951 2345706/      2: 0.000  mx/mn/av 0.007874/     0/6.714e-09  eps:0.0002    
     0001            :                       TO SR SA :  209049   209049  :    209049 1881441/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    ox_dv maxdvmax:0.000152587890625 maxdv:[0.000152587890625, 9.1552734375e-05] 
     0000            :                          TO SA :  390951   390951  :    390951 6255216/      0: 0.000  mx/mn/av 0.0001526/     0/2.651e-06  eps:0.0002    
     0001            :                       TO SR SA :  209049   209049  :    209049 3344784/      0: 0.000  mx/mn/av 9.155e-05/     0/1.408e-06  eps:0.0002    
    c2p : {'seqmat_ana': 0.0, 'pflags_ana': 0.0, 'seqhis_ana': 0.0} c2pmax: 0.0  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.007874011993408203, 'rpost_dv': 0.013763847773702764} rmxs_max_: 0.0137638477737  CUT ok.rdvmax 0.1  RC:0 
    pmxs_ : {'ox_dv': 0.000152587890625} pmxs_max_: 0.000152587890625  CUT ok.pdvmax 0.001  RC:0 
    [2017-12-01 18:21:18,797] p90506 {/Users/blyth/opticks/ana/tboolean.py:43} INFO - early exit as non-interactive
    simon:opticksdata blyth$ 
    simon:opticksdata blyth$ 



FIXED : polz sign flip at specular reflection normal incidence
----------------------------------------------------------------

Move to::


    458 __device__ void propagate_at_specular_reflector_geant4_style(Photon &p, State &s, curandState &rng)
    459 {
    460     // NB no-s-pol throwing 
    461 
    462     const float c1 = -dot(p.direction, s.surface_normal );      // G4double PdotN = OldMomentum * theFacetNormal;
    463 
    464     float normal_coefficient = dot(p.polarization, s.surface_normal);    // G4double EdotN = OldPolarization * theFacetNormal;
    465     // EdotN : fraction of E vector perpendicular to plane of incidence, ie S polarization
    466 
    467     p.direction += 2.0f*c1*s.surface_normal  ;
    468 
    469     p.polarization = -p.polarization + 2.f*normal_coefficient*s.surface_normal  ;  // NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
    470 
    471     p.flags.i.x = 0 ;  // no-boundary-yet for new direction
    472 }
    473 

Gets polz to match, at normal incidence anyhow::


    [2017-12-01 15:01:53,546] p61223 {/Users/blyth/opticks/ana/ab.py:156} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171201-1501 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171201-1501 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000              8ad    600000    600000             0.00        1.000 +- 0.001        1.000 +- 0.001  [3 ] TO SR SA
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000             1280    600000    600000             0.00        1.000 +- 0.001        1.000 +- 0.001  [3 ] TO|SR|SA
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000              122    600000    600000             0.00        1.000 +- 0.001        1.000 +- 0.001  [3 ] Vm Vm Rk
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    ab.a.metadata                  /tmp/blyth/opticks/evt/tboolean-box/torch/1 2722694edd3a8a19f6dd2915b66ce147 600b943ab3855243ca6e162794591dd7  600000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0 {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.013763847773677895] 
     0000            :                       TO SR SA :  600000   600000  :    600000 7200000/     21: 0.000  mx/mn/av 0.01376/     0/4.014e-08  eps:0.0002    
    rpol_dv maxdvmax:0.0 maxdv:[0.0] 
     0000            :                       TO SR SA :  600000   600000  :    600000 5400000/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    ox_dv maxdvmax:1.40129846432e-45 maxdv:[1.401298464324817e-45] 
     0000            :                       TO SR SA :  600000   600000  :    600000 9600000/      0: 0.000  mx/mn/av 1.401e-45/     0/8.758e-47  eps:0.0002    
    c2p : {'seqmat_ana': 0.0, 'pflags_ana': 0.0, 'seqhis_ana': 0.0} c2pmax: 0.0  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.0, 'rpost_dv': 0.013763847773677895} rmxs_max_: 0.0137638477737  CUT ok.rdvmax 0.1  RC:0 
    pmxs_ : {'ox_dv': 1.401298464324817e-45} pmxs_max_: 1.40129846432e-45  CUT ok.pdvmax 0.001  RC:0 
    [2017-12-01 15:01:55,250] p61223 {/Users/blyth/opticks/ana/tboolean.py:43} INFO - early exit as non-interactive
    2017-12-01 15:01:55.360 INFO  [866285] [SSys::run@46] tboolean.py --tag 1 --tagoffset 0 --det tboolean-box --src torch   rc_raw : 0 rc : 0
    2017-12-01 15:01:55.361 INFO  [866285] [OpticksAna::run@79] OpticksAna::run anakey tboolean cmdline tboolean.py --tag 1 --tagoffset 0 --det tboolean-box --src torch   rc 0 rcmsg -
    2017-12-01 15:01:55.361 INFO  [866285] [SSys::WaitForInput@145] SSys::WaitForInput OpticksAna::run paused : hit RETURN to continue...






::

   tboolean-;tboolean-box --okg4 --testauto --noab --nosc -D


    (lldb) b DsG4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&) 
    Breakpoint 1: where = libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&) + 39 at DsG4OpBoundaryProcess.cc:174, address = 0x00000001043545e7
    (lldb) 


    (lldb) c
    Process 59698 resuming
    Process 59698 stopped
    * thread #1: tid = 0xcfb3d, 0x0000000104354760 libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x000000010cf8e170, aTrack=0x000000011caf0e20, aStep=0x000000010cf0be10) + 416 at DsG4OpBoundaryProcess.cc:248, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x0000000104354760 libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x000000010cf8e170, aTrack=0x000000011caf0e20, aStep=0x000000010cf0be10) + 416 at DsG4OpBoundaryProcess.cc:248
       245      Material1 = pPreStepPoint  -> GetMaterial();
       246      Material2 = pPostStepPoint -> GetMaterial();
       247  
    -> 248      const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
       249  
       250      thePhotonMomentum = aParticle->GetTotalMomentum();
       251      OldMomentum       = aParticle->GetMomentumDirection();
    (lldb) p Material1
    (G4Material *) $0 = 0x000000010cf40ad0
    (lldb) p *Material1
    (G4Material) $1 = {
      fName = (std::__1::string = "Vacuum")
      fChemicalFormula = (std::__1::string = "")
      fDensity = 0.00000062415096471204161
      fState = kStateGas
      fTemp = 293.14999999999998
      fPressure = 632420964.9944762
      maxNbComponents = 1
      fArrayLength = 1
      fNumberOfComponents = 1
      fNumberOfElements = 1
      theElementVector = 0x000000010cf41020 size=1
      fMassFractionVector = 0x000000010cf40470
      fAtomsVector = 0x0000000000000000
      fMaterialPropertiesTable = 0x000000010cf43670
      fIndexInTable = 1
      VecNbOfAtomsPerVolume = 0x000000010cf40d50
      TotNbOfAtomsPerVolume = 0.000059625166237623757
      TotNbOfElectPerVolume = 0.000059625166237623757
      fRadlen = 6.3172309490184856E+27
      fNuclInterLen = 3.500000003326212E+27
      fIonisation = 0x000000010cf43320
      fSandiaTable = 0x000000010cf40d80
      fBaseMaterial = 0x0000000000000000
      fMassOfMolecule = 0.010467911522873029
      fMatComponents = size=0 {}
    }
    (lldb) p *Material2
    (G4Material) $2 = {}
    (lldb) 


    (lldb) p Surface
    (G4LogicalSurface *) $6 = 0x000000010cf48720
    (lldb) p *Surface
    (G4LogicalSurface) $7 = {
      theName = (std::__1::string = "perfectSpecularSurface")
      theSurfaceProperty = 0x000000010cf48c70
      theTransRadSurface = 0x0000000000000000
    }
    (lldb) 


    (lldb) c
    Process 59698 resuming
    Process 59698 stopped
    * thread #1: tid = 0xcfb3d, 0x00000001043551af libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x000000010cf8e170, aTrack=0x000000011caf0e20, aStep=0x000000010cf0be10) + 3055 at DsG4OpBoundaryProcess.cc:367, queue = 'com.apple.main-thread', stop reason = breakpoint 5.1
        frame #0: 0x00000001043551af libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x000000010cf8e170, aTrack=0x000000011caf0e20, aStep=0x000000010cf0be10) + 3055 at DsG4OpBoundaryProcess.cc:367
       364  
       365      if (Surface) OpticalSurface = dynamic_cast <G4OpticalSurface*> (Surface->GetSurfaceProperty());
       366  
    -> 367      if (OpticalSurface) 
       368      {
       369  #ifdef SCB_BND_DEBUG
       370            if(m_dbg || m_other)
    (lldb) p OpticalSurface
    (G4OpticalSurface *) $8 = 0x000000010cf48c70
    (lldb) p *OpticalSurface
    (G4OpticalSurface) $9 = {
      G4SurfaceProperty = {
        theName = (std::__1::string = "perfectSpecularSurface")
        theType = dielectric_dielectric
      }
      theModel = unified
      theFinish = polishedfrontpainted
      sigma_alpha = 0
      polish = 1
      theMaterialPropertiesTable = 0x000000010cf48120
      AngularDistribution = 0x0000000000000000
      DichroicVector = 0x0000000000000000
    }
    (lldb) 


SR reflectivity fork happens here::

    (lldb) c
    Process 59698 resuming
    Process 59698 stopped
    * thread #1: tid = 0xcfb3d, 0x0000000104356210 libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x000000010cf8e170, aTrack=0x000000011caf0e20, aStep=0x000000010cf0be10) + 7248 at DsG4OpBoundaryProcess.cc:650, queue = 'com.apple.main-thread', stop reason = breakpoint 12.1
        frame #0: 0x0000000104356210 libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x000000010cf8e170, aTrack=0x000000011caf0e20, aStep=0x000000010cf0be10) + 7248 at DsG4OpBoundaryProcess.cc:650
       647          {
       648              if ( theFinish == polishedfrontpainted || theFinish == groundfrontpainted ) 
       649              {
    -> 650                  if( !G4BooleanRand(theReflectivity) ) 
       651                  {
       652                      DoAbsorption();
       653                  }
    (lldb) 


     646         else if (type == dielectric_dielectric)
     647         {
     648             if ( theFinish == polishedfrontpainted || theFinish == groundfrontpainted )
     649             {
     650                 if( !G4BooleanRand(theReflectivity) )
     651                 {
     652                     DoAbsorption();
     653                 }
     654                 else
     655                 {
     656                     if ( theFinish == groundfrontpainted ) theStatus = LambertianReflection;
     657                     DoReflection();
     658                 }
     659             }
     660             else
     661             {
     662                 DielectricDielectric();
     663             }
     664         }


::

    (lldb) b DsG4OpBoundaryProcess::DoReflection()
    Breakpoint 13: where = libcfg4.dylib`DsG4OpBoundaryProcess::DoReflection() + 19 at DsG4OpBoundaryProcess.h:314, address = 0x000000010435bba3
    (lldb) 

    (lldb) c
    Process 59698 resuming
    Process 59698 stopped
    * thread #1: tid = 0xcfb3d, 0x000000010435beab libcfg4.dylib`DsG4OpBoundaryProcess::DoReflection(this=0x000000010cf8e170) + 795 at DsG4OpBoundaryProcess.h:330, queue = 'com.apple.main-thread', stop reason = breakpoint 14.1
        frame #0: 0x000000010435beab libcfg4.dylib`DsG4OpBoundaryProcess::DoReflection(this=0x000000010cf8e170) + 795 at DsG4OpBoundaryProcess.h:330
       327          }
       328          else {
       329  
    -> 330            theStatus = SpikeReflection;
       331            theFacetNormal = theGlobalNormal;
       332            G4double PdotN = OldMomentum * theFacetNormal;
       333            NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
    (lldb) p theGlobalNormal
    (G4ThreeVector) $21 = (dx = 0, dy = 0, dz = -1)
    (lldb) p OldMomentum
    (G4ThreeVector) $22 = (dx = -0, dy = -0, dz = 1)
    (lldb) 


    311 inline
    312 void DsG4OpBoundaryProcess::DoReflection()
    313 {
    314         if ( theStatus == LambertianReflection ) {
    315 
    316           NewMomentum = G4LambertianRand(theGlobalNormal);
    317           theFacetNormal = (NewMomentum - OldMomentum).unit();
    318 
    319         }
    320         else if ( theFinish == ground ) {
    321 
    322       theStatus = LobeReflection;
    323           theFacetNormal = GetFacetNormal(OldMomentum,theGlobalNormal);
    324           G4double PdotN = OldMomentum * theFacetNormal;
    325           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
    326 
    327         }
    328         else {
    329 
    330           theStatus = SpikeReflection;
    331           theFacetNormal = theGlobalNormal;
    332           G4double PdotN = OldMomentum * theFacetNormal;
    333           NewMomentum = OldMomentum - (2.*PdotN)*theFacetNormal;
    334 
    335         }
    336         G4double EdotN = OldPolarization * theFacetNormal;
    337         NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
    338 }



::

    (lldb) c
    Process 59698 resuming
    Process 59698 stopped
    * thread #1: tid = 0xcfb3d, 0x000000010435c0c7 libcfg4.dylib`DsG4OpBoundaryProcess::DoReflection(this=0x000000010cf8e170) + 1335 at DsG4OpBoundaryProcess.h:338, queue = 'com.apple.main-thread', stop reason = breakpoint 16.4
        frame #0: 0x000000010435c0c7 libcfg4.dylib`DsG4OpBoundaryProcess::DoReflection(this=0x000000010cf8e170) + 1335 at DsG4OpBoundaryProcess.h:338
       335          }
       336          G4double EdotN = OldPolarization * theFacetNormal;
       337          NewPolarization = -OldPolarization + (2.*EdotN)*theFacetNormal;
    -> 338  }
       339  
       340  #endif /* DsG4OpBoundaryProcess_h */
    (lldb) p NewPolarization
    (G4ThreeVector) $27 = (dx = 0, dy = 1, dz = -0)
    (lldb) p OldPolarization
    (G4ThreeVector) $28 = (dx = 0, dy = -1, dz = 0)
    (lldb) p EdotN
    (G4double) $29 = 0
    (lldb) p theFacetNormal
    (G4ThreeVector) $30 = (dx = 0, dy = 0, dz = -1)
    (lldb) 





FIXED : testauto giving NaN polarizaton for SR
-------------------------------------------------

Getting NaN in photon polarization for specular reflection at normal incidence.

* was due to incorrect normal incidence detection in propagate_at_specular_surface


APPROACH
~~~~~~~~~~~

Narrow autoemitconfig uv domain such that all photons will SR
and SC AB are switched off

* note that the autoemitconfig option must be given to the python geometry prep stage, 
  not the OKG4Test executable

::

     tboolean-;tboolean-box --okg4 --testauto --noab --nosc 


::

     710 tboolean-box--(){ cat << EOP 
     711 import logging
     712 log = logging.getLogger(__name__)
     713 from opticks.ana.base import opticks_main
     714 from opticks.analytic.polyconfig import PolyConfig
     715 from opticks.analytic.csg import CSG  
     716 
     717 autoemitconfig="photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x3f,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55"
     718 args = opticks_main(csgpath="$TMP/$FUNCNAME", autoemitconfig=autoemitconfig)
     719 
     720 emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75" 
     721 
     722 CSG.kwa = dict(poly="IM",resolution="20", verbosity="0",ctrl="0", containerscale="3", emitconfig=emitconfig  )
     723 
     724 container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container="1" )  # no param, container="1" switches on auto-sizing
     725 
     726 box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2" )
     727 
     728 CSG.Serialize([container, box], args )
     729 EOP
     730 }


cu/propagate.h DEBUG_POLZ::

    2017-12-01 13:22:15.641 INFO  [832957] [OPropagator::prelaunch@166] 1 : (0;10,1) prelaunch_times vali,comp,prel,lnch  0.0001 3.4463 0.1303 0.0000
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    2017-12-01 13:22:15.655 INFO  [832957] [OContext::launch@322] OContext::launch LAUNCH time: 0.01389




::

    2017-12-01 13:05:45,200] p54370 {/Users/blyth/opticks/ana/ab.py:156} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171201-1305 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171201-1305 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000              8ad    600000    600000             0.00        1.000 +- 0.001        1.000 +- 0.001  [3 ] TO SR SA
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000             1280    600000    600000             0.00        1.000 +- 0.001        1.000 +- 0.001  [3 ] TO|SR|SA
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000              122    600000    600000             0.00        1.000 +- 0.001        1.000 +- 0.001  [3 ] Vm Vm Rk
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    ab.a.metadata                  /tmp/blyth/opticks/evt/tboolean-box/torch/1 edfd1a210c3da6e4b725d3e4c2a2a59e 88d3ee8cc1674e4766a5b293d552ca26  600000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0 {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.013763847773677895] 
     0000            :                       TO SR SA :  600000   600000  :    600000 7200000/     18: 0.000  mx/mn/av 0.01376/     0/3.441e-08  eps:0.0002    
    rpol_dv maxdvmax:2.0 maxdv:[2.0] 
     0000            :                       TO SR SA :  600000   600000  :    600000 5400000/3000000: 0.556  mx/mn/av      2/     0/0.6667  eps:0.0002    
    /Users/blyth/opticks/ana/dv.py:58: RuntimeWarning: invalid value encountered in greater
      discrep = dv[dv>eps]
    ox_dv maxdvmax:nan maxdv:[nan] 
     0000            :                       TO SR SA :  600000   600000  :    600000 9600000/      0: 0.000  mx/mn/av    nan/   nan/   nan  eps:0.0002    
    c2p : {'seqmat_ana': 0.0, 'pflags_ana': 0.0, 'seqhis_ana': 0.0} c2pmax: 0.0  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 2.0, 'rpost_dv': 0.013763847773677895} rmxs_max_: 2.0  CUT ok.rdvmax 0.1  RC:88 
    pmxs_ : {'ox_dv': nan} pmxs_max_: nan  CUT ok.pdvmax 0.001  RC:88 





::

    [2017-12-01 12:35:15,285] p50967 {/Users/blyth/opticks/ana/ab.py:156} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171201-1233 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171201-1233 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         1.12/5 =  0.22  (pval:0.953 prob:0.047)  
    0000               8d    391943    391952             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] TO SA
    0001              8ad    207533    207524             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] TO SR SA
    0002              86d       368       368             0.00        1.000 +- 0.052        1.000 +- 0.052  [3 ] TO SC SA
    0003             8a6d        58        64             0.30        0.906 +- 0.119        1.103 +- 0.138  [4 ] TO SC SR SA
    0004             86ad        50        42             0.70        1.190 +- 0.168        0.840 +- 0.130  [4 ] TO SR SC SA
    0005               4d        37        34             0.13        1.088 +- 0.179        0.919 +- 0.158  [2 ] TO AB
    0006            8a6ad         6        10             0.00        0.600 +- 0.245        1.667 +- 0.527  [5 ] TO SR SC SR SA
    0007              4ad         5         6             0.00        0.833 +- 0.373        1.200 +- 0.490  [3 ] TO SR AB
    .                             600000    600000         1.12/5 =  0.22  (pval:0.953 prob:0.047)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.14/4 =  0.04  (pval:0.998 prob:0.002)  
    0000             1080    391943    391952             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] TO|SA
    0001             1280    207533    207524             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] TO|SR|SA
    0002             10a0       368       368             0.00        1.000 +- 0.052        1.000 +- 0.052  [3 ] TO|SA|SC
    0003             12a0       114       116             0.02        0.983 +- 0.092        1.018 +- 0.094  [4 ] TO|SR|SA|SC
    0004             1008        37        34             0.13        1.088 +- 0.179        0.919 +- 0.158  [2 ] TO|AB
    0005             1208         5         6             0.00        0.833 +- 0.373        1.200 +- 0.490  [3 ] TO|SR|AB
    .                             600000    600000         0.14/4 =  0.04  (pval:0.998 prob:0.002)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.15/3 =  0.05  (pval:0.986 prob:0.014)  
    0000               12    391943    391952             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] Vm Rk
    0001              122    207901    207892             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] Vm Vm Rk
    0002             1222       108       106             0.02        1.019 +- 0.098        0.981 +- 0.095  [4 ] Vm Vm Vm Rk
    0003               22        37        34             0.13        1.088 +- 0.179        0.919 +- 0.158  [2 ] Vm Vm
    0004            12222         6        10             0.00        0.600 +- 0.245        1.667 +- 0.527  [5 ] Vm Vm Vm Vm Rk
    0005              222         5         6             0.00        0.833 +- 0.373        1.200 +- 0.490  [3 ] Vm Vm Vm
    .                             600000    600000         0.15/3 =  0.05  (pval:0.986 prob:0.014)  



ISSUE : propagate_at_specular_reflector giving NaN polz
----------------------------------------------------------


cu/generate.cu::

    516 
    517         command = propagate_to_boundary( p, s, rng );
    518         if(command == BREAK)    break ;           // BULK_ABSORB
    519         if(command == CONTINUE) continue ;        // BULK_REEMIT/BULK_SCATTER
    520         // PASS : survivors will go on to pick up one of the below flags, 
    521 
    522         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    523         {
    524             command = propagate_at_surface(p, s, rng);
    525             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    526             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    527         }
    528         else
    529         {
    530             //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    531             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    532             // tacit CONTINUE
    533         }



cu/propagate.h::

    518 __device__ int
    519 propagate_at_surface(Photon &p, State &s, curandState &rng)
    520 {
    521 
    522     float u = curand_uniform(&rng);
    523 
    524     if( u < s.surface.y )   // absorb   
    525     {
    526         s.flag = SURFACE_ABSORB ;
    527         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    528         return BREAK ;
    529     }
    530     else if ( u < s.surface.y + s.surface.x )  // absorb + detect
    531     {
    532         s.flag = SURFACE_DETECT ;
    533         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    534         return BREAK ;
    535     }
    536     else if (u  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    537     {
    538         s.flag = SURFACE_DREFLECT ;
    539         propagate_at_diffuse_reflector_geant4_style(p, s, rng);
    540         return CONTINUE;
    541     }
    542     else
    543     {
    544         s.flag = SURFACE_SREFLECT ;
    545         propagate_at_specular_reflector(p, s, rng );
    546         return CONTINUE;
    547     }
    548 }
    549 



::

    413 __device__ void propagate_at_specular_reflector(Photon &p, State &s, curandState &rng)
    414 {
    415     const float c1 = -dot(p.direction, s.surface_normal );     // c1 arranged to be +ve   
    416 
    417     // TODO: make change to c1 for normal incidence detection
    418 
    419     float3 incident_plane_normal = fabs(s.cos_theta) < 1e-6f ? p.polarization : normalize(cross(p.direction, s.surface_normal)) ;
    420 
    421     float normal_coefficient = dot(p.polarization, incident_plane_normal);  // fraction of E vector perpendicular to plane of incidence, ie S polarization
    422 
    423     p.direction += 2.0f*c1*s.surface_normal  ;
    424 
    425     bool s_polarized = curand_uniform(&rng) < normal_coefficient*normal_coefficient ;
    426 
    427     p.polarization = s_polarized
    428                        ?
    429                           incident_plane_normal
    430                        :
    431                           normalize(cross(incident_plane_normal, p.direction))
    432                        ;
    433 
    434     p.flags.i.x = 0 ;  // no-boundary-yet for new direction
    435 }





All final photon polz in "TO SR SA" are NaN
---------------------------------------------

::

    simon:opticks blyth$ tboolean-;tboolean-box-ip

    In [2]: ab.aselhis = "TO SR SA"

    In [3]: ab.a.ox
    Out[3]: 
    A()sliced
    A([[[-133.4443,   -1.4124, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

    In [6]: ab.a.ox[:,2,:3]
    Out[6]: 
    A()sliced
    A([[ nan,  nan,  nan],
           [ nan,  nan,  nan],
           [ nan,  nan,  nan],
           ..., 
           [ nan,  nan,  nan],
           [ nan,  nan,  nan],
           [ nan,  nan,  nan]], dtype=float32)

    In [7]: np.isnan(ab.a.ox[:,2,:3])
    Out[7]: 
    A()sliced
    A([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           ..., 
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)

    In [8]: np.all(np.isnan(ab.a.ox[:,2,:3]))
    Out[8]: 
    A()sliced
    A(True, dtype=bool)




Point-by-point pol are unset beyond first point::

    In [4]: ab.a.rpol()
    Out[4]: 
    A()sliced
    A([[[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],






Confirmed that NaN polz issue is specific to testauto/SR
------------------------------------------------------------

::

    simon:opticks blyth$ tboolean-;tboolean-box --okg4 
    ...

    .                             100000    100000         1.61/4 =  0.40  (pval:0.807 prob:0.193)  
    ab.a.metadata                  /tmp/blyth/opticks/evt/tboolean-box/torch/1 8210ebdae5967a9ef905291542364a4b 54be6772c3093360d09fefc4346e74a0  100000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0 {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.0, 0.013763847773674343, 0.0, 0.0, 0.0] 
     0000            :                          TO SA :   55321    55303  :     55249  441992/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :                    TO BT BT SA :   39222    39231  :     34492  551872/      8: 0.000  mx/mn/av 0.01376/     0/1.995e-07  eps:0.0002    
     0002            :                       TO BR SA :    2768     2814  :       188    2256/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :                 TO BT BR BT SA :    2425     2369  :       125    2500/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :              TO BT BR BR BT SA :     151      142  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    rpol_dv maxdvmax:0.0 maxdv:[0.0, 0.0, 0.0, 0.0, 0.0] 
     0000            :                          TO SA :   55321    55303  :     55249  331494/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :                    TO BT BT SA :   39222    39231  :     34492  413904/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                       TO BR SA :    2768     2814  :       188    1692/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :                 TO BT BR BT SA :    2425     2369  :       125    1875/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :              TO BT BR BR BT SA :     151      142  :         1      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    ox_dv maxdvmax:3.0517578125e-05 maxdv:[3.0517578125e-05, 5.960464477539063e-08, 1.401298464324817e-45, 5.960464477539063e-08, 5.960464477539063e-08] 
     0000            :                          TO SA :   55321    55303  :     55249  883984/      0: 0.000  mx/mn/av 3.052e-05/     0/1.907e-06  eps:0.0002    
     0001            :                    TO BT BT SA :   39222    39231  :     34492  551872/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0002            :                       TO BR SA :    2768     2814  :       188    3008/      0: 0.000  mx/mn/av 1.401e-45/     0/8.758e-47  eps:0.0002    
     0003            :                 TO BT BR BT SA :    2425     2369  :       125    2000/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0004            :              TO BT BR BR BT SA :     151      142  :         1      16/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
    c2p : {'seqmat_ana': 0.40311601124980434, 'pflags_ana': 1.0829369776001112, 'seqhis_ana': 0.88772768790641765} c2pmax: 1.0829369776  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.0, 'rpost_dv': 0.013763847773674343} rmxs_max_: 0.0137638477737  CUT ok.rdvmax 0.1  RC:0 
    pmxs_ : {'ox_dv': 3.0517578125e-05} pmxs_max_: 3.0517578125e-05  CUT ok.pdvmax 0.001  RC:0 
    [2017-12-01 12:27:18,399] p49848 {/Users/blyth/opticks/ana/tboolean.py:43} INFO - early exit as non-interactive




Saving into photon buffer
--------------------------


     71 __device__ void psave( Photon& p, optix::buffer<float4>& pbuffer, unsigned int photon_offset)
     72 {
     73     pbuffer[photon_offset+0] = make_float4( p.position.x,    p.position.y,    p.position.z,     p.time );
     74     pbuffer[photon_offset+1] = make_float4( p.direction.x,   p.direction.y,   p.direction.z,    p.weight );
     75     pbuffer[photon_offset+2] = make_float4( p.polarization.x,p.polarization.y,p.polarization.z, p.wavelength );
     76     pbuffer[photon_offset+3] = make_float4( p.flags.f.x,     p.flags.f.y,     p.flags.f.z,      p.flags.f.w);
     77 }
     78 



::

    tboolean-;tboolean-box --okg4 --testauto
    tboolean-;tboolean-box-ip

    In [2]: ab.dvtabs[2]
    Out[2]: 
    ox_dv maxdvmax:3.0517578125e-05 maxdv:[3.0517578125e-05, nan] 
     0000            :                          TO SA :  391943   391952  :    391558 6264928/      0: 0.000  mx/mn/av 3.052e-05/     0/1.907e-06  eps:0.0002    
     0001            :                       TO SR SA :  207533   207524  :    207394 3318304/      0: 0.000  mx/mn/av    nan/   nan/   nan  eps:0.0002    


    In [8]: dvt.dvs[1].av
    Out[8]: 
    A()sliced
    A([[[-133.4443,   -1.4124, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -44.3963, -116.7347, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -43.5826, -147.5403, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           ..., 
           [[-144.0839,  450.    ,  -23.8085,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[  71.1732,  450.    ,   56.2633,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -91.8347,  450.    ,   29.8083,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]]], dtype=float32)

    In [9]: dvt.dvs[1].bv
    Out[9]: 
    A()sliced
    A([[[-133.4443,   -1.4124, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [   0.    ,    1.    ,    0.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -44.3963, -116.7347, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [   0.    ,    1.    ,    0.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -43.5826, -147.5403, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [   0.    ,    1.    ,    0.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           ..., 
           [[-144.0839,  450.    ,  -23.8085,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [   0.    ,    0.    ,    1.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[  71.1732,  450.    ,   56.2633,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [   0.    ,    0.    ,    1.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -91.8347,  450.    ,   29.8083,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [   0.    ,    0.    ,    1.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]]], dtype=float32)

    In [10]: 



::


    In [16]: ab.a.ox[:20,2]
    Out[16]: 
    A()sliced
    A([[   0.,   -1.,    0.,  380.],
           [  nan,   nan,   nan,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [   0.,   -1.,    0.,  380.],
           [  nan,   nan,   nan,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [  nan,   nan,   nan,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.]], dtype=float32)

    In [18]: ab.a.ox.shape
    Out[18]: (600000, 4, 4)

    In [20]: ab.a.seqhis.shape
    Out[20]: (600000,)

    In [21]: ab.a.seqhis[:20]
    Out[21]: 
    A()sliced
    A([ 141, 2221,  141,  141,  141,  141, 2221, 2221,  141, 2221,  141,  141,  141,  141,  141, 2221,  141,  141, 2221, 2221], dtype=uint64)

    In [22]: hex(2221)
    Out[22]: '0x8ad'


    In [23]: ab.selhis = "TO SR SA"

    In [25]: ab.a.ox[:20,2]
    Out[25]: 
    A()sliced
    A([[  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.]], dtype=float32)

    In [27]: ab.a.ox.shape
    Out[27]: (207533, 4, 4)

    In [28]: ab.a.rpol()
    Out[28]: 
    A()sliced
    A([[[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           ..., 
           [[ 0.,  0., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0.,  0., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0.,  0., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.]]], dtype=float32)

    In [29]: ab.b.rpol()
    Out[29]: 
    A()sliced
    A([[[ 0., -1.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  1.,  0.]],

           [[ 0., -1.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  1.,  0.]],

           [[ 0., -1.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  1.,  0.]],

           ..., 
           [[ 0.,  0., -1.],
            [ 0.,  0.,  1.],
            [ 0.,  0.,  1.]],

           [[ 0.,  0., -1.],
            [ 0.,  0.,  1.],
            [ 0.,  0.,  1.]],

           [[ 0.,  0., -1.],
            [ 0.,  0.,  1.],
            [ 0.,  0.,  1.]]], dtype=float32)



