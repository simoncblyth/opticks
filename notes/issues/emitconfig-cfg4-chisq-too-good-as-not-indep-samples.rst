emitconfig-cfg4-chisq-too-good-as-not-indep-samples
===================================================






ISSUE  : Need different comparison approach for emitconfig input photons
---------------------------------------------------------------------------

With *emitconfig* OK/G4 samples are not independant, 
as the input photons are identical. BUT the chisq comparison 
machinery was setup for comparing indep samples. 


::

    simon:ana blyth$ tboolean-;tboolean-box-ip
    args: /opt/local/bin/ipython -i -- /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    [2017-11-24 19:31:56,600] p33909 {/Users/blyth/opticks/ana/base.py:316} INFO - envvar OPTICKS_ANA_DEFAULTS -> defaults {'src': 'torch', 'tag': '1', 'det': 'concentric'} 
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    ok.smry 1 
    [2017-11-24 19:31:56,602] p33909 {/Users/blyth/opticks/ana/tboolean.py:27} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython True 
    [2017-11-24 19:31:56,603] p33909 {/Users/blyth/opticks/ana/ab.py:81} INFO - AB.load START smry 1 
    [2017-11-24 19:31:56,774] p33909 {/Users/blyth/opticks/ana/ab.py:108} INFO - AB.load DONE 
    [2017-11-24 19:31:56,778] p33909 {/Users/blyth/opticks/ana/ab.py:150} INFO - AB.init_point START
    [2017-11-24 19:31:56,780] p33909 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171124-1909 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171124-1909 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         4.44/5 =  0.89  (pval:0.488 prob:0.512)  
    0000      55321     55303             0.00  TO SA
    0001      39222     39231             0.00  TO BT BT SA
    0002       2768      2814             0.38  TO BR SA
    0003       2425      2369             0.65  TO BT BR BT SA
    0004        151       142             0.28  TO BT BR BR BT SA
    0005         54        74             3.12  TO SC SA
    0006         13        16             0.00  TO BT BT SC SA
    0007         12         8             0.00  TO BT BR BR BR BT SA



Test --reflectcheat
----------------------

* succeeds to point-by-point align "TO BR SA"

::

    tboolean-;tboolean-box --okg4 --reflectcheat 


::

    [2017-11-30 18:06:48,337] p36797 {/Users/blyth/opticks/ana/ab.py:154} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171130-1806 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171130-1806 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         1.02/3 =  0.34  (pval:0.797 prob:0.203)  
    0000               8d     55321     55312             0.00        1.000 +- 0.004        1.000 +- 0.004  [2 ] TO SA
    0001             8ccd     41828     41816             0.00        1.000 +- 0.005        1.000 +- 0.005  [4 ] TO BT BT SA
    0002              8bd      2754      2754             0.00        1.000 +- 0.019        1.000 +- 0.019  [3 ] TO BR SA
    0003              86d        54        65             1.02        0.831 +- 0.113        1.204 +- 0.149  [3 ] TO SC SA
    0004            86ccd        12        11             0.00        1.091 +- 0.315        0.917 +- 0.276  [5 ] TO BT BT SC SA
    0005              4cd         6         8             0.00        0.750 +- 0.306        1.333 +- 0.471  [3 ] TO BT AB
    0006               4d         6         6             0.00        1.000 +- 0.408        1.000 +- 0.408  [2 ] TO AB
    0007           8cbc6d         5         5             0.00        1.000 +- 0.447        1.000 +- 0.447  [6 ] TO SC BT BR BT SA
    0008            8c6cd         4         5             0.00        0.800 +- 0.400        1.250 +- 0.559  [5 ] TO BT SC BT SA
    0009             86bd         2         2             0.00        1.000 +- 0.707        1.000 +- 0.707  [4 ] TO BR SC SA
    0010          8cbbc6d         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO SC BT BR BR BT SA
    0011       bbbbbbb6cd         2         5             0.00        0.400 +- 0.283        2.500 +- 1.118  [10] TO BT SC BR BR BR BR BR BR BR
    0012           8cb6cd         1         2             0.00        0.500 +- 0.500        2.000 +- 1.414  [6 ] TO BT SC BR BT SA
    0013             8b6d         1         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO SC BR SA
    0014            8cc6d         1         3             0.00        0.333 +- 0.333        3.000 +- 1.732  [5 ] TO SC BT BT SA
    0015             4ccd         1         3             0.00        0.333 +- 0.333        3.000 +- 1.732  [4 ] TO BT BT AB
    0016          8cc6ccd         0         2             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO BT BT SC BT BT SA
    0017         8cbc6ccd         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] TO BT BT SC BT BR BT SA
    .                             100000    100000         1.02/3 =  0.34  (pval:0.797 prob:0.203)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         1.44/4 =  0.36  (pval:0.837 prob:0.163)  
    0000             1080     55321     55312             0.00        1.000 +- 0.004        1.000 +- 0.004  [2 ] TO|SA
    0001             1880     41828     41816             0.00        1.000 +- 0.005        1.000 +- 0.005  [3 ] TO|BT|SA
    0002             1480      2754      2754             0.00        1.000 +- 0.019        1.000 +- 0.019  [3 ] TO|BR|SA
    0003             10a0        54        65             1.02        0.831 +- 0.113        1.204 +- 0.149  [3 ] TO|SA|SC
    0004             18a0        17        21             0.42        0.810 +- 0.196        1.235 +- 0.270  [4 ] TO|BT|SA|SC
    0005             1ca0         8         8             0.00        1.000 +- 0.354        1.000 +- 0.354  [5 ] TO|BT|BR|SA|SC
    0006             1808         7        11             0.00        0.636 +- 0.241        1.571 +- 0.474  [3 ] TO|BT|AB
    0007             1008         6         6             0.00        1.000 +- 0.408        1.000 +- 0.408  [2 ] TO|AB
    0008             14a0         3         2             0.00        1.500 +- 0.866        0.667 +- 0.471  [4 ] TO|BR|SA|SC
    0009             1c20         2         5             0.00        0.400 +- 0.283        2.500 +- 1.118  [4 ] TO|BT|BR|SC
    .                             100000    100000         1.44/4 =  0.36  (pval:0.837 prob:0.163)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.02/2 =  0.01  (pval:0.988 prob:0.012)  
    0000               12     55321     55312             0.00        1.000 +- 0.004        1.000 +- 0.004  [2 ] Vm Rk
    0001             1232     41828     41816             0.00        1.000 +- 0.005        1.000 +- 0.005  [4 ] Vm F2 Vm Rk
    0002              122      2808      2819             0.02        0.996 +- 0.019        1.004 +- 0.019  [3 ] Vm Vm Rk
    0003            12232        12        11             0.00        1.091 +- 0.315        0.917 +- 0.276  [5 ] Vm F2 Vm Vm Rk
    0004               22         6         6             0.00        1.000 +- 0.408        1.000 +- 0.408  [2 ] Vm Vm
    0005              332         6         8             0.00        0.750 +- 0.306        1.333 +- 0.471  [3 ] Vm F2 F2
    0006           123322         5         5             0.00        1.000 +- 0.447        1.000 +- 0.447  [6 ] Vm Vm F2 F2 Vm Rk
    0007            12332         4         5             0.00        0.800 +- 0.400        1.250 +- 0.559  [5 ] Vm F2 F2 Vm Rk
    0008             1222         3         2             0.00        1.500 +- 0.866        0.667 +- 0.471  [4 ] Vm Vm Vm Rk
    0009          1233322         2         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Vm Vm F2 F2 F2 Vm Rk
    0010       3333333332         2         5             0.00        0.400 +- 0.283        2.500 +- 1.118  [10] Vm F2 F2 F2 F2 F2 F2 F2 F2 F2
    0011             2232         1         3             0.00        0.333 +- 0.333        3.000 +- 1.732  [4 ] Vm F2 Vm Vm
    0012           123332         1         2             0.00        0.500 +- 0.500        2.000 +- 1.414  [6 ] Vm F2 F2 F2 Vm Rk
    0013            12322         1         3             0.00        0.333 +- 0.333        3.000 +- 1.732  [5 ] Vm Vm F2 Vm Rk
    0014         12332232         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [8 ] Vm F2 Vm Vm F2 F2 Vm Rk
    0015          1232232         0         2             0.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] Vm F2 Vm Vm F2 Vm Rk
    .                             100000    100000         0.02/2 =  0.01  (pval:0.988 prob:0.012)  
    ab.a.metadata                  /tmp/blyth/opticks/evt/tboolean-box/torch/1 7c3396a4bcfc21cba051ba98f6f0b667 781d1ab8f0adbf585c197cf43a538446  100000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0 {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75', u'resolution': u'20', u'emit': -1}
    c2p : {'seqmat_ana': 0.011978598910194116, 'pflags_ana': 0.36007827166347472, 'seqhis_ana': 0.33975348502498387} c2pmax: 0.360078271663  CUT ok.c2max 2.0  RC:0 
    [2017-11-30 18:06:48,345] p36797 {/Users/blyth/opticks/ana/tboolean.py:38} INFO - early exit as non-interactive
    2017-11-30 18:06:48.381 INFO  [707800] [SSys::run@46] tboolean.py --tag 1 --tagoffset 0 --det tboolean-box --src torch --nosmry   rc_raw : 0 rc : 0


    rpost_dv
     0000            :                          TO SA :   55321    55312  :     55258/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    
     0001            :                    TO BT BT SA :   41828    41816  :     41776/      8: 0.000  mx/mn/av 0.0138/0.0000/2.636e-06    
     0002            :                       TO BR SA :    2754     2754  :      2751/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    
    rpol_dv
     0000            :                          TO SA :   55321    55312  :     55258/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    
     0001            :                    TO BT BT SA :   41828    41816  :     41776/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    
     0002            :                       TO BR SA :    2754     2754  :      2751/      0: 0.000  mx/mn/av 0.0000/0.0000/     0    
    c2p : {'seqmat_ana': 0.011978598910194116, 'pflags_ana': 0.36007827166347472, 'seqhis_ana': 0.33975348502498387} c2pmax: 0.360078271663  CUT ok.c2max 2.0  RC:0 
    [2017-11-30 18:10:53,190] p37096 {/Users/blyth/opticks/ana/tboolean.py:42} INFO - early exit as non-interactive




APPROACHES
------------

* avoid issue using indep samples (devise some seeding convention)
  then the existing chisq comparisons would be valid 

* direct photons need step-by-step value-to-value comparison, 

  * check avg deviations, see **ana/dv.py** used by ab.rpost_dv and ab.rpol_dv 
  * investigate outliers, not automated 
  * perhaps expand direct to include SR/BR with u_cheat ?

* Q: non-direct photons, is chisq history comparison valid for them, despite identical birth ?

  * probably not, anyhow once BR/SR can be cheated there are not so many of them, 
    so any comparisons would need very high stats 
      


BEFORE ANY CHEATING : NUMPY MACHINERY FOR ALIGNED COMPARISON
---------------------------------------------------------------

* :doc:`emitconfig-aligned-comparison`


implementing reflectcheat
--------------------------

::

    simon:opticks blyth$ opticks-find reflectcheat
    ./optixrap/cu/generate.cu:    s.ureflectcheat = 0.f ; 
    ./optixrap/cu/generate.cu:        s.ureflectcheat = debug_control.w > 0u ? float(photon_id)/float(num_photon) : -1.f ;
    ./cfg4/DsG4OpBoundaryProcess.cc:             m_reflectcheat(m_ok->isReflectCheat()),
    ./cfg4/DsG4OpBoundaryProcess.cc:          // --reflectcheat 
    ./optickscore/Opticks.cc:bool Opticks::isReflectCheat() const  // reflectcheat
    ./optickscore/Opticks.cc:   return m_cfg->hasOpt("reflectcheat");
    ./optickscore/OpticksCfg.cc:       ("reflectcheat",  
    ./optixrap/OPropagator.cc:    unsigned reflectcheat = m_ok->isReflectCheat() ? 1 : 0 ; 
    ./optixrap/OPropagator.cc:    if(reflectcheat > 0 )
    ./optixrap/OPropagator.cc:        LOG(error) <<  "OPropagator::initParameters --reflectcheat ENABLED "  ;
    ./optixrap/OPropagator.cc:    optix::uint4 debugControl = optix::make_uint4(m_ocontext->getDebugPhoton(),0,0, reflectcheat);
    ./cfg4/CG4Ctx.hh:    float _record_fraction ; // used with --reflectcheat
    ./cfg4/DsG4OpBoundaryProcess.h:    bool          m_reflectcheat ; 
    ./optixrap/cu/propagate.h:    const float u = s.ureflectcheat >= 0.f ? s.ureflectcheat : curand_uniform(&rng) ;
    ./optixrap/cu/state.h:   float ureflectcheat ;  
    simon:opticks blyth$ 



THOUGHTS ON CHEATING REFLECTION RANDOMNESS BR/SR : u_cheat=photon_index/num_photons
-------------------------------------------------------------------------------------

* very tempting to cheat the single random throw deciding to reflect or not (BR or SR)
  using *u_cheat=photon_index/num_photons* instead of *curand_uniform(&rng)*
  easy enough for Opticks, but what about G4 ?

* needs to be done in a manner indep of order (so parallel Opticks and G4 do same), 
  ie absolute external photon_index and num_photons

* would greatly enlarge the "direct non-random photons" category 

* would keep both simulations doing exactly the same thing for all non AB/RE/SC categories, 
  and those can all be switched off (--noab/--nore/--nosc) to make purely non-random samples

* what about photons, with more that one BR/SR ? What to use to keep the same seqhis fractions ?


G4 
~~~~

* custom DsG4OpBoundaryProcess already in use, just need to add a cheat flag and arrange that 
  *u_cheat* is set for each photon 


cfg4/OpNovicePhysicsList.cc::


    171 void OpNovicePhysicsList::ConstructProcess()
    172 {
    173   setupEmVerbosity(0);
    174 
    175   //AddTransportation();
    176   addTransportation();
    177 
    178 
    179   ConstructDecay();
    180   ConstructEM();
    181 
    182   ConstructOpDYB();
    183 
    184   dump("OpNovicePhysicsList::ConstructProcess");
    185 }


    221 void OpNovicePhysicsList::ConstructOpDYB()
    222 {


::


    (lldb) b OpNovicePhysicsList::ConstructOpDYB


    (lldb) c
    Process 41096 resuming
    2017-11-24 20:47:04.444 INFO  [7043277] [OpNovicePhysicsList::ConstructOpDYB@225] Using customized DsG4Cerenkov.
    2017-11-24 20:47:04.444 INFO  [7043277] [OpNovicePhysicsList::ConstructOpDYB@265] Using customized DsG4Scintillation.
    2017-11-24 20:47:04.444 INFO  [7043277] [DsG4OpBoundaryProcess::DsG4OpBoundaryProcess@124] DsG4OpBoundaryProcess::DsG4OpBoundaryProcess processName OpBoundary
    Process 41096 stopped
    * thread #1: tid = 0x6b78cd, 0x00000001043582fb libcfg4.dylib`OpNovicePhysicsList::ConstructOpDYB(this=0x000000011283ce40) + 2235 at OpNovicePhysicsList.cc:329, queue = 'com.apple.main-thread', stop reason = breakpoint 2.1
        frame #0: 0x00000001043582fb libcfg4.dylib`OpNovicePhysicsList::ConstructOpDYB(this=0x000000011283ce40) + 2235 at OpNovicePhysicsList.cc:329
       326  
       327      //G4OpBoundaryProcess* boundproc = new G4OpBoundaryProcess();
       328      DsG4OpBoundaryProcess* boundproc = new DsG4OpBoundaryProcess(m_g4);
    -> 329      boundproc->SetModel(unified);
       330  
       331      //G4FastSimulationManagerProcess* fast_sim_man = new G4FastSimulationManagerProcess("fast_sim_man");
       332      
    (lldb) p boundproc
    (DsG4OpBoundaryProcess *) $0 = 0x0000000112902390
    (lldb) 






g4op-;g4op-vi::

     393      983               G4double E2_total = E2_perp*E2_perp + E2_parl*E2_parl;         // square up s and p amplitudes to get overall intensity
     394      984               G4double s2 = Rindex2*cost2*E2_total;   //  is this the planar angle term    (24)
     395      985 
     396      986               G4double TransCoeff;
     397      987 
     398      988               if (theTransmittance > 0) TransCoeff = theTransmittance;
     399      989               else if (cost1 != 0.0) TransCoeff = s2/s1;     //  transmission probability  "Transmittance = 1 - Reflectance"
     400      990               else TransCoeff = 0.0;
     401 
     402      ///   fresnel-eoe.pdf
     403      ///       ...the intensity is calculated per unit of the wavefront area, and the wavefronts of the incident 
     404      ///       and transmitted wave are tilted with respect to the interface at different angles theta_i and theta_t, respectively. 
     405      ///       Therefore, the intensity transmissivity is given by (24)
     406      ///
     407      ///
     408      ///                         n2 cost2 |Et|^2        n2 cost2
     409      ///                   T = ------------------- =   ---------- |t|^2
     410      ///                         n1 cost1 |Ei|^2        n1 cost1 
     411      ///
     412      ...
     413      992           G4double E2_abs, C_parl, C_perp;
     414      993 
     415      994           if ( !G4BooleanRand(TransCoeff) ) {   // not transmission, so reflection
     416      998                  if (Swap) Swap = !Swap;
     417     1000                  theStatus = FresnelReflection;
     418     1002                  if ( theModel == unified && theFinish != polished )
     419     1003                                 ChooseReflection();
     420     1004 
     421     1005                  if ( theStatus == LambertianReflection ) {
     422     1006                      DoReflection();
     423     1007                  }




    simon:optixrap blyth$ g4-cc G4BooleanRand
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc:                   if ( !G4BooleanRand(theReflectivity) ) {
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc:              } while ( !G4BooleanRand(AngularDistributionValue) );
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc:        if ( !G4BooleanRand(theTransmittance) ) { // Not transmitted, so reflect
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc:                                     G4BooleanRand(SurfaceRoughnessCriterion);
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc:         if ( !G4BooleanRand(TransCoeff) ) {
    simon:optixrap blyth$ 
    simon:optixrap blyth$ 
    simon:optixrap blyth$ g4-hh G4BooleanRand
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/include/G4OpBoundaryProcess.hh:   G4bool G4BooleanRand(const G4double prob) const;
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/include/G4OpBoundaryProcess.hh:G4bool G4OpBoundaryProcess::G4BooleanRand(const G4double prob) const
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/include/G4OpBoundaryProcess.hh:              if ( G4BooleanRand(theEfficiency) ) {
    simon:optixrap blyth$ 


G4BooleanRand tis private method and used for other things like absorption::


    178 private:
    179 
    180     G4bool G4BooleanRand(const G4double prob) const;
    181 
    ...
    260 ////////////////////
    261 // Inline methods
    262 ////////////////////
    263 
    264 inline
    265 G4bool G4OpBoundaryProcess::G4BooleanRand(const G4double prob) const
    266 {
    267   /* Returns a random boolean variable with the specified probability */
    268 
    269   return (G4UniformRand() < prob);
    270 }






Opticks
~~~~~~~~~~


::

    243 
    244 __device__ void propagate_at_boundary_geant4_style( Photon& p, State& s, curandState &rng)
    245 {
    246     // see g4op-/G4OpBoundaryProcess.cc annotations to follow this
    ...
    283     const float E2_perp_r = E2_perp_t - E1_perp;           // Fresnel S-pol reflectance
    284     const float E2_parl_r = (n2*E2_parl_t/n1) - E1_parl ;  // Fresnel P-pol reflectance
    285 
    286     const float2 E2_t = make_float2( E2_perp_t, E2_parl_t ) ;
    287     const float2 E2_r = make_float2( E2_perp_r, E2_parl_r ) ;
    288 
    289     const float  E2_total_t = dot(E2_t,E2_t) ;
    290 
    291     const float2 T = normalize(E2_t) ;
    292     const float2 R = normalize(E2_r) ;
    293 
    294     const float TransCoeff =  tir ? 0.0f : n2c2*E2_total_t/n1c1 ;
    295     //  above 0.0f was until 2016/3/4 incorrectly a 1.0f 
    296     //  resulting in TIR yielding BT where BR is expected
    297 
    298     bool reflect = curand_uniform(&rng) > TransCoeff  ;
    299 
    300     p.direction = reflect
    301                     ?
    302                        p.direction + 2.0f*c1*s.surface_normal
    303                     :
    304                        eta*p.direction + (eta_c1 - c2)*s.surface_normal
    305                     ;
    306 
    307     const float3 A_paral = normalize(cross(p.direction, A_trans));
    308 
    309     p.polarization = reflect ?
    310                                 ( tir ?
    311                                         -p.polarization + 2.f*EdotN*s.surface_normal
    312                                       :
    313                                         R.x*A_trans + R.y*A_paral
    314                                 )
    315                              :
    316                                 T.x*A_trans + T.y*A_paral
    317                              ;





::

    517 __device__ int
    518 propagate_at_surface(Photon &p, State &s, curandState &rng)
    519 {
    520 
    521     float u = curand_uniform(&rng);
    522 
    523     if( u < s.surface.y )   // absorb   
    524     {
    525         s.flag = SURFACE_ABSORB ;
    526         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    527         return BREAK ;
    528     }
    529     else if ( u < s.surface.y + s.surface.x )  // absorb + detect
    530     {
    531         s.flag = SURFACE_DETECT ;
    532         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    533         return BREAK ;
    534     }
    535     else if (u  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    536     {
    537         s.flag = SURFACE_DREFLECT ;
    538         propagate_at_diffuse_reflector_geant4_style(p, s, rng);
    539         return CONTINUE;
    540     }
    541     else
    542     {
    543         s.flag = SURFACE_SREFLECT ;
    544         propagate_at_specular_reflector(p, s, rng );
    545         return CONTINUE;
    546     }
    547 }





No RNG impact "direct photons"
--------------------------------

Directly absorbed and straight thru photons, are not effected by RNG 
so should have identical values at every step.
Domain compression is identical between branches ? 

* TO SA  
* TO BT BT SA 
 
Same positions, pol, wavelength, times
they can be step-by-step one-to-one compared
and average deviation distances/times formed.

::

    In [1]: ab.sel = "TO BT BT SA"   # straight thru, is same in both simulations 

    In [2]: ab.a.rpost_(slice(0,4))     # but some presence differences 
    Out[2]: 
    A()sliced
    A([[[-133.4405,   -1.4177, -449.8989,    0.2002],
            [-133.4405,   -1.4177,  -99.9944,    1.3672],
            [-133.4405,   -1.4177,   99.9944,    2.5788],
            [-133.4405,   -1.4177,  449.9952,    3.7465]],

           [[ -44.4022, -116.7312, -449.8989,    0.2002],
            [ -44.4022, -116.7312,  -99.9944,    1.3672],
            [ -44.4022, -116.7312,   99.9944,    2.5788],
            [ -44.4022, -116.7312,  449.9952,    3.7465]],

           [[ -93.6355,  105.1833, -449.8989,    0.2002],
            [ -93.6355,  105.1833,  -99.9944,    1.3672],
            [ -93.6355,  105.1833,   99.9944,    2.5788],
            [ -93.6355,  105.1833,  449.9952,    3.7465]],

           ..., 
           [[ -20.6182,   16.8469, -449.8989,    0.2002],
            [ -20.6182,   16.8469,  -99.9944,    1.3672],
            [ -20.6182,   16.8469,   99.9944,    2.5788],
            [ -20.6182,   16.8469,  449.9952,    3.7465]],

           [[-112.0515,   -6.8682, -449.8989,    0.2002],
            [-112.0515,   -6.8682,  -99.9944,    1.3672],
            [-112.0515,   -6.8682,   99.9944,    2.5788],
            [-112.0515,   -6.8682,  449.9952,    3.7465]],

           [[  -9.4558,   -7.2673, -449.8989,    0.2002],
            [  -9.4558,   -7.2673,  -99.9944,    1.3672],
            [  -9.4558,   -7.2673,   99.9944,    2.5788],
            [  -9.4558,   -7.2673,  449.9952,    3.7465]]])

    In [3]: ab.b.rpost_(slice(0,4))
    Out[3]: 
    A()sliced
    A([[[-133.4405,   -1.4177, -449.8989,    0.2002],
            [-133.4405,   -1.4177,  -99.9944,    1.3672],
            [-133.4405,   -1.4177,   99.9944,    2.5788],
            [-133.4405,   -1.4177,  449.9952,    3.7465]],

           [[ -44.4022, -116.7312, -449.8989,    0.2002],
            [ -44.4022, -116.7312,  -99.9944,    1.3672],
            [ -44.4022, -116.7312,   99.9944,    2.5788],
            [ -44.4022, -116.7312,  449.9952,    3.7465]],
         
          ## some diffs
           [[  24.3758,  139.9646, -449.8989,    0.2002],
            [  24.3758,  139.9646,  -99.9944,    1.3672],
            [  24.3758,  139.9646,   99.9944,    2.5788],
            [  24.3758,  139.9646,  449.9952,    3.7465]],


           ..., 
           [[ -20.6182,   16.8469, -449.8989,    0.2002],
            [ -20.6182,   16.8469,  -99.9944,    1.3672],
            [ -20.6182,   16.8469,   99.9944,    2.5788],
            [ -20.6182,   16.8469,  449.9952,    3.7465]],

           [[-112.0515,   -6.8682, -449.8989,    0.2002],
            [-112.0515,   -6.8682,  -99.9944,    1.3672],
            [-112.0515,   -6.8682,   99.9944,    2.5788],
            [-112.0515,   -6.8682,  449.9952,    3.7465]],

           [[  -9.4558,   -7.2673, -449.8989,    0.2002],
            [  -9.4558,   -7.2673,  -99.9944,    1.3672],
            [  -9.4558,   -7.2673,   99.9944,    2.5788],
            [  -9.4558,   -7.2673,  449.9952,    3.7465]]])

    In [4]: 







RNG impact
-------------
With RNG effect:

* BR (which photons get reflected depend on RNG throw)
* AB (which photons get absorbed and the position depend on RNG) 
* SC/RE (which photons scatter/reemit, the position and param afterwards depend on RNG) 


But reflection brings in RNG, its random which photons get reflected::


    In [28]: ab.sel = "TO BR SA"

    In [29]: ab.a.rpost_(slice(0,3))
    Out[29]: 
    A()sliced
    A([[[ -43.5763, -147.5347, -449.8989,    0.2002],
            [ -43.5763, -147.5347,  -99.9944,    1.3672],
            [ -43.5763, -147.5347, -449.9952,    2.5349]],

           [[  24.3758,  139.9646, -449.8989,    0.2002],
            [  24.3758,  139.9646,  -99.9944,    1.3672],
            [  24.3758,  139.9646, -449.9952,    2.5349]],

           [[ -11.135 ,  -82.762 , -449.8989,    0.2002],
            [ -11.135 ,  -82.762 ,  -99.9944,    1.3672],
            [ -11.135 ,  -82.762 , -449.9952,    2.5349]],

           ..., 
           [[  46.5631,  117.8874, -449.8989,    0.2002],
            [  46.5631,  117.8874,  -99.9944,    1.3672],
            [  46.5631,  117.8874, -449.9952,    2.5349]],

           [[-106.2156,  101.1643, -449.8989,    0.2002],
            [-106.2156,  101.1643,  -99.9944,    1.3672],
            [-106.2156,  101.1643, -449.9952,    2.5349]],

           [[ -70.2094, -142.2218, -449.8989,    0.2002],
            [ -70.2094, -142.2218,  -99.9944,    1.3672],
            [ -70.2094, -142.2218, -449.9952,    2.5349]]])

    In [30]: ab.b.rpost_(slice(0,3))
    Out[30]: 
    A()sliced
    A([[[-149.5993, -110.5099, -449.8989,    0.2002],
            [-149.5993, -110.5099,  -99.9944,    1.3672],
            [-149.5993, -110.5099, -449.9952,    2.5349]],

           [[ 120.2547,   24.7749, -449.8989,    0.2002],
            [ 120.2547,   24.7749,  -99.9944,    1.3672],
            [ 120.2547,   24.7749, -449.9952,    2.5349]],

           [[-111.2945,  140.2261, -449.8989,    0.2002],
            [-111.2945,  140.2261,  -99.9944,    1.3672],
            [-111.2945,  140.2261, -449.9952,    2.5349]],

           ..., 
           [[  88.4602,  102.3755, -449.8989,    0.2002],
            [  88.4602,  102.3755,  -99.9944,    1.3672],
            [  88.4602,  102.3755, -449.9952,    2.5349]],

           [[ 123.2553,  -67.8282, -449.8989,    0.2002],
            [ 123.2553,  -67.8282,  -99.9944,    1.3672],
            [ 123.2553,  -67.8282, -449.9952,    2.5349]],

           [[ -13.9978,  -80.6424, -449.8989,    0.2002],
            [ -13.9978,  -80.6424,  -99.9944,    1.3672],
            [ -13.9978,  -80.6424, -449.9952,    2.5349]]])

    In [31]: 



