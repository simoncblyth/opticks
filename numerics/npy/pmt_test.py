#!/usr/bin/env python
"""
PmtInBox Opticks vs cfg4
==================================

Without and with cfg4 runs::

   ggv-;ggv-pmt-test 
   ggv-;ggv-pmt-test --cfg4 

Visualize the cfg4 created evt in interop mode viewer::

   ggv-;ggv-pmt-test --cfg4 --load

Issues
-------

* source issue in cfg4 (b), allmost all photons doing same thing, 
  fixed by handling discLinear

* different G4 geometry, photon interaction positions in the interop visualization 
  show that primitives are there, but Z offsets not positioned correctly so the 
  boolean processing produces a more complicated geometry 

  * modify cfg4-/Detector to make just the Pyrex for simplification 

* after first order fixing G4 geometry to look OK, 
  still very different sequence histories because are missing surface/SD
  handling that leads to great simplification for Opticks as most photons 
  are absorbed/detected on the photocathode

* suspect the DYB detdesc G4 positioning of the photocathode inside the vacuum 
  with coincident surfaces will lead to comparison problems, as this "feature"
  was fixed for the surface-based translation  

  May need to compare against a correspondingly "fixed" G4 geometry

* examining the sensdet/hit handling in LXe example observe
  that its essentially a manual thing for optical photons, so 
  the overhead of sensdet and hits is not useful for cfg4 purposes.
  Instead just need to modify cfg4-/RecordStep to return done=true 
  on walking onto the photocathode : but need to identify, and
  need the EFFICIENCY ? 

  * hmm what happened to to the EFFICIENCY ? 
    transmogrified to GSurfaceLib "detect" property that gets
    copied across to GPU texture


assimpwrap-/AssimpGGeo.cc/AssimpGGeo::convertMaterials::

     438             if(hasVectorProperty(mat, EFFICIENCY ))
     439             {
     440                 assert(gg->getCathode() == NULL && "only expecting one material with an EFFICIENCY property" );
     441                 gg->setCathode(gmat) ;
     442                 m_cathode = mat ;
     443             }
     ...
     466 void AssimpGGeo::convertSensors(GGeo* gg)
     467 {
     468 /*
     469 Opticks is a surface based simulation, as opposed to 
     470 Geant4 which is CSG volume based. In Geant4 hits are formed 
     471 on stepping into volumes with associated SensDet.
     472 The Opticks equivalent is intersecting with a "SensorSurface", 
     473 which are fabricated by AssimpGGeo::convertSensors.
     474 */
     475     convertSensors( gg, m_tree->getRoot(), 0);
     476 



G4 Efficiency
~~~~~~~~~~~~~~~

Where does the random check against EFFICIENCY as
function of wavelength happen for G4 ? Need to get G4 to decide between
absorb/detect and return status ? 

* answer: G4OpBoundaryProcess::DoAbsorption

::

    simon:geant4.10.02 blyth$ find source -name '*.cc' -exec grep -H EFFICIENCY {} \;
    source/global/HEPNumerics/src/G4ConvergenceTester.cc:   out << std::setw(20) << "EFFICIENCY = " << std::setw(13)  << efficiency << G4endl;
    source/processes/optical/src/G4OpBoundaryProcess.cc:              aMaterialPropertiesTable->GetProperty("EFFICIENCY");
    simon:geant4.10.02 blyth$ 


    165 G4VParticleChange*
    166 G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    167 {
    ...
    387               PropertyPointer =
    388               aMaterialPropertiesTable->GetProperty("EFFICIENCY");
    389               if (PropertyPointer) {
    390                       theEfficiency =
    391                       PropertyPointer->Value(thePhotonMomentum);
    392               }


    306 inline
    307 void G4OpBoundaryProcess::DoAbsorption()
    308 {
    309               theStatus = Absorption;
    310 
    311               if ( G4BooleanRand(theEfficiency) ) {
    312 
    313                  // EnergyDeposited =/= 0 means: photon has been detected
    314                  theStatus = Detection;
    315                  aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
    316               }
    317               else {
    318                  aParticleChange.ProposeLocalEnergyDeposit(0.0);
    319               }
    320 
    321               NewMomentum = OldMomentum;
    322               NewPolarization = OldPolarization;
    323 
    324 //              aParticleChange.ProposeEnergy(0.0);
    325               aParticleChange.ProposeTrackStatus(fStopAndKill);
    326 }






Very different::

    In [54]: a.history_table()
    Evt(1,"torch","PmtInBox","", seqs="[]")
                              noname 
                     8cd       375203       [3 ] TO BT SA
                     7cd       118603       [3 ] TO BT SD
                     8bd         4458       [3 ] TO BR SA
                     4cd         1471       [3 ] TO BT AB
                      4d           87       [2 ] TO AB
                 8cccccd           78       [7 ] TO BT BT BT BT BT SA
                     86d           68       [3 ] TO SC SA
                    8c6d           15       [4 ] TO SC BT SA
                  8bcccd            5       [6 ] TO BT BT BT BR SA
              ccccccbccd            3       [10] TO BT BT BR BT BT BT BT BT BT
                    4ccd            2       [4 ] TO BT BT AB
                    86bd            2       [4 ] TO BR SC SA
               8cccbbccd            1       [9 ] TO BT BT BR BR BT BT BT SA
                   8c6cd            1       [5 ] TO BT SC BT SA
                  8c6ccd            1       [6 ] TO BT BT SC BT SA
                     4bd            1       [3 ] TO BR AB
                    7c6d            1       [4 ] TO SC BT SD
                              500000 


    In [3]: b.history_table()
    Evt(-1,"torch","PmtInBox","", seqs="[]")
                              noname 
                8ccccccd       276675       [8 ] TO BT BT BT BT BT BT SA
                 8ccbccd       157768       [7 ] TO BT BT BR BT BT SA
              cccccccccd        14167       [10] TO BT BT BT BT BT BT BT BT BT
              ccccbccccd        13398       [10] TO BT BT BT BT BR BT BT BT BT
                     8bd         5397       [3 ] TO BR SA
              cbcccccccd         5153       [10] TO BT BT BT BT BT BT BT BR BT
              bbbbcccccd         4528       [10] TO BT BT BT BT BT BR BR BR BR
                  8ccbcd         4033       [6 ] TO BT BR BT BT SA
              cccbcccccd         3316       [10] TO BT BT BT BT BT BR BT BT BT
               8cccccccd         2700       [9 ] TO BT BT BT BT BT BT BT SA
              ccbcbcbccd         1895       [10] TO BT BT BR BT BR BT BR BT BT
                     4cd         1741       [3 ] TO BT AB




"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from env.numerics.npy.evt import Evt

X,Y,Z,W = 0,1,2,3


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4, linewidth=200)

    plt.ion()
    plt.close()

    tag = "1"
    a = Evt(tag="%s" % tag, src="torch", det="PmtInBox")
    b = Evt(tag="-%s" % tag , src="torch", det="PmtInBox")




