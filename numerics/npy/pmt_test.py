#!/usr/bin/env python
"""
PmtInBox Opticks vs cfg4
==================================

Without and with cfg4 runs::

   ggv-;ggv-pmt-test 
   ggv-;ggv-pmt-test --cfg4 

Visualize the cfg4 created evt in interop mode viewer::

   ggv-;ggv-pmt-test --cfg4 --load

Use G4 ui for G4 viz::

   ggv-;ggv-pmt-test --cfg4 --g4ui


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


::

    [2016-Feb-25 13:15:57.521542]:info: Detector::convertMaterial   name Pyrex materialIndex 13
    [2016-Feb-25 13:15:57.523946]:info: Detector::convertMaterial   name Vacuum materialIndex 12
    [2016-Feb-25 13:15:57.526100]:info: Detector::convertMaterial   name Bialkali materialIndex 4
    [2016-Feb-25 13:15:57.527616]:info: Detector::convertMaterial   name OpaqueVacuum materialIndex 10
    [2016-Feb-25 13:15:57.528817]:info: Detector::convertMaterial   name OpaqueVacuum materialIndex 10

    simon:GMaterialLib blyth$ cat order.json 

        "MineralOil": "4",
        "Bialkali": "5",
        "OpaqueVacuum": "11",
        "Vacuum": "13",
        "Pyrex": "14",


G4 Efficiency
~~~~~~~~~~~~~~~

Where does the random check against EFFICIENCY as
function of wavelength happen for G4 ? Need to get G4 to decide between
absorb/detect and return status ? 

* answer: G4OpBoundaryProcess::DoAbsorption

* but it seems that will never be called without an optical surface to 
  set the EFFICIENCY to allow DoAbsorption to get called


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


suspect g4dae export is missing some optical surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Huh DDDB/PMT/properties.xml appears to set EFFICIENCY and REFLECIVITY to zero 
for PMT border surfaces


/usr/local/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT/properties.xml::


     01 <?xml version='1.0' encoding='UTF-8'?>
      2 <!DOCTYPE DDDB SYSTEM "../DTD/geometry.dtd">
      3 
      4 <DDDB>
      5 
      6   <catalog name="PmtSurfaces">
      7     <surfaceref href="#PmtGlassPhoCatSurface"/>
      8     <surfaceref href="#PmtGlassVacuumSurface"/>
      9   </catalog>
     10 
     11   <catalog name="PmtSurfaceTabProperty">
     12     <tabpropertyref href="#PmtGlassPhoCatReflectivity"/>
     13     <tabpropertyref href="#PmtGlassPhoCatEfficiency"/>
     14     <tabpropertyref href="#PmtGlassVacuumReflectivity"/>
     15     <tabpropertyref href="#PmtGlassVacuumEfficiency"/>
     16   </catalog>
     17 
     18   <!-- Surfaces -->
     19 
     20   <surface name="PmtGlassPhoCatSurface"
     21        model="glisur"
     22        finish="polished"
     23        type="dielectric_dielectric"
     24        value="0"
     25        volfirst="/dd/Geometry/PMT/lvPmtHemiGlass"
     26        volsecond="/dd/Geometry/PMT/lvPmtHemiCathode">
     //
     //   name lvPmtHemiGlass looks wrong would expect /dd/Geometry/PMT/lvPmtHemi
     //   also the lvPmtHemiCathode is child of the lvPmtHemiVacuum ??
     //
     //
     27     <tabprops address="/dd/Geometry/PMT/PmtSurfaceTabProperty/PmtGlassPhoCatReflectivity"/>
     28     <tabprops address="/dd/Geometry/PMT/PmtSurfaceTabProperty/PmtGlassPhoCatEfficiency"/>
     29   </surface>
     30 
     31   <surface name="PmtGlassVacuumSurface"
     32        model="glishur"
     //
     //    typo but glisur is default anyhow
     //
     33        finish="polished"
     34        type="dielectric_dielectric"
     35        value="0"
     36        volfirst="/dd/Geometry/PMT/lvPmtHemiGlass"
     37        volsecond="/dd/Geometry/PMT/lvPmtHemiVacuum">
     //
     //   again name lvPmtHemiGlass looks wrong, would expect /dd/Geometry/PMT/lvPmtHemi
     //   border surfaces work 
     //
     38     <tabprops address="/dd/Geometry/PMT/PmtSurfaceTabProperty/PmtGlassVacuumReflectivity"/>
     39     <tabprops address="/dd/Geometry/PMT/PmtSurfaceTabProperty/PmtGlassVacuumEfficiency"/>
     40   </surface>
     41 
     42   <!-- Tabled properties -->
     43 
     44   <tabproperty name="PmtGlassPhoCatReflectivity"
     45            type="REFLECTIVITY"
     46            xunit="eV"
     47            xaxis="PhotonEnergy"
     48            yaxis="Reflectivity">
     49     1.0  0.0
     50     2.0  0.0
     51     3.0  0.0
     52     4.0  0.0
     53     5.0  0.0



The above looks to be outdated, this one is better

* /usr/local/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PmtPanel/properties.xml::

::

    simon:npy blyth$ diff /usr/local/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PMT/properties.xml /usr/local/env/dyb/NuWa-trunk/dybgaudi/Detector/XmlDetDesc/DDDB/PmtPanel/properties.xml
    25c25
    <      volfirst="/dd/Geometry/PMT/lvPmtHemiGlass"
    ---
    >      volfirst="/dd/Geometry/PMT/lvPmtHemi"
    36c36
    <      volfirst="/dd/Geometry/PMT/lvPmtHemiGlass"
    ---
    >      volfirst="/dd/Geometry/PMT/lvPmtHemi"
    simon:npy blyth$ 



With theReflectivity and theTransmittance set to zero via the logical border surface
the DoAbsorption will always get called on the photocathode boundary::

     483         else if (type == dielectric_dielectric) {
     484 
     485           if ( theFinish == polishedbackpainted ||
     486                theFinish == groundbackpainted ) {
     487              DielectricDielectric();
     488           }
     489           else {
     490              G4double rand = G4UniformRand();
     491              if ( rand > theReflectivity ) {
     492                 if (rand > theReflectivity + theTransmittance) {
     493                    DoAbsorption();
     494                 } else {
     495                    theStatus = Transmission;
     496                    NewMomentum = OldMomentum;
     497                    NewPolarization = OldPolarization;
     498                 }
     499              }



names for breakpointing
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:env blyth$ nm /usr/local/env/g4/geant4.10.02.install/lib/libG4processes.dylib | c++filt | grep G4OpBoundaryProcess
    0000000001390d00 unsigned short G4OpBoundaryProcess::DoAbsorption()
    0000000001390de0 unsigned short G4OpBoundaryProcess::DoReflection()
    0000000001391b70 unsigned short G4OpBoundaryProcess::IsApplicable(G4ParticleDefinition const&)
    0000000001386f40 T G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)
    000000000138afe0 T G4OpBoundaryProcess::DielectricLUT()
    000000000138a4a0 T G4OpBoundaryProcess::DielectricMetal()
    000000000138ee20 T G4OpBoundaryProcess::GetMeanFreePath(G4Track const&, double, G4ForceCondition*)
    000000000138eef0 T G4OpBoundaryProcess::GetReflectivity(double, double, double, double, double)
    0000000001391430 unsigned short G4OpBoundaryProcess::ChooseReflection()
    000000000138ee50 T G4OpBoundaryProcess::GetIncidentAngle()
    000000000138b920 T G4OpBoundaryProcess::DielectricDichroic()
    000000000138c560 T G4OpBoundaryProcess::DielectricDielectric()
    0000000001389ec0 T G4OpBoundaryProcess::CalculateReflectivity()
    000000000138e470 T G4OpBoundaryProcess::InvokeSD(G4Step const*)
    0000000001386e20 T G4OpBoundaryProcess::G4OpBoundaryProcess(G4String const&, G4ProcessType)
    0000000001386a70 T G4OpBoundaryProcess::G4OpBoundaryProcess(G4String const&, G4ProcessType)
    0000000001386f10 T G4OpBoundaryProcess::~G4OpBoundaryProcess()
    0000000001386ef0 T G4OpBoundaryProcess::~G4OpBoundaryProcess()
    0000000001386e50 T G4OpBoundaryProcess::~G4OpBoundaryProcess()
    00000000013913f0 unsigned short G4OpBoundaryProcess::G4BooleanRand(double) const
    000000000138e590 T G4OpBoundaryProcess::GetFacetNormal(CLHEP::Hep3Vector const&, CLHEP::Hep3Vector const&) const
    0000000001389200 T G4OpBoundaryProcess::BoundaryProcessVerbose() const
    000000000191a720 S typeinfo for G4OpBoundaryProcess
    00000000017ba0c0 S typeinfo name for G4OpBoundaryProcess
    000000000191a650 S vtable for G4OpBoundaryProcess
    simon:env blyth$ 


::


    simon:env blyth$ ggv-;ggv-pmt-test --cfg4 --dbg

    (lldb) b "G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)"
    Breakpoint 1: no locations (pending).
    WARNING:  Unable to resolve breakpoint to any actual locations.
    (lldb) r
    (lldb) expr verboseLevel = 1
    (lldb) b "G4OpBoundaryProcess::DielectricDielectric()"


* a Opticks
* b CFG4

After kludge photocathode are getting closer to the ballpark
::

    In [18]: a.history_table()
    Evt(1,"torch","PmtInBox","", seqs="[]")
                              noname 
                     8cd       351558       [3 ] TO BT SA
                     7cd       111189       [3 ] TO BT SD
                      4d        18047       [2 ] TO AB
                    8ccd        11661       [4 ] TO BT BT SA
                     86d         3040       [3 ] TO SC SA
                     4cd         1736       [3 ] TO BT AB
                    4ccd          884       [4 ] TO BT BT AB
                     8bd          742       [3 ] TO BR SA
                    8c6d          394       [4 ] TO SC BT SA
                     46d          187       [3 ] TO SC AB
                   86ccd          140       [5 ] TO BT BT SC SA
                    7c6d          103       [4 ] TO SC BT SD
                     4bd           63       [3 ] TO BR AB
               8cccccbcd           54       [9 ] TO BT BR BT BT BT BT BT SA
                    866d           33       [4 ] TO SC SC SA


    In [19]: b.history_table()
    Evt(-1,"torch","PmtInBox","", seqs="[]")
                              noname 
                     8cd       337740       [3 ] TO BT SA
                     7cd       106833       [3 ] TO BT SD
 
                    8ccd        23119       [4 ] TO BT BT SA   << CFG4 (vs 11661 for Opticks) : edge scims MO Py MO MO

                      4d        19117       [2 ] TO AB
                     86d         3199       [3 ] TO SC SA
                     4cd         2174       [3 ] TO BT AB
                8ccccbcd         1999       [8 ] TO BT BR BT BT BT BT SA

                    4ccd         1693       [4 ] TO BT BT AB    << CFG4 (vs 884 for Opticks) ... scimmers again
                     8bd         1267       [3 ] TO BR SA       << CFG4 (vs 742 for Opticks) ???

              ccccbccbcd          641       [10] TO BT BR BT BT BR BT BT BT BT
              cbccbccbcd          458       [10] TO BT BR BT BT BR BT BT BR BT
                    8c6d          405       [4 ] TO SC BT SA
                   86ccd          285       [5 ] TO BT BT SC SA
                  8cbbcd          198       [6 ] TO BT BR BR BT SA
                     46d          179       [3 ] TO SC AB
                4ccccbcd          136       [8 ] TO BT BR BT BT BT BT AB
                    7c6d          115       [4 ] TO SC BT SD


Detection to absorb fraction matches efficiency fed in for 380nm::

    A
    375203+118603=493806
    118603./493806.=0.24018

    351558+111189=462747
    111189./462747.=0.24028
    
    B
    337740+106833=444573
    106833./444573.=0.24030

    ggv --surf 6    shows
    380                0.24                0.76     



Agreement when avoid edges (reduce beam radius 100mm to 50mm)
---------------------------------------------------------------

::

    In [1]: run pmt_test.py

                      1:PmtInBox   -1:PmtInBox           c2 
                 8cd       363294       363027             0.10  [3 ] TO BT SA
                 7cd       114962       114826             0.08  [3 ] TO BT SD
                  4d        16924        17342             5.10  [2 ] TO AB
                 86d         2798         2821             0.09  [3 ] TO SC SA
                 4cd         1238         1198             0.66  [3 ] TO BT AB
                8c6d          381          385             0.02  [4 ] TO SC BT SA
                 46d          181          164             0.84  [3 ] TO SC AB
                7c6d          110          111             0.00  [4 ] TO SC BT SD
                 8bd           38           29             1.21  [3 ] TO BR SA
                866d           32           34             0.06  [4 ] TO SC SC SA
               8cc6d           28           24             0.31  [5 ] TO SC BT BT SA
                8b6d            3           22             0.00  [4 ] TO SC BR SA
                466d            0            4             0.00  [4 ] TO SC SC AB
                4c6d            4            2             0.00  [4 ] TO SC BT AB
               8c66d            3            3             0.00  [5 ] TO SC SC BT SA
              8cbc6d            1            2             0.00  [6 ] TO SC BT BR BT SA
               4cc6d            2            1             0.00  [5 ] TO SC BT BT AB
                4b6d            0            1             0.00  [4 ] TO SC BR AB
                86cd            0            1             0.00  [4 ] TO BT SC SA
               8c6cd            1            0             0.00  [5 ] TO BT SC BT SA
                 4bd            0            1             0.00  [3 ] TO BR AB
             8cbc66d            0            1             0.00  [7 ] TO SC SC BT BR BT SA
            8ccbc66d            0            1             0.00  [8 ] TO SC SC BT BR BT BT SA
                          500000       500000         0.77 



Annular beam zenith 0.9,1 with radius 100mm accentuates skimmers
------------------------------------------------------------------

::

    INFO:env.numerics.npy.evt:rx shape (500000, 10, 2, 4) 
                          1:PmtInBox   -1:PmtInBox           c2 
                     8cd       252271       234626           639.45  [3 ] TO BT SA
                    8ccd       117329       121978            90.32  [4 ] TO BT BT SA
                     7cd        79667        74423           178.46  [3 ] TO BT SD
                      4d        21673        21783             0.28  [2 ] TO AB

                8ccccbcd          215        10721         10092.91  [8 ] TO BT BR BT BT BT BT SA

                    4ccd         8795         9246            11.27  [4 ] TO BT BT AB
                     8bd         6985         6500            17.44  [3 ] TO BR SA
                     4cd         5116         4919             3.87  [3 ] TO BT AB
                     86d         3623         3775             3.12  [3 ] TO SC SA

              ccccbccbcd            0         3632          3632.00  [10] TO BT BR BT BT BR BT BT BT BT
              cbccbccbcd            0         2448          2448.00  [10] TO BT BR BT BT BR BT BT BR BT

                   86ccd         1415         1526             4.19  [5 ] TO BT BT SC SA
                  8cbbcd            0          985           985.00  [6 ] TO BT BR BR BT SA
                4ccccbcd            9          766           739.42  [8 ] TO BT BR BT BT BT BT AB
               8cccccbcd          599            0           599.00  [9 ] TO BT BR BT BT BT BT BT SA



Select skimmers with::

    seqs=["TO BT BR BT BT BT BT SA"] 
    see 215/10721 from Opticks/CfG4


Note all the 10k CfG4 skimmers within ~0.4mm of radial range 96.71:97.1mm::
    
    In [25]: a0r.shape, a0r.min(), a0r.max(), b0r.shape,  b0r.min(), b0r.max()
    Out[25]: 
    ((215,),
     97.090540043228771,
     99.972196508157936,
     (10721,),
     96.710607829543051,
     97.092838713641243)


Focus on this range with zenith 0.9671,0.9709 radius 100mm
See very differnt skimmer behavior within that annulus::

                      1:PmtInBox   -1:PmtInBox           c2 
                 8cd       348896            0        348896.00  [3 ] TO BT SA
            8ccccbcd            0       275990        275990.00  [8 ] TO BT BR BT BT BT BT SA
                 7cd       110471            0        110471.00  [3 ] TO BT SD
          ccccbccbcd            0        91652         91652.00  [10] TO BT BR BT BT BR BT BT BT BT
          cbccbccbcd            0        61105         61105.00  [10] TO BT BR BT BT BR BT BT BR BT
                  4d        22079        22258             0.72  [2 ] TO AB
            4ccccbcd            0        19812         19812.00  [8 ] TO BT BR BT BT BT BT AB
                 4cd         6553         6596             0.14  [3 ] TO BT AB
                 8bd         4545         3895            50.06  [3 ] TO BR SA
                 86d         3681         3653             0.11  [3 ] TO SC SA
           86ccccbcd            0         3351          3351.00  [9 ] TO BT BR BT BT BT BT SC SA
          cccccccbcd            1         3060          3057.00  [10] TO BT BR BT BT BT BT BT BT BT
                8ccd         2183         2182             0.00  [4 ] TO BT BT SA
            8ccbbbcd            0         1959          1959.00  [8 ] TO BT BR BR BR BT BT SA
             4cccbcd            0          966           966.00  [7 ] TO BT BR BT BT BT AB
                8c6d          489          420             5.24  [4 ] TO SC BT SA

::

    In [28]: a0r.shape, a0r.min(), a0r.max(), b0r.shape,  b0r.min(), b0r.max()
    Out[28]: 
    ((500000,),
     96.704678194845826,
     97.095561650354725,
     (500000,),
     96.704262560710802,
     97.095972587881292)


After fixing photocathode shape (had omitted startTheta/deltaThera) 
get better agreement for the 0.9671,0.9709 radius 100mm skimmers.
Note still some difference in reflection off Pyrex, maybe dont have same 
polarization distrib between Opticks/CFG4::

                      1:PmtInBox   -1:PmtInBox           c2 
                 8cd       348896       349456             0.45  [3 ] TO BT SA
                 7cd       110471       110351             0.07  [3 ] TO BT SD
                  4d        22079        22131             0.06  [2 ] TO AB
                 4cd         6553         6534             0.03  [3 ] TO BT AB
                 8bd         4545         4002            34.50  [3 ] TO BR SA
                 86d         3681         3819             2.54  [3 ] TO SC SA
                8ccd         2183         2182             0.00  [4 ] TO BT BT SA
                8c6d          489          423             4.78  [4 ] TO SC BT SA
                 4bd          351          333             0.47  [3 ] TO BR AB
                 46d          219          219             0.00  [3 ] TO SC AB
                4ccd          139          153             0.67  [4 ] TO BT BT AB
                7c6d          136          138             0.01  [4 ] TO SC BT SD
                86bd           69           43             6.04  [4 ] TO BR SC SA
               86ccd           42           26             3.76  [5 ] TO BT BT SC SA
                8b6d            5           42            29.13  [4 ] TO SC BR SA
               8cc6d           38           41             0.11  [5 ] TO SC BT BT SA





Attempt to get G4 vis operational to check the geometry

* https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/AllResources/Control/UIcommands/_vis_ASCIITree_.html
* https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/AllResources/Control/UIcommands/_vis_rayTracer_.html

TODO: hookup G4DAE so can export and check the COLLADA

* circularity of this is interesting as can then run Opticks running on 2nd generation geometry 
* also visualization with my tools is much better than G4 


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

    #seqs = ["TO BT BR BT BT BT BT SA"] 
    seqs=[]

    a = Evt(tag="%s" % tag, src="torch", det="PmtInBox", seqs=seqs)
    b = Evt(tag="-%s" % tag , src="torch", det="PmtInBox", seqs=seqs)


    cf = a.history.table.compare(b.history.table)
    print cf


    a0 = a.rpost_(0)
    a0r = np.linalg.norm(a0[:,:2],2,1)

    b0 = b.rpost_(0)
    b0r = np.linalg.norm(b0[:,:2],2,1)


