Progress
=========


.. comment
   
    Needs lots of change to titles and splitting by year to make the toc useful 

    .. contents:: Table of Contents https://bitbucket.org/simoncblyth/opticks/src/master/notes/progress.rst
       :depth: 3


Tips for making yearly summaries
-----------------------------------

1. review commit messages month by month with eg ``o ; ./month.sh -1`` 
2. review presentations month by month, find them with presentation-index
3. while doing the above reviews. compile a list of topics and check 
   that the vast majority of commit messages and presentation pages 
   can fit under the topics : if not add more topics or re-scope the topics

   * for communication purposes do not want too many topics, aim for ~8, 
     think about how they related to each other 


2022 February 
---------------

* CSG_OVERLAP : a multi-INTERSECTION equivalent of the CSG_CONTIGUOUS multi-UNION
   
  * added new compound primitive implemented in CSG/csg_intersect_node.h:intersect_node_overlap
    based on farthest_enter and nearest_exit 
  * list based : so it can mop up intersection nodes into a compound node 
  * https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/OverlapBoxSphere.rst
  * :doc:`/notes/issues/OverlapBoxSphere`
  * TODO: test with more than 2 sub nodes, test the compound prim can work in CSG tree 
  * TODO: think about intersecting with complemented (and unbounded phicut/thetacut/plane nodes) : 
    can CSG_OVERLAP be made to work with such leaves ?
  * potentially be used for general sphere combining intersects  

* thoughts on UK GPU hackathon

  * :doc:`/docs/geometry_testing`
  * https://bitbucket.org/simoncblyth/opticks/src/master/docs/geometry_testing.rst 

* multiunion CSG_CONTIGUOUS : trying to replace large trees with instead small trees with some large compound nodes

  * reorganize intersect and distance functions into three levels tree/node/leaf to avoid recursive CSG_CONTIGUOUS node functions that OptiX disallows 
  * make start at implementing CSG_CONTIGUOUS NMultiUnion as its looking doubtful that balanced trees can be made to work with the CSG intersection
  * generalize NCSG to saving lists of nodes needed by NMultiUnion as well as the normal trees of nodes needed for booleans 

* phicut thetacut

  * unbounded like CSG_THETACUT CSG_PHICUT require csg_tree_intersect special handling to promote MISS into an EXIT at infinity 
    a bit similar to complemented but more involved as depends on the ray direction and starting within the shape,

  * avoiding inconsistent plane side decisions on phicut knife edge by making only one decision appears to avoid the problem of a line of misses along the edge
  * testing phicut intersection with sphere throwing up lots of issues : tails, seam lines 
    
    * handling the cases making the phicut imp much more involved that hoped for
    * unbounded and other complexities makes me question if this is the right approach 

      * https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/GeneralSphereDEV.rst
      * :doc:`/notes/issues/GeneralSphereDEV`

      * perhaps implementing CSG_ICONTIGUOUS (need better name) that does for intersections what CSG_CONTIGUOUS  
        does for unions would allow implementing the general sphere directly with planes and cones 
        rather than with pairs-of-planes and pairs-of-cones 


2022 January 
-------------

* http://simoncblyth.bitbucket.io/env/presentation/opticks_20220115_innovation_in_hep_workshop_hongkong.html
* http://localhost/env/presentation/opticks_20220115_innovation_in_hep_workshop_hongkong.html

* http://simoncblyth.bitbucket.io/env/presentation/opticks_20220118_juno_collaboration_meeting.html
* http://localhost/env/presentation/opticks_20220118_juno_collaboration_meeting.html

  * Opticks 2D slicing
  * PMT mask fix
  * Fastener interfering sub-sub  
  * cutdown PMT issue 
  * render speed check
  * history matching check 
  * XJfixtureConstruction solid : many spurious intersects 
  * XJfixtureConstruction positions : 64 renders : find many overlaps 
  * XJanchorConstruction
  * SJReceiverConstruction

* RTP tangential frame for investigation of some overlaps in global geometry 

* JUNO XJFixtureConstruction (height 4 OR 5 CSG tree composed of many boxes and cylinders)

  * re-modelling at Geant4 level to avoid coincident constituent faces avoids most spurious intersects but very unusually **NOT ALL ARE REMOVED** 
  * :doc:`/notes/issues/spurious-internal-boundary-intersects-in-high-node-count-solids` 
  * https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/spurious-internal-boundary-intersects-in-high-node-count-solids.rst 

    * when CSG tree balancing is not done the problem does not occur
    * find simpler shape BoxFourBoxUnion that exhibits the same issue
    * CSGRecord debugging in CSG proj with newly developed csg_geochain.sh reveals
      issue with the CSG algorithm and balanced trees : could be bug in balancing (changing 
      traversal order for example).  Brief attempts to modify the CSG alg and tree balancing 
      to get them to work together so far not successful. Are more hopeful over the below 
      contiguous union approach, as it simplifies modelling.

    * Issue with balancing motivates a new simpler approach at bit similar to G4MultiUnion that 
      mops up lists of union constituent leaves into lists (not trees) into a new  CSG_CONTIGUOUS primitive node.
      Intersection with contiguous unions of leaves can be implemented more simply than the fully 
      general intersection with CSG trees and the lists of leaves can be stored much more efficiently 
      than with complete binary tree serialization. 

      * :doc:`/notes/issues/csg_contiguous_discontiguos_multiunion`  
      * https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/csg_contiguous_discontiguos_multiunion.rst


* LHCb RICH theta and phi cut G4Sphere  

  * exploring use of CSG intersection with unbounded primitives CSG_PHICUT and CSG_THETACUT
  * https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/LHCb_Rich_Lucas_unclear_sphere_phisegment_issue.rst
  * :doc:`/notes/issues/LHCb_Rich_Lucas_unclear_sphere_phisegment_issue.` 



JUNO Opticks Progress : Short Summary 2021 : 
------------------------------------------------------------

From scratch development of a shared GPU+CPU geometry model enabling 
state-of-the-art NVIDIA OptiX 7 ray tracing of CSG based detector geometries, 
flattened into a two-level structure for optimal performance harnessing ray trace 
dedicated NVIDIA GPU hardware. Development was guided by frequent consultation with NVIDIA engineers. 

JUNO Opticks development, validation and performance testing revealed issues with PMT 
and Fastener geometry, Cerenkov photon generation and PMT parameter services.
This has led to improved geometry modelling, Cerenkov numerical integration
and sampling and PMT services resulting in substantial improvements to the correctness
and performance of the JUNO Geant4 and Opticks based simulations.


Broad headings 2021 progress for a 600 word medium length summary
--------------------------------------------------------------------

* do all commits and presentation pages fit under these headings : or are some more topics needed ?


New OptiX7 Opticks Packages Developed for all new OptiX 7 API 
    short summary mentions only the shared GPU+CPU geometry (ie CSG pkg) as a simplfication and because its the central thing, 
    but in reality for the new model to do anything useful need supporting packages : CSG_GGeo, QUDARap, CSGOptiX
    also changes to existing GGeo was needed to work with the new model 

JUNO Opticks-Geant4 simulation history matching 
    using newly developed G4OpticksRecorder 

JUNO/Opticks Geometry : finding issues and fixing them
    developed new approach to creating 2D planar ray tracing cross sections where geometry visualizations
    are created directly from ray intersections with the geometry : providing an ideal way to check for
    overlapping geometry or spurious intersects arising rom poor geometry modelling   

    * PMTs, several components of support fasteners
    * sometimes source geometry issue, sometimes translation issue 
    * improves CPU sim, enables GPU sim 
    
JUNO PMT Efficiencies : detection efficiency culling
    Development of detection efficiency culling on GPU led to improvements in PMT parameter services 
    and substantially reduced GPU to CPU transfers and CPU memory for hits.
    Worked with young JUNO developers to incorporate the needed changes. 

    * scale CPU memory for hits by a factor of the efficiency

JUNO Cerenkov photon generation : finding issues an fixing them 
    this kinda sprouted off both simulation matching cerenkov wavelength discrep from rejection sampling float/double
    and JUNO issues with Cerenkov wavelength bug that I found and Cerenkov hangs 

Opticks Improvements directed by the needs of users 
    working with Opticks users : bug fixes when applying Opticks geometry tranlation to LHCb RICH geometry, 
    (improving Opticks by applying it to more detectors and coordinating with people to add primitives needed
    for those geometries)
    new primitives working LHCb RICH and LZ students and postdocs

Opticks integration with Geant4 allowing inclusion as example with 1100 distrib
    Opticks updates for Geant4 1070 at start of 2021 and 1100 at end of 2021 and associated Geant4 bug reports from early access to 1100 : that 
    resulted in inclusion of Opticks example in 1100 Geant4 distrib  : working with Geant4 devs

Opticks Publicity : raising awareness of Opticks in the community 
    CAF talk, vCHEP talk, CHEP proceedings paper
    (not development topic, but its an activity that takes time just like others : and needs to be mentioned)

JUNO/Opticks infrastructure integration
    junoenv scripts, CMake machinery, Opticks snapshot releases on github
    (skip this 9th topic)
    

2021 Dec : work with LHCb RICH people on phicut/thetacut primitive
-------------------------------------------------------------------------------------------

* http://simoncblyth.bitbucket.io/env/presentation/opticks_20211223_pre_xmas.html
* http://localhost/env/presentation/opticks_20211223_pre_xmas.html

  * ZSolid applied to Hama and NNVT PMTs
  * Offline CMake integration
  * PolyconeWithMultipleRmin translation 
  * render speed tests following lots of geometry fixes
  * cxr_solid renders
  * speed tables : now much smaller range 
  * LHCb RICH mirror geometry reveals cut sphere bug, quick fixed, 
    plus working with student to add a better way using phicut thetacut primitives  

* rework X4Solid::convertPolycone to handle multiple R_inner, eg base_steel
* found spurious Geant4 and Opticks intersects from flush unions in solidXJfixture and solidXJanchor, these could explain the 0.5 percent history mismatch in ab.sh


2021 Nov : Z-cutting G4VSolid that actually cuts the CSG tree, Geant4 2D cross-sections with (Q->X4)IntersectSolidTest, (Q->X4)IntersectVolumeTest 
-----------------------------------------------------------------------------------------------------------------------------------------------------

* http://simoncblyth.bitbucket.io/env/presentation/opticks_20211117.html
* http://localhost/env/presentation/opticks_20211117.html

  * Hama PMT Solid Breaking Opticks translation 
  * avoid profligate CSG modelling by actually cutting CSG tree  
  * spurious Geant4 intersects
  * Geant4 geometry 2D cross sections
  * new GeoChain package 


* GeoChain testing of the ZCutSolid from j/PMTSIM
* generalize CXS_CEGS center-extent-gensteps config to allow specification of dx:dy:dz offset grids
* pass metadata from the CSGFoundry to the QEvent and persist with it
* check placement new to replace node in a tree
* simplify bookkeeping by extracting zcut from name
* try tree pruning based on crux nodes with XOR INCLUDE and EXCLUDE children
* crux node tree pruning approach seems workable, and handling for no nodes left
* single G4VSolid zcut and tree pruning seems to be working, start expanding GeoChainTest to work with small collections of G4VSolid such as PMTs
* getting PMT PV thru the GeoChain
* move ce-genstep handling down to SEvent for use from X4Intersect aiming for a G4 xxs equivalent to cxs for ground truth comparison of intersects
* X4Intersect scan within GeoChainSolidTest
* possible fix for notes/issues/ellipsoid_not_maintaining_shape_within_boolean_combination.rst in X4Solid::convertDisplacedSolid
* factor off Feature subselection to allow easy swapping between boundary and prim identity partitioning
* remove --gparts_transform_offset to see of that explains the recent removal of the unexpected PMTSim innards 
* notes on need for --gparts_transform_offset see notes/issues/PMT_body_phys_bizarre_innards_confirmed_fixed_by_using_gparts_transform_offset_option.rst
* generalize XZ ZX mp and pv presentation of intersects depending on nx:nz ratio
* X4IntersectVolumeTest by combining intersects from a PV tree of solids with structure transforms 
* remove env switches from the scripts, now controlled based on name suffix interpreted in j/PMTSim::SetEnvironmentSwitches
* thinking about how to special case handle maximally unbalanced trees in fewer passes, suspect can check INCLUDE/EXCLUDE transitions in RPRE-order which is kinda an undo order for typical construction order which is POST-order



2021 Oct : QUDARap : QTex, QCerenkov : new world order simulation atoms, JUNO Fastenener void subtraction reveals CSG limitation, Geant4 1100 property debug
-------------------------------------------------------------------------------------------------------------------------------------------------------------

* http://simoncblyth.bitbucket.io/env/presentation/opticks_autumn_20211019.html
* http://localhost/env/presentation/opticks_autumn_20211019.html

  * Cerenkov : Rejection vs Lookup sampling, S2 integration, ICDF curves, chi2 compare rejection vs lookup samples  
  * Geant4 : Opticks updates for 1100
  * Greater than 500 Opticks unit tests proved useful for pre-release testing of Geant4 11 : several issues 
    immediately discoved simply by running the Opticks unit tests 
  * made the case to avoid proposed changes to Geant4 material property API
  * reported several issues and suggested fixes to Geant4 developers which they eventually accepted
  * NEW 2d planar ray tracing : new geometry testing tools via 2d cross sections 
  * interfering sub-sub bug in fasteners : overcomplex CSG modelling 


* QCerenkov lookup GPU texture testing
* investigate 12 opticks-t fails with unreleased 91072, four might be fixed by X4PropertyMap createNewKey=true 
* ideas for bringing icdf lookup Cerenkov into QSim, need to start by making QSim/qsim into more of an umbrella manager of capable components for sustainable development, also the non-CUDA using QCerenkovIntegral needs to move downwards so it can be formally used pre-cache from CSG_GGeo
* add options --x4nudgeskip --x4pointskip enabling parts of the translation to be skipped for problematic solids, get G4Material name prefix stripping to work again
* down to 0/501 fails with 1100, probably
* change gears to look at CSGOptiXSimulate again, aiming to look into JUNO sticks geometry issue using the planar genstep rendering that kinda combines rendering and simulation
* add SPath::Resolve create_dirs argument 
* potentially serious problem with cxx17/devtoolset-8/cuda-10.1 nvcc
* avoid cxx17 warnings for QUDARap 
* try to avoid cxx17 nvcc templated undefined 
* look into cxx17/devtoolset-8/centos-7/nvcc issue
* simplify QTex by splitting off QTexRotate
* CSG_GGeo dumping to see whats happening with r8 and the ginormous bbox, CSGNode.desc needs complement
* exclude bbox from complemented leaf nodes with only intersect ancestry from contributing to the CSGPrim bbox
* exclude the zero nodes bbox from inclusion into the CSGPrim bbox, giving ridx:8 the expected bbox from p40 of 
 
  * https://simoncblyth.bitbucket.io/env/presentation/juno_opticks_20210712.html 
  * https://localhost/env/presentation/juno_opticks_20210712.html 

* formalizing CSGOptiXSimulate a bit
* add gridscale to concentate the genstep grid on the target geometry
* move CSG/qat4.h,AABB.h down to sysrap/sqat4.h,saabb.h for wider use, preparation for transforming local frame genstep positions/directions into global frame
* 3d histogam of local positions, can potentially sparse-ify genstep locations to make geometry visualization via intersects more efficient
* add pipe cylinder demo solid
* try to get planar ray trace geometry slicing to work with demo geometry
* checking for CSG suprious intersect issue in simple box minus subsub cyl
* new GeoChain pkg for fast iteration geometry debugging by doing all geometry conversions in a single executable
* need to create GVolume/GMergedMesh for the GGeo machinery to work, even with a single G4VSolid 
* look into flakiness of the G4Tubs subsub bug, in some demo solids it did not manifest when expected, add --x4tubsnudgeskip to see effect of switching off the usual inner nudge
* review cylinder intersection techniques to see how difficult it would be to implement pipe cylinder within the primitive
* 758c026a6 - GPts SCount to investigate which solids are failing to be instanced

  * https://bitbucket.org/simoncblyth/opticks/commits/758c026a6

* fix NTreeBuilder issue where some balanced trees are left with a hanging ze placeholder using NTreeBuilder::rootprune, see notes/issues/deep-tree-balancing-unexpected-un-ze.rst
* try cxs for PMTSim::GetSolid checking PMTSim GeoChain integration
* improve NNodeNudger debugging, add primitiveIndexOffset to CSGPrimSpec
* PMTSim_Z test



2021 Sept : Cerenkov S2 integration, Geant4 1100 compat
---------------------------------------------------------

* http://simoncblyth.bitbucket.io/env/presentation/juno_opticks_cerenkov_20210902.html
* http://localhost/env/presentation/juno_opticks_cerenkov_20210902.html

  * G4Cerenkov/G4Cerenkov_modified imprecision, -ve photon yields
  * S2 advantages : more accurate, simpler, faster 
  * QUDARap paired hh/h CPU/GPU headers pattern 
  * keep most GPU code in simple headers : testable from multiple environments 
  * having to use double precision for Cerenkov rejection sampling is a performance problem
  * ana/rindex.py prototype
  * Hama translated ellipsoid bug is visible and not noted in this presentation
  * random aligned Cerenkov comparison
  * PMTAngular : efficiency>1


* encapsulating QCerenkov ICDF into QCK for ease of testing 
* piecewise sympy RINDEX and S2 fails to integrate, perhaps doing each bin separately would work
* replace bugged QCerenkov::GetS2Integral by QCerenkov::GetS2Integral_WithCut, energy sampling vs lookup histo chi2 comparisons in tests/QCKTest.py
* systematic chi2 comparison between QCK energy lookup and sampling
* rejig aiming to avoid problems with Geant4 11 G4MaterialPropertyVector typedef change, by making more use G4PhysicsVector rather than G4PhysicsOrderedFreeVector
* avoid STTF and Opticks dependency on OPTICKS_STTF_PATH envvar using an OKConf::DefaultSTTFPath fallback
* avoid matplotlib.plt at top level for scripts useful remotely as they fail when cannot connect to display
* e2w_reciprocal_check trying to see if the difference can all be explained by CLHEP changed constants
* remove all use of G4PhysicsVector::SetSpline due to Geant4 API change, implicitly assuming the default stays a sensible false 





2021 Aug : Cerenkov S2, QRng, QBuf, integrating the new world packages
------------------------------------------------------------------------

* doing the G4Cerenkov numerical integration directly on s2 = 1 - BetaInverse*BetaInverse/(n*n)  avoids GetAverageNumberOfPhotons going negative when only a small rindex peak is left 
* maximally simple use of skipahead still failing within optixrap/cu/generate.cu but no such problem with qudarap QRngTest
* fix subtle char/unsigned char bug in NP that only manifested when the header length exceeds 128, causing the char values to go negative
* remove GGeo+OpticksCore dependency from QUDARap using NP arrays via CSGFoundry or NP::Load opening door to adding QUDARap dependency to CSGOptiX
* bringing CSG from https://github.com/simoncblyth/CSG/ under the Opticks umbrella
* bring CSG_GGeo from https://github.com/simoncblyth/CSG_GGeo/ under opticks umbrella
* bring CSGOptiX from https://github.com/simoncblyth/CSGOptiX/ under the Opticks umbrella
* start trying to use QUDARap within CSGOptiX for photon seeding via QSeed within CSGOptiX::prepareSimulateParam
* succeed to access gensteps at photon level via seeds with CSGOptiXSimulate in OptiX7Test.cu::simulate 
* fix Cerenkov low wavelength photons, by using the RINDEX range passed by Genstep see notes/issues/cerenkov_wavlength_inconsistency.rst
* QUDARap dependency up from SysRap to OpticksCore for OpticksGenstep_TORCH and eventually for OpticksEvent

  * TODO: probably should move the enum down rather than upping the dependency pkg  

* CSGOptiXSimulate : start checking optix7 raytrace from gensteps, save photons 
* reuse of OptiX7Test.cu intersection code for both rendering and simulation means cannot pre-diddle normals etc..
* thinking about versioning and tagging, turns out OpticksVersionNumber.hh already exists providing OPTICKS_VERSION_NUMBER, see notes/releases-and-versioning.rst 
* retire ancient tests CG4Test OKG4Test that are unclear how to bring into the CManager Geant4 integration approach without lots of additional code
* fix Cerenkov wavelength regression, must reciprocalize otherwise wavelength not properly peaked towards low wavelengths
* forcing use of common en_cross from full bin integral for the partial bin integrals seems to fix slightly non-monotonic issue with cumulative integrals



2021 July : QProp, Cerenkov matching 
--------------------------------------------

* http://simoncblyth.bitbucket.io/env/presentation/juno_opticks_20210712.html
* http://localhost/env/presentation/juno_opticks_20210712.html

  * JUNO Opticks/Geant4 Optical Photon Simulation Matching 
  * matching tools : GtOpticksTool input photon running, photon repetition, G4OpticksRecorder  
  * reemission bookkeeping
  * photon history comparisons (skipping setupCD_Sticks to allow fair comparison)
  * list of fixes for Geant4 implicits, special cases, remove degenerates 
  * scintillation wavelength well matched
  * G4Cerenkov_modified bug  


* http://simoncblyth.bitbucket.io/env/presentation/lz_opticks_optix7_20210727.html
* http://localhost/env/presentation/lz_opticks_optix7_20210727.html

  * QUDARap : pure CUDA photon generation
  * Cerenkov GPU wavelength generation needing double precision



* GDML 2d plot for slow geometry : lAddition
* review recent issues notes to decide what else to present, plus start reviving the comparison plotting machinery
* expt with piecemeal reemission texture giving tenfold bins in the probability extremes
* get the multiresolution scintillation texture approach into the standard workflow, plus a rejig of scintillator persisting to facilitate geant4 processing postcache with original energy domain quantities
* preparing for qudarap QCtx cerenkov wavelength generation, testing boundary tex lookups, move to Wmin_nm Wmax_nm in Cerenkov genstep rather the Pmin Pmax
* ignore gcc attributes warning on QTex template instanciation lines, try non-deprecated cudaMemcpy2DToArray to allow future avoidance of deprecation warning for cudaMemcpyToArray
* integrate QProp/qprop into QCtx/qctx 
* templated QProp/qprop, C++ extern for CUDA calling templated global function QProp.cu _QProp_lookup
* can the cerenkov rejection sampling be converted into an icdf lookup ? What distinguises situations amenable to icdf ?
* Cerenkov photon energy sampling via inverse CDF for many BetaInverse in a 2d texture looks like it might work, prototyping in ana/rindex.py




2021 June : Simulation Matching, workarounds for Geant4 implicits/special-cases   
-----------------------------------------------------------------------------------

* CManager::Get for use from the non-G4Opticks CFG4 S+C processes as now need to declare CManager::BeginOfGenstep before record track steps
* try switching CGenerator to ONESTEP/DYNAMIC recording in all cases
* start updating CerekovMinimal to use G4OpticksRecorder
* rename (getNumPhotons,getNumPhotons2) -> (getNumPhotonsSum,getNumPhotons) Sum is significantly slower for large numbers of gensteps as shown by Zike
* G4OpticksRecorder/CManager/CRecorder/CWriter machinery is working with CKM with KLUDGE-d Scintillation for Geant4 lifecycle testing of REJOINed full photon recording
* allow to override id in CPhotonInfo to allow passing along the ancestral_id thru RE-generations
* make CPhotonInfo::Get fabricate_unlabelled optional as Scinitillation needs not to do it
* review CRecorder/CDebug in preparation for implementing skipping one of the double BT BT observed from Geant4 with very close geometry
* looking for implicit absorption surfaces due to NoRINDEX-to-RINDEX transitions in X4PhysicalVolume::convertImplicitSurfaces_r
* find and add implicit RINDEX_NoRINDEX border surface to the GSurfaceLib in order to mimic implicit Geant4 G4OpBoundaryProcess behavior for such transitions from transparent to opaque
* communicate efficiency collect/cull EC/EX from junoSD_PMT_v2::ProcessHits via G4OpticksRecorder::ProcessHits CManager::ProcessHits
* new qudarap pkg for updated CUDA-centric developments
* split QGen from QRng, use QRng and QTex within QScint to generate reemission wavelengths
* observe an incorrect Pyrex///Pyrex border that should be Water///Pyrex due to degenerate geometry with bbox too similar to be distinguished, this may explain the excess AB and lack of SA due to use of Pyrex ABSLENTH inplace of Water ABSLENGTH 
* increase microStep_mm cut from 0.002 to 0.004 to remove PyPy, see notes/issues/ok_lacks_SI-4BT-SD.rst
* try to fix loss of all surfaces following float to double, see notes/issues/OK_lacking_SD_SA_following_prop_shift.rst 



2021 May : GGeo enhancemends needed for CSG_GGeo conversion, Machinery for Matching : CManager, G4OpticksRecorder
-------------------------------------------------------------------------------------------------------------------

* http://simoncblyth.bitbucket.io/env/presentation/lz_opticks_optix7_20210504.html
* http://localhost/env/presentation/lz_opticks_optix7_20210504.html

  * CSGFoundry model near final : 7, pre-7, CPU testing
  * duplicate 7 environment in pre-7
  * lots of noshow images in the presentation, directory name change perhaps?

* http://simoncblyth.bitbucket.io/env/presentation/opticks_vchep_2021_may19.html
* http://localhost/env/presentation/opticks_vchep_2021_may19.html

  * 1st JUNO Opticks OptiX7 ray trace  
  * efficiency culling decison moved to GPU, reducing CPU hit memory  
  * series of meetings with NVIDIA engineers suggested and organized by LZ. LBNL, NERSC

* http://simoncblyth.bitbucket.io/env/presentation/lz_opticks_optix7_20210518.html
* http://localhost/env/presentation/lz_opticks_optix7_20210518.html

  * debugging CSG_GGeo
  * comparing OptiX 5,6,7 cxr_solid views : last prim bug 
  * Hammamatsu ellipsoid bug is apparent : prior to my realizing it 
 

* GParts enhancements needed for CSGOptiXGGeo (which later becomes  CSG_GGeo)
* fix GParts:add which was omitting to offset the tranform indices in combination, changes motivated by CSGOptiXGGeo
* update to latest https://github.com/simoncblyth/np/ move TTF bitmap annotation from https://github.com/simoncblyth/CSGOptiX to sysrap/SIMG
* d56c432ad - notes on how the renders and tables of https://simoncblyth.bitbucket.io/env/presentation/juno_opticks_20210426.html were created
* https://bitbucket.org/simoncblyth/opticks/commits/d56c432ad
* https://bitbucket.org/simoncblyth/opticks/src/master/docs/misc/snapscan-varying-enabledmergedmesh-option.rst
* G4OpticksRecorder_shakedown
* make OpticksRun event handling symmetric, avoiding createEvent stomping on prior event of the opposite tag
* BeginOfGenstep EndOfGenstep lifecycle tracing in preparation for single-genstep-chunked CRecorder mode
* CRecorder/CWriter debug
* CTrackInfo debug
* handle input photon carrier gensteps in CGenstepCollector::collectOpticksGenstep by passing along OpticksActionControl and Aux


2021 April : machinery for geometry performance scanning, video making for investigating slow geometry
----------------------------------------------------------------------------------------------------------

* http://simoncblyth.bitbucket.io/env/presentation/lz_opticks_optix7_20210406.html
* http://localhost/env/presentation/lz_opticks_optix7_20210406.html

  * first mention of "Foundry" based CSG geometry model : called this because you create everything Solid/Node/Prim 
    via the Foundry and they get contiguously stored into Foundry vectors ready for upload to GPU 
  * "CSG" working  
  * CSG model looks pretty complete at this stage  

* https://simoncblyth.bitbucket.io/env/presentation/juno_opticks_20210426.html
* https://localhost/env/presentation/juno_opticks_20210426.html

* http://simoncblyth.bitbucket.io/env/presentation/juno_opticks_20210426.html
* http://localhost/env/presentation/juno_opticks_20210426.html

  * bash junoenv opticks (replace old pkg based approach, treat opticks like sniper, not Geant4)  
  * gdmlkludge
  * PMTEfficiencyCheck : 1-in-a-million-ce issue : improving efficiency lookup
  * interestingly bad pre-7 OpSnapTest ray trace times : clearly many issues left in geometry, huge time range 
  * fly around fastener movie
  * tds-mu timings  **TODO: redo these with current geom**


* work over in https://github.com/simoncblyth/OptiXTest bringing CSG to OptiX 7 revealed a bug in cone intersects for axial rays from one direction due to an enum 0, fix that issue here too
* arranging for X4PhysicalVolume::convertMaterials X4MaterialTable::Convert to only convert used materials, to match the materials that G4GDML exports
* GDMLKludgeFixMatrixTruncation using xercesc to trim values from truncated matrix attributed to make them able to be parsed
* integrate stb_truetype.h in STTF.hh for annotating ray trace bitmap images
* FlightPath rationalizations and add sliding scale applied across the entire period of the InterpolatedView
* okc/FlightPath using SRenderer protocol base
* rationalize OpTracer snap analogously to FlightPath, getting reusable view control machinery out of OpTracer
* reworked GTree::makeInstanceIdentityBuffer to handle CSG skips 
* snap.py sorting the snap results by render speed and creating table of times
* pin down ordering of GInstancer repeat_candidates using two-level sort to avoid notes/issues/GParts_ordering_difference_on_different_machine.rst
* use SBit::FromString for --enabledmergedmesh/-e for the brevity/flexibility of bitfield control 



2021 March : OptiX7 expts in OptiXTest
-------------------------------------------

* http://simoncblyth.bitbucket.io/env/presentation/opticks_detector_geometry_caf_mar2021.html
* http://localhost/env/presentation/opticks_detector_geometry_caf_mar2021.html

  * detailed look at Opticks geometry approach (prior to OptiX7 CSG developments, but IAS/GAS mentioned) 


* http://simoncblyth.bitbucket.io/env/presentation/lz_opticks_optix7_20210315.html
* http://localhost/env/presentation/lz_opticks_optix7_20210315.html
 
  * resolve the compound GAS issue, by switching to using singe BI containing all AABB
  * intersect_node.h allowing CPU testing  
  * run into identity limitations


OptiXTest : 2021/03/11 -> 2021/05/07
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/simoncblyth/OptiXTest/commits/main
* Geo, Grid, IAS, GAS, Shape, Foundry, Ctx, BI, PIP, PrimSpec

Opticks repo
~~~~~~~~~~~~~~

* curand skipahead
* check for CUDA capable GPU before opticks-full-prepare 
* always save origin.gdml into geocache to try to avoid fails of tests that need GDML when running from geocache created live
* standalone-ish L4CerenkovTest exercising the branches of L4Cerenkov::GetAverageNumberOfPhotons and plotting NumPhotons vs BetaInverse with branches distinguished



2021 Feb : first expts with OptiX 7
---------------------------------------

* http://simoncblyth.bitbucket.io/env/presentation/lz_opticks_optix7_20210208.html
* http://localhost/env/presentation/lz_opticks_optix7_20210208.html

  * very early stage of OptiX 7 expts 

* http://simoncblyth.bitbucket.io/env/presentation/lz_opticks_optix7_20210225.html
* http://localhost/env/presentation/lz_opticks_optix7_20210225.html

  * compound GAS issue : bbox fudge, boxy spheres 


* OptiX 7 learning : getting to grips with the entirely new API : lots of boilerplate, learning by expts, bbox fudge etc 
* OptiX 7 with custom prim not well documented, so useful to get advice from NVIDIA engineers
* Opticks leak checking revealed some significant ones : working with Geant4 people
* unified OptiX pre-7 7 approach for high level 
* SIMG compressed jpg, png rather than uncompressed ppm, for easier remote OptiX 7 work 
* double precision transform handling as new JUNO geometry seems to need it
* review and document G4OpticksHitExtra including how --boundary option feeds into the way_control in GPU context




2021 Jan : Geant4 1070,  first OptiX 7 expts
-------------------------------------------------

* http://simoncblyth.bitbucket.io/env/presentation/opticks_jan2021_juno_sim_review.html
* http://localhost/env/presentation/opticks_jan2021_juno_sim_review.html
  
  * mainly review of 2020 : leap in Opticks awareness
  * Geant4 bug 2305 (optical surfaces) reported 2020-12-22 
  * Geant4 bug 2311 (vector to map API change) reported 2021-01-20
  * about LZ+Opticks+OptiX7 meeting series

* compiletime -> runtime control for way data and angular efficiencies 
* create orientation docs for NVIDIA + LZ colleagues : https://simoncblyth.bitbucket.io/opticks/docs/orientation.html
* attempt to handle the g4 1070 G4LogicalBorderSurface vector to map change, currently without controlling the order
* fixes for g4_1070 including name order sorting of G4LogicalBorderSurfaceTable which has become a std::map, see notes/issues/g4_1070_G4LogicalBorderSurface_vector_to_map_problems.rst
* fix the nhit nhiy inconsistency, the GPU side way buffer was not being resized in OEvent causing the stuck at first events hiy issue, see notes/issues/G4OKTest_fail_from_nhit_nhiy_mismatch.rst
* completing the hits 



2021 : Review of Opticks with OptiX 7 Development History
----------------------------------------------------------

As of the end of 2021 the Opticks packages directly relevant to NVIDIA OptiX 7 are:

CSG
    designed from scratch shared GPU/CPU geometry model  

CSG_GGeo
    conversion of Opticks/GGeo geometries into CSG model 

QUDARap
    simulation building blocks, depending on CUDA : no OptiX dependency 

CSGOptiX
    rendering and simulation with CSG model geometries, drawing on functionality from QUDARap

    Guiding principals:

    * minimize code in CSGOptiX : everything that can be implemented in QUDARap or CSG should be 


Development of these packages started in early 2021 and progressed
through multiple repositories in the first half of 2021 before being 
incorporated into sub-packages of the Opticks repository in summer 2021.


Initial OptiX 7 Expts : 2021/02/04 -> 2021/02/28
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting from scratch, learning the all new NVIDIA OptiX 7 API by simple geometry experiments 

* https://bitbucket.org/simoncblyth/opticks/src/master/examples/UseOptiX7GeometryStandalone/ 2019/11/19
* https://bitbucket.org/simoncblyth/opticks/src/master/examples/UseOptiX7/   2021/02/04 common CMake infrastructure for OptiX pre 7 + 7
* https://bitbucket.org/simoncblyth/opticks/src/master/examples/UseOptiX7GeometryModular/  2021/02/04
* https://bitbucket.org/simoncblyth/opticks/src/master/examples/UseOptiX7GeometryInstanced/  2021/02/04-05
* https://bitbucket.org/simoncblyth/opticks/src/master/examples/UseOptiX7GeometryInstancedGAS/ 2021/02/06-07
* https://bitbucket.org/simoncblyth/opticks/src/master/examples/UseOptiX7GeometryInstancedGASComp/ 2021/02/07-08
* https://bitbucket.org/simoncblyth/opticks/src/master/examples/UseOptiX7GeometryInstancedGASCompDyn/ 2021/02/08-28 
* IAS, GAS, AS, GAS_Builder, IAS_Builder 

OptiXTest : 2021/03/11 -> 2021/05/07
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/simoncblyth/OptiXTest/commits/main
* Geo, Grid, IAS, GAS, Shape, Foundry, Ctx, BI, PIP, PrimSpec

CSG : 2021-04-27 -> 2021-08-19 : after which incorporated into Opticks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/simoncblyth/CSG 
* CSGFoundry, CSGNode, CSGPrim, CSGPrimSpec, CSGView, CSGTarget, CSGScan
 


2020 Focus : Generalizing Opticks to facilitate integration with detector simulation frameworks
--------------------------------------------------------------------------------------------------

Looking at commits in 2020::

    git lg --since 2020-01-01 --until 2020-12-31 

Currently starts from

* https://bitbucket.org/simoncblyth/opticks/commits/?page=60


Remote Working : ssh tunneling + rsync scripts + old CUDA + image handling 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* developed ssh tunneling scripts that avoid repetitive steps to connect to non-publicly accessible remote nodes  
  such as the GPU workstation I use at IHEP

* using scripts that cooperate with other instances of themselves run on the remote node allows 
  repetitive manual remote working operations such as copying to be avoided

* for example the git.py svn.py utilities automate syncing to a remote working copy directory 
  which allows working on a remote node without having to suffer slow editing across network connections
  and also avoids excessive numbers of "sync" commits

* restored Opticks operation with CUDA 9 to allow local testing on my laptop that is limited to this old CUDA version 

* as interactive use of a remote GPU is problematic over the network I improved Opticks image handling allowing writing 
  of annotated images to allow visualization checks to proceed via saving images and tranferring the files

* adopted highly compressed jpg image saving to speedup network transfers between remote GPU workstation at IHEP 
  and laptop in England

   

Generalization of Opticks Build/Install Machinery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

0. Opticks now builds against "foreign" externals using CMAKE_PREFIX_PATH mechanism  
1. opticks-config  machinery (after some expts with other approaches decided to use Boost-CMake-Modules BCM .pc generation capabilities) 
   that allows integration of CMake based Opticks build with non-CMake (CMT) based Offline build  

   * this entailed changes to every one of Opticks 20 packages with build test scripts added for all of them 

2. Opticks as a JUNOenv external 

Housekeeping : Migrate Opticks repo from Mercurial to Git as bitbucket ending support for Mercurial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




Code level integration of Opticks with JUNO Offline using G4Opticks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* PMT Geometry Changes needed for Opticks Translation


2. GDML parsing and matplotlib geometry plotting developed for PMT neck simplifications, removing G4Torus


2020 Dec : tidy up in prep for release candidates, remove old externals, G4 10.6 tests reveals Geant4 bug, way buffer for hit completion 
-------------------------------------------------------------------------------------------------------------------------------------------

* bug link https://bugzilla-geant4.kek.jp/show_bug.cgi?id=2305 
* capture the g4_1062 bordersurface/skinsurface repeated property bug in extg4/tests/G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc
* both skin surface and border surface properties have all values zero in 1062, values ok in 1042 from same gdml
* debugging why Opticks conversion from Geant4 1062 sees all zero efficiency values while Geant4 1042 sees non-zero values
* notes on trying to use devtoolset-9 devtoolset-8 to use newer gcc to install g4 1062 and test G4OpticksTest BUT CUDA 10.1 needed by OptiX 6.5 is not compatible with gcc 9
* pass the opticks_geospecific_options from GDMLAux via BOpticksResource into G4Opticks for the embedded opticks instanciation commandline
* rejig allowing BOpticksResource to run prior to Opticks and OpticksResource instanciation
* remove YoctoGL external, YoctoGLRap pkg and GLTF saving, eliminate the OLD_RESOURCE blocks 
* plugging OpticksEvent leaks, whilst testing with OpticksRunTest 
* add WAY_BUFFER needed for JUNO acrylic point on-the-way recording 
* take at look at nlohmann::json v3.9.1 as potential new external to replace the old one from yoctogl when remove that and GLTF functionality
* remove externals OpenMesh ImplicitMesher and corresponding OpenMeshRap proj and NPY classes and tests 

2020 Nov : async revival, NP transport  
---------------------------------------

* add EFFICIENCY_CULL EFFICIENCY_COLLECT photon flags, plus WITH_DEBUG_BUFFER macro to shake down the inputs to the efficiency cull decision
* investigate slimming PerRayData_propagate prior to adding local f_theta f_phi for sensor efficiency
* switch to 1-based unsigned sensorIndex doubling the maximum number of sensor indices in 2 bytes to 0xffff
* change prefix network header to 16 bytes for xxd clarity, experiment with npy reading and writing over network using async/await in py3 with asyncio, notes on asyncio
* np:think about set_dtype type shifting shape changes, experiment with std::future std::async and NP arrays
* np:migrate all tests and server/client to non-templated NP 
* np:np_client np_server now working with boost::asio async send/recv of NP objects over TCP socket
* Explore cleaner approach to network transport of arrays in np_client/np_server 
  over in np:(https://github.com/simoncblyth/np.git) based on boost::asio only (avoids the need for ZMQ or asio-zmq glue)
* review old ZMQ asio-zmq based numpyserver, implement npy transport with python socket over TCP in bin/npy.py
* liveline config over UDP is restored in OpticksViz using boostrap/BListenUDP
* add BListenUDP m_listen_udp to OpticksViz allowing commands to be passed to the visualization via UDP messages
* incorporate BListenUDP into brap, when boost/asio.hpp header is found with FindBoostAsio
* take a look at the state of the async machinery ZeroMQ BoostAsio used for the old NumpyServer, old asiozmq project seems dead with the 
  version used not operational with current Boost Asio so needs reworking  
* look into bit packing of signed integers, compare using two-complement reinterpretation in SPack::unsigned_as_int with the union trick
* GDML Aux info capture into NMeta json to CGDML

2020 Oct : SensorLib for on GPU angular efficiency culling, ggeo rejig fixing mm0 exceptionalism and adopting geometry model native identity
----------------------------------------------------------------------------------------------------------------------------------------------

* for OSensorLibGeoTest add optickscore/SphereOfTransforms npy/NGLMExt methods to assist creation of a set of 
  transforms to orient and position geometry instances around a sphere with reference directions all pointing at the global origin
* OCtx3dTest reveals OptiX 2d and 3d buffer serialization is column-major contrary to NPY row-major
* GPU uploading SensorLib with OSensorLib based on OCtx (watertight API)
* prepare for setup of angular efficiency via G4Opticks, tested with G4OKTest using MockSensorAngularEfficiencyTable
* remove Assimp external and AssimpRap 
* OpticksIdentity triplet RPO ridx/pidx/oidx 32-bit encoded identifiers : this is the native identity 
  for the Opticks geometry model unlike the straight node index which is needed for Geant4 model  
* start moving all volume GMergedMesh slot 0 (mm0) usage to GNodeLib : aiming to eliminate mm0 special caused
  that has caused 
* start getting python scripts to work with py3  


2020 Sept
----------

* work with Hans (Fermilab Geant4) on changes need for current Geant4 1062 

  * next release of Geant4 will allow genstep collection without changing processes
  * discussing how to change Geant4 API to make Opticks Genstep collection simpler

* IntersectSDF, per-pixel identity, transform lookup, comparison with SDF

* (22) test fail fixes, OPTICKS_PYTHON
* (15) adopt the new FindG4 within Opticks
* (Norfolk)
* (3) examples/UseG4NoOpticks/FindG4.cmake that works with 1042 + 1062

* (1-3)  examples/UseOptiXGeometryInstancedOCtx IntersectSDF
   systematic checking of intersect SDF using "posi" 3d pixel position and geo-identity
   allows to recover local coordinate of every pixel intersect and calculate its distance
   to the surface : which should be within epsilon (so far find within 4e-4)

* (1st) examples/UseOptiXGeometry : using exported oxrap headers allowing Opticks CSG primitives 


2020 Aug
----------

* Opticks ended up in a least 3 Snomass 2021 LoI

* (31) Linux OptiX 6.5 wierd sphere->box bug 
* (30) fixed NPY::concat bug which could have caused much layered tex problems, but still decide to stay with separated 
* (24-30) fighting layered 2d tex, failed : separated ones working OK though
* (24-30) develop OCtx : OptiX 6.5 wrapper with no OptiX types in the interface (thinking about the OptiX 7 future)
* (21st) image annotation for debugging the texture mapping 
* (20th) texture mapping debug : wrapping Earth texture onto sphere 
* (19th) SPPM ImageNPY : expand image handling for 2d texture 
* (18th) examples/UseOptiXTexture examples/UseOptiXTextureLayered examples/UseOptiXTextureLayeredPP explore texturing 
* GNode::getGlobalProgeny

* (17th) notes/performance.rst thoughts : motivated by Sam Eriksen suggestion of an Opticks Hackathon organized with NERSC NVIDIA contacts
* mid-august : neutrino telescope workshop presentation
* (14th) ana/ggeo.py : python transform and bbox access from identity triplet + ana/vtkbboxplt.py checking global bbox
* (8th) notice that current Opticks identity approach needs overhaul to work for global volumes   

  * notes/issues/ggeo-id-for-transform-access.rst 
  * aim to form ggeo-id combining : (mm-index,transform-index-within-mm,volume-within-the-instance) 
  * add globalinstance type of GMergedMesh (kept in additional slot, opposite end to zero), 
    which handles global volumes just like instances : but with only one transform
  * initially only enabled with --globalinstance, from 17th made standard
  * need to fix this in order to be able to convert global coordinates of intersects into local 
    frame coordinates for any volume (this is needed for hit local_pos) 


2020 Aug 13 : SJTU Neutrino Telescope Simulation Workshop
-------------------------------------------------------------

Donglian Xu from SJTU::

    https://indico-tdli.sjtu.edu.cn/event/238/overview

    Tao told us you are in UK now, so we've tentatively scheduled your talk to be
    on ~16:00 of 8.13 Beijing time (9:00am London time). Please let us know if you
    can accept our invitation to speak via ZOOM. If the answer is positive, we will
    be more than happy to reallocate any time slot that works best for you.


2020 July
----------

* (29th) LSExpDetectorConstruction::SetupOpticks 

  * G4Opticks::setGeometry 
  * G4Opticks::getSensorPlacements vector of G4PVPlacement of sensors
  * G4Opticks::setSensorData( sensor_index, ... , pmtCAT, pmtID)  
  * G4Opticks::setSensorAngularEfficiency 
 
  * devise interface that communicates geometry/sensor information without any JUNO assumptions
    (eg on ordering of sensors, or pmtcat relationship to pmtid, or pv.copyNo to pmtid ... all that 
    must be done in detector specific code : as Opticks cannot make JUNO assumptions).
    Done explicitly spelling out the pmtcat and pmtid of each sensor with 
    setSensorData based on the G4PVPlacement returned for each sensor with getSensorPlacements.

  * one assumption : only one volume with a sensitive surface within each repeated geometry instance 

* G4Opticks::getHit 
* revisit PMT identity to work with JUNO copyNo
* iidentity reshaping, 
* remove WITH_AII dead code eradicating AnalyticInstanceIdentity, instead now using InstanceIdentity for both analytic and triangulated geometry
* start on angular efficiency

* (6th) JUNO collab meeting report : next steps 

  * local_pos (play to use new instance identity approach, 
    to give access to the transform to convert global_pos to local_pos)
  * move ce culling to GPU : added texture handling for this 

* add github opticks repo, for making releases : as need tarball to integrate with junoenv 


2020 June
----------

* getting updated geometry to work 
* create GDML matplotlib plotter 
* genstep versioning enum in G4Opticks, motivated by Hans
* polycone neck work over in juno SVN
* svn.py git.py for working copy sync between Linux and Darwin installs
  without huge numbers of "sync" commits
* opticks/junoenv/offline integration done 


2020 May
---------

* pkg-config non-CMake config work ongoing, Linux testing 
* start trying to build opticks against the junoenv externals
* get build against OptiX 5 to work again, for CUDA 9.1 limited macOS laptop
* add higher level API for genstep collection, motivated by Hans (Fermilab Geant4) 
* invited present Opticks at HSF meeting 
  with small audience including several of the core Geant4 developers from CERN  

* HSF meeting link is https://indico.cern.ch/event/921244/ 


May 13::

    Dear Simon,

    in the context of the HSF Simulation Working Group we would like to focus our
    future discussion on accelerators for simulation. 
    We think that the community would profit from the experience of people that
    have already used GPU to tackle their specific simulation environment, from
    their successes as well as the problems they encountered. 

    We are contacting you to ask if you (one of you) would be willing to present
    Opticks and your experience with Nvidia OptiX at the HSF Simulation Working
    Group meeting that we are scheduling for May 27th at 16h00 CET ?

    We will follow it up with one or two meeting in June with lighting talks of R&D
    projects and proposals.

    Please let us know if you can attend the (virtual) meeting and share your
    experience with the HSF community.

    Keep safe,
    Witek, Philippe, Gloria



Some notes on progress:

* bitbucket mercurial to git migrations of ~16 repositories completed

* integration Opticks builds met an issue with multiple CLHEP in junoenv, 
  fixed by preventing the building of the geant4 builtin 
  G4clhep via -DGEANT4_USE_SYSTEM_CLHEP=ON 

* currently working on the geometry translation which happens at BeginOfRun
  where the world pointer is passed to Opticks. 
  The first problem is multiple types of cathodes : I need to generalize 
  Opticks to handle this 


2020 April
-----------

* create parallel universe pkg-config build opticks-config system,  
  supporting use of the Opticks tree of packages without using CMake.
  The pkg-config wave took more than an week to cover all packages.

  * developed using examples/gogo.sh running all the examples/-/go.sh scripts 
  
* introduce "foreign" externals approach, so can build opticks 
  against another packages externals using CMAKE_PREFIX_PATH 
  (boost, clhep, xercesc, g4)
 
* crystalize installation configuration into opticks-setup.sh 
  generated by opticks-setup-generate when running opticks-full



2019 Q4
---------

* looking ahead : start to make some headway with OptiX7 in standalone examples
* making the release a reality, ease of usage via single top level script

2019 Q3
---------

* remove photon limits, photon scanning performance testing with Quadro RTX 8000
* developing the release and sharedcache approach

2019 Q2
---------

* aligned validation scanning over 40 solids
* OptiX 6.0.0 RTX mode, an eventful migration
* get serious with profiling to investigate memory/time issues
* TITAN RTX performance bottleneck investigation and resolution : f64 in the PTX 
* RTX mode showing insane performance with very simple geometry

2019 Q1
----------



2019 Dec
----------

* seminar motivated investigations of CUB and MGPU


2019 Nov
---------

* get down to standalone OptiX7 examples : a different world, GAS, PIP, SBT : using lighthouse2 for high level guidance 

2019 Oct
----------

* investigate some user geometry issues
* bin/opticks-site.bash single top level environment script for used of shared opticks
  release on /cvmfs for example
* fix flags + colors breakages from the cache rejig for release running 
* restrict height of tree exports to avoid huge binary tree crashes


2019 Sept
-----------

* license headers
* glance at OptiX7
* push out the photon ceiling to 100M (then 400M) for Quadro RTX 8000 tests
* develop a binary distribution approach okdist-
* scanning result recording and plotting machinery based on persisted ana/profilesmrytab.py
* avoid permissions problems for running from release by reorganization of caches

2019 August
------------

* travel 


2019 July
-----------

* proposal writing 

* try raising the photon ceiling from 3M to 100M, by generation of curandstate files
  and adoption of dynamic TCURAND for curand randoms on host without having to 
  store enormous files of randoms : only manage to get to 60M   

* Virtual Memory time profiling finds memory bugs, eventually get to plateau profile
* fix CUDA OOM crashes on way to 100M by making production mode zero size the debug buffers 

* fix slow deviation analysis with large files by loop inversion
* adopt np.load mmap_mode to only read slices of large arrays into memory   

* absmry.py for an overview of aligned matching across the 40 solids
* investigate utaildebug idea for decoupling maligned from deviant 

* profilesmryplot.py benchplot.py for results plotting  


2019 June
----------

* revive the tboolean test machinery
* standardize profiling with OK_PROFILE
* RTX mode photon count performance scanning with tboolean-box, > 10,000x at 3M photons only 
* implement proxied in solids from base geometry in tboolean-proxy 
* generalize hit selection functor
* tboolean-proxy scan over 40 JUNO solids, with aligned randoms
* improve python analysis deviation checking 


2019 May 
--------

* Taiwan trip 4/1-8 

  * mulling over sphere tracing SDF implicits as workaround for Torus (guidetube)
    and perhaps optimization for PMT 
  * idea : flatten CSG trees for each solid into SDF functions via CUDA code generation 
    at geometry translation time, compiled into PTX using NVRTC (runtime compilation)  
  * reading on deep learning 
  * working with NEXO user 

* add Linux time/memory profiling : to start investigating the memory hungry translation 
* resume writing 

* develop benchmark machinery and metadata handling
* OptiX 6.0.0 RTX mode debuugging

  * immediate good RTX speedup with triangles
  * analytic started as being 3x slower in RTX mode

    * eventually find the problem as f64 in PTX, even when unused
      causes large performance slowdown with analytic geometry
    
    * eventually using geocache-bench360 reach RTX mode speedups 
      of 3.4x with TITAN RTX (due to its RT cores) and 1.25x with TITAN V 

    * ptx.py : hunting the f64

* develop equirectangular bench360 as a benchmark for raytrace 
  performance using a view that sees all PMTs at once

  * geocache-360 

* start cleanup of optixrap, formerly had all .cu together 
  (mainly because of the CMake setup pain) 

  * now migrating tests from "production" cu into tests/cu 

  * lessons from the RTX performance scare : need to care about whats in the ptx,  
    things permissable in test code are not appropriate in production code 

* use benchmark machinery to measure scaling performance on 8 GPU cluster nodes,
  scales well up to 4 GPUs 
  

2019 April
-----------

* work with user to fix issue on Ubuntu 18.04.2 with gcc 7.3.0 

  * virtualbox proved very handy for reproducing user issues

* failed to get Linux containers LXD working on Precision (snap problem with SELinux)

* updating to OptiX 6.0.0. in a hurry to profit from borrowed NVIDIA RTX, proved eventful

  * NVIDIA driver update somehow conspired with long dormant "sleeper" visualization bug 
    to wakeup at just the wrong moment : causing a week of frenzied debugging 
    due to limited time to borrow the GPU, which eventually bought anyhow : as it had perplexing 
    3x worse RTX performance

  * resulted in a development of quite a few OpenGL + OptiX minimal test case examples 
  * optix::GeometryTriangles 
  * torus causes "misaligned address" crash with OptiX 6.0.0 
  * GDML editing to remove torus using CTreeJUNOTest 
  * ended up buying the RTX GPU 

* developed tarball distribution opticks-dist-*  adopted ORIGIN/.. RPATH
* setup opticks area of cvmfs : for when am ready to make a release
* Opticks installed onto GPU cluster

  * got bad alloc memory issue on lxslc, workaround is to do translation where have more memory 

* raycast benchmark to test NVIDIA RTX 
  

2019 March
-----------

* getting back in saddle after ~5 months hiatus
* redtape : not as bad as last year 
* improve CAlignEngine error handling of missing seq
* getting logging under control 
* Qingdao 2nd Geant4 school in China 3/25-29


2018 October
-------------

* CHEP 2018 proceedings
* viz flightpath enhancements, simple control language 

2018 September
---------------

* CCerenkovGenerator : G4-G4 matching to 1e-8 : so can resume from gensteps, bi-executable convenience
* PMT neck tests : hyperboloid/cone 
* Qingdao seminar ~21st (1.5hr), preparation in env repo
* looking into usage of GPUs for reconstruction

2018 August
-------------

* AB test validating the direct geometry by comparison of geometry NPY buffers

  * plethora of issues surfaces/materials/boundaries/sensors 
  * only way to get a match is to fix problems both in the old and new approaches, 
    even down to the forked assimp external 

* start prototype "user" example project : "CerenkovMinimal" 

  * with SensitiveDetector, Hit collections etc..
  * configured against only the G4OK interface project 
  * used for guiding development of the G4OK package, that
    provides interface between Geant4 user code with an embedded Opticks propagator

* update to Geant4 10.4.2 in preparation for aligned validation 

* adopt two executable with shared geocache pattern for validation,
  (expanding on tboolean using the new capabilities of direct translation of 
   any geometry)

  * 1st executable : anything from a simple Geant4 example to a full detector simulation package 
    with Opticks embedded inside the Geant4 user code using the G4OK package 

  * 2nd executable : operating from geocache+gensteps persisted from the 1st executable 

    * fully instrumented gorilla (records all steps of all photons) OKG4Test executable, 
      with Geant4 embedded inside Opticks 
    * simple purely optical physics : "cleanroom" environment making 
      it possible to attempt alignment of generation + propagation 

* implemented CCerenkovGenerator + CGenstepSource : to allow 2nd executable Geant4 
  to run from gensteps by generating photons at primary level 
  (turning secondary photons from the 1st executable into primaries of the 2nd)

   * **notice this is turning gensteps into first class citizens**

* implemented CAlignEngine for simple switching between pre-cooked RNG streams 



2018 July : discussions with Geant4 members, Linux port, direct translation debug
--------------------------------------------------------------------------------------------------------------

* **discuss proposed extended optical example with Geant4 members**
* **port to Linux CentOS7 Workstation with Volta GPU (NVIDIA Titan V), OptiX 5.1.0, CUDA 9.2**
* **debugging direct geometry translation**

* port python tree balancing to C++ NTreeBalance  
* CHEP + JUNO meetings 
* movie making machinery 
* port the old python opticks-nnt codegen to C++ for the direct route, see x4gen-
  giving code generation of all solids in the geometry 
* refactoring analytic geometry code NCSG, splitting into NCSGData 
* NCSG level persisting 


2018 June : direct Geant4 to Opticks geometry conversion : **simplifies usage**
---------------------------------------------------------------------------------

* simplifies applying Opticks acceleration to any Geant4 geometry

* X4/ExtG4 package for direct conversion of in memory Geant4 model into Opticks GGeo
* YoctoGLRap YOG package for direct conversion from Geant4 into glTF 
* direct fully analytic conversions of G4VSolid into Opticks CSG nnode trees, 
* direct conversions of G4 polgonizations (triangle approximation) into Opticks GMesh 
* adopt integrated approach for analytic and approximate geometry, incorporating 
  both into GGeo rather than the former separate GScene approach 
* direct conversions of materials and surfaces

2018 May : adopt modern CMake target export/import : **simplifies configuration**
-----------------------------------------------------------------------------------

* greatly simplifies Opticks configuration internally and for users

* research modern CMake (3.5+) capabilities for target export/import, find BCM
* adopt Boost CMake Modules (BCM) http://bcm.readthedocs.io/  (proposed for Boost)
  to benefit from modern CMake without the boilerplate 
* much simpler CMakeLists.txt both inside Opticks and in the use of Opticks
  by user code, only need to be concerned with direct dependencies, the tree
  of sub-dependencies is configured  automatically 
* BCM wave over all ~100 CMakeLists.txt took ~10 days
* G4OK project for Geant4 based user code with embedded Opticks, via G4Opticks singleton
* simplify logging OPTICKS_LOG.hh 
* geometry digests to notice changed geometry 

2018 March ; Opticks updated ; macOS High Sierra 10.13.4, Xcode 9.3, CUDA 9.1, OptiX 5.0.1  
---------------------------------------------------------------------------------------------------

* get installation opational onto "new" machine, latest macOS ; High Sierra 10.13.4, Xcode 9.3 with CUDA 9.1 and OptiX 5.0.1


2017 Dec : aligned bi-simulation ~perfect match with simple geometry after fixes 
-----------------------------------------------------------------------------------

* **aligning RNG consumption of GPU/CPU simulations -> trivial validation** 
* **fix polarization + specular reflection discrepancies revealed by aligned running**

* investigate approaches allowing use of the same RNG sequence with Opticks and Geant4

  * near perfect (float precision level) matching with input photons (no reemission yet) 

* add diffuse emitters for testing all angle incidence
* rework specular reflection to match Geant4, fixing polarization discrepancy

2017 Nov ; improved test automation/depth, help LZ user installation 
------------------------------------------------------------------------

* work with LZ user, on AssimpImporter issue
* introduce "--reflectcheat" so photons can stay aligned thru BR/SR 
* direct point-by-point deviation comparisons, for use with common input photons, 
  photons stay aligned until meet RNG (eg from BR/SR/SC/AB)  
* introduce "--testauto" mode that dynamically changes surfaces (simplifying photon histories)
  allowing checks of intersect positions against SDFs without duplicating all the ~50 integration test 
  geometries 
* introduce G4 only universe wrapper volume, to reconcile the boundary-vs-volume 
  model difference between G4 and Opticks
* get bounce truncation to match between Opticks and CFG4, eg for hall-of-mirrors situation
* reimplement the cfg4/CRecorder monolith into many pieces including CG4Ctx for better clarity 
* translation of optical surfaces to Geant4 motivates a reworking of surface geometry
  representation, enhanced persisting simplifies processing and conversion to Geant4  

2017 Oct : emissive test geometry, CPU input photons, Opticks presented to Geant4 plenary
--------------------------------------------------------------------------------------------

* **Opticks presented to plenary session of Geant4 Collaboration Meeting**

* enable any CSG solid to emit test photons, generated CPU side such that 
  Opticks and Geant4 simulations are given exactly the same input photons
* pushed Opticks analytic geometry support thru to okg4, allowing Opticks test geometries to 
  be auto-converted to Geant4 ones ; for okg4 comparisons
* Opticks integration testing ; automate comparison of intersect positions with geometry SDF values 
* debugged Opticks installs on two new Linux distros, Axel desktop, Shandong headless GPU server 
* presenting Opticks to the plenary session of the Geant4 Collaboration Meeting in Australia

2017 Sept : embedded Opticks with Tao Lin, headless GPU server tools at SDU
--------------------------------------------------------------------------------------

* work on some techniques (ffmpeg, okop-snap) to use Opticks on headless GPU server machines, 
  such as combining pure compute raytrace geometry snapshots into mp4 movies
* work with Tao on Opticks/JUNO embedding 
* implement embedded mode of Opticks operation using okop/OpMgr to run  
  inside another process, such as JUNO offline
* introduce okop/OpMgr (pure compute Opticks manager) 
  and usage on headless GPU servers

Big Geometry
~~~~~~~~~~~~~~~

* Eureka ; avoiding having two InstLODCull active regains sanity, with this proviso frustum culling and LOD forking are both working
* InstLODCull simplifications from moving uniform handling to UBO in RContext


2017 Aug : primitives for JUNO : ellipsoid, torus, hyperboloid : solve-quartic troubles
---------------------------------------------------------------------------------------------

* Focus on tricky primitives

Overview
~~~~~~~~~~~

* implemented the primitives needed for JUNO ; torus was difficult, also 
  implemented hyperboloid  ; perhaps we can look into replacing torus with 
  hyperboloid for the PMT (it is much much easier computationally, just quadratics rather than quartics)

* moved analytic geometry processing pre-cache ; so launch time is 
  reduced from ~50 s to < 5 s

* improved OpenGL visualisation performance using 
  instance frustum culling and variable level-of-detail meshes for instances (=PMTs) based on 
  distance to the instance.  These techniques use GPU compute (OpenGL transform feedback) 
  prior to rendering each frame to skip instances that are not visible and replace distant instances with simpler
  geometry.   The improved performance will make it much easier to capture movies…

  As Macs only go to OpenGL 4.1 ; I am limited to techniques available to that version 
  which means no OpenGL compute shaders.  I could of use CUDA interop techniques but 
  if possible it is better to stick with OpenGL for visualisation as that  can work on AMD 
  (and perhaps even Intel) GPUs, meaning much more users can benefit from it.


Solids
~~~~~~~~~

* using doubles for quartic/cubic Solve now seems inevitable, issues are much reduced with doubles but not entirely fixed
* op --j1707 --gltf 3 ; fully analytic raytrace works, not having any triangles saves gobs of GPU memory ; investigate ways to skip torus intersects
* start on hyperbolic hyperboloid of one sheet, hope to model PMT neck with hyperboloid rather than subtracted torus
* torus artifacts gone, after move SolveCubicStrobachPolyFit to use initial gamma using SolveCubicPolyDivision instead of the cursed SolveCubicNumeric

Big Geometry
~~~~~~~~~~~~~~~

* investigate OpenGL LOD and Culling for coping with big geometry
* start checking whats needed to enable instance culling, over in  env- instcull-
* moving analytic GScene into geocache fixes j1707 slow startup, reducing from 50 secs to under 5 secs
* threading LODified meshes thru GGeoLib/GGeoTest
* prep for bringing dynamic GPU LOD fork+frustum culling like env- instcull- into oglrap-, plan to use first class citizen RBuf (of Renderer) to simplify the buffer juggling


2017 July : Solid level bbox Validations and fixes
----------------------------------------------------------------------------------------------------

Solids
~~~~~~~~~

* fix trapezoid misinterpretation (caused impingment) using new unplaced mesh dumping features added to both branches
* fixed cone-z misinterpretation
* added deltaphi imp via CSG_SEGMENT intersect, tboolean-cyslab tboolean-segment
* start on primitives needed for juno1707
* add zcut ellipsoid by using zsphere with scaling adjusted to be 1 for z
* investigate torus artifacts, by looking into cubic approach

Validation ; machinery for comparison G4DAE vs GDML/glTF geometries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* generalize GMeshLib to work in analytic and non-analytic branches, regularize GNodeLib to follow same persistency/reldir pattern
* factor GMeshLib out of GGeo and add pre-placed base solid mesh persisting into/from geocache, see GMeshLibTest and --gmeshlib option
* get nnode_test_cpp.py codegen to work with nconvexpolyhedron primitives defined by planes and bbox

* impingement debug by comparison of GDML/glTF and G4DAE branches
* comparing GMesh bbox between branches, reveals lots of discrepancies ; GScene_compareMeshes.rst
* bbox comparisons are productive ; cone-z misinterp, missing tube deltaphi
* csg composite/prim bbox avoids polyfail noise reduces discrepant meshes to 12 percent
* moving to parsurf bbox, avoids overlarge analytic bbox with complicated CSG trees
* adopting adaptive parsurf_level to reach a parsurf_target number of surface points knocks 5 lvidx down the chart
* complete classification of top 25 parsurf vs g4poly bbox discrepancies, down to 1mm



2017 June : tapering poly dev, tree balancing, build out validation machinery, uncoincidence
----------------------------------------------------------------------------------------------------

Polygonization ; move on HY poly taking too long
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* try subdivide border tris approach to boolean mesh combination, tboolean-hyctrl
* decide to proceed regardless despite subdiv problems, forming a zippering approach

Solids ; analytic bbox combination, tree balancing positivize, ndisc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* nbbox::CombineCSG avoids the crazy large bbox
* CSG.subdepth to attempt tree balancing by rotation, swapping left right of UNION and INTERSECTIN nodes when that would improve balance
* honouring the complement in bbox and sdf, testing with tboolean-positivize 
* checking deep csg trees with tboolean-sc
* nbox::nudge finding coincident surfaces in CSG difference and nudging them to avoid the speckled ghost surface issues
* tboolean-uncoincide for debugging uncoincide failure 
* tboolean-esr ; investigate ESR speckles and pole artifacting, from degenerate cylinder
* add disc primitive tboolean-disc as degenerate cylinder replacement
* make CSG_DISC work as a CSG subobject in boolean expressions by adding otherside intersects and rigidly oriented normals
* mono bileaf CSG tree balancing to handle mixed deep trees, used for unions of cylinders with inners done via subtraction

Structure
~~~~~~~~~~~~

* completed transfer of node identity, boundary and sensor info, from triangulated G4DAE to analytic GDML/glTF branches in GScene
* moving to absolute tree handling in gltf with selection mask gets steering of the branches much closer

Validation ; intersect point SDF, SDF scanning, containment(child surf vs parent SDF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* factor GNodeLib out of GGeo to avoid duplication between GScene and GGeo, aiming to allow comparison of triangulated and analytic node trees
* node names and order from full geometry traversals in analytic and triangulated branches are matching, see ana/nodelib.py
* analytic geometry shakedown begins
* prep automated intersect debug by passing OpticksEvent down from OpticksHub into GScene::debugNodeIntersects

* autoscan all CSG trees looking for internal SDF zeros
* tablulate zero crossing results for all trees, odd crossings almost all unions, no-crossing mostly subtraction
* NScanTest not outside issue fixed via minimum absolute cage delta, all the approx 10 odd crossings CSG trees are cy/cy or cy/co unions in need of uncoincidence nudges

* expand parametric surface coverage to most primitives, for object-object coincidence testing of bbox hinted coincidences
* nnode::getCompositePoints collecting points on composite CSG solid surface using nnode::selectBySDF on the parametric points of the primitives


* NScene::check_surf_points classifying node surface points against parent node SDF reveals many small coincidence/impingement issues 
* avoiding precision issues in node/parent collision (coincidence/impingement) by using parent frame does not make issue go away




2017 May : last primitive (trapezoid/convexpolyhedron), tree balancing, hybrid poly, scene structure
-------------------------------------------------------------------------------------------------------

Solids ; trapezoid, nconvexpolyhedron ; tree balancing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* tboolean-trapezoid ; trapezoid, nconvexpolyhedron 
* nconvexpolyhedron referencing sets of planes just like transforms referencing
* icosahedron check 
* investigate 22 deep CSG solids with binary tree height greater than 3 in DYB near geometry
* implement complemented primitives ; thru the chain from python CSG into npy NCSG, NNode, NPart and on into oxrap csg_intersect_part
* Tubs with inner radius needs an inner nudge, making the inner subtracted cylinder slightly thicker than the outer one
* handling poles and seams in sphere parametrisation 

Polygonization ; hybrid implicit/parametric
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* start HY ; hybrid implicit/parametric polygonization
* parametric primitive meshing with NHybridMesher code HY, test with tboolean-hybrid
* try subdivide border tris approach to boolean mesh combinatio
* adopt centroid splitting succeeds to stay manifold 

Structure ; gltf transport
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* start on GPU scene conversion sc.py, gltf, NScene, GScene
* booting analytic gdml/gltf root from gdml snippets with tgltf-
* repeat candidate finding/using (ie instanced analytic and polygonized subtrees) in NScene/GScene
* integration with oyoctogl- ; for gltf parsing
* tgltf-gdml from oil maxdepth 3, now working with skipped overheight csg nodes (may 20th)



2017 Apr : faster IM poly, lots of primitives, bit twiddle postorder pushes height limit, start with GDML
----------------------------------------------------------------------------------------------------------

Polygonization
~~~~~~~~~~~~~~~~

* integrate implicit mesher IM over a couple of days - much faster than MC or DCS 
  as uses continuation approach and produces prettier meshes
* boot DCS out of Opticks into optional external 
* conclude polygonization fails for cathode and base are a limitation of current poly techniques, 
  need new approach to work with thin volumes, find candidate env-;csgparametric-

Solids ; lots of new primitives ncylinder, nzsphere, ncone, box3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* start adding transform handling to the CSG tree
* add scaling transform support, debug normal transforms
* fix implicit assumption of normalized ray directions bug in sphere intersection 
* introduce python CSG geometry description into tboolean 
* implement ncylinder
* implement nzsphere
* implement ncone 
* implement CSG_BOX3
* polycones as unions of cones and cylinders
* start looking at CSG tree balancing

CSG Engine ; bit twiddle postorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* remove CSG tree height limitation by adoption of bit twiddling postorder, 
  benefiting from morton code experience gained whilst debugging DCS Octree construction

* attempts to use unbounded and open geometry as CSG sub-objects drives home 
  the theory behind CSG - S means SOLID, endcaps are not optional 

Structure ; jump ship to GDML
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* complete conversion of detdesc PMT into NCSG (no uncoincide yet)
* conclude topdown detdesc parse too painful, jump ship to GDML
* GDML parse turns out to be much easier
* implement GDML tree querying to select general subtrees 


2017 Mar : GPU CSG raytracing implementation, SDF modelling, MC and DCS polygonization of CSG trees 
-----------------------------------------------------------------------------------------------------

CSG Engine ; reiteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* moving CSG python prototype to CUDA
* reiteration, tree gymnastics
* CSG stacks in CUDA
* fix a real painful rare bug in tree reiteration  

Solids ; implicit modelling with SDFs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* OpticksCSG unification of type shape codes
* learn geometry modelling with implicit functions, SDFs

Polygonization ; Marching Cubes, Dual Contouring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* start adding polygonization of CSG trees using SDF isosurface extraction
* integrate marching cubes, MC
* integrate dual contouring sample DCS, detour into getting Octree operational in acceptably performant,
  painful at the time, by got real experience of z-order curves, multi-res and morton codes


2017 Feb : GPU CSG raytracing prototyping
-------------------------------------------

CSG Engine ; python prototyping, recursive into iterative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* prototyping GPU CSG in python
* Ulyanov iterative CSG paper pseudocode leads me astray
* GPU binary tree serialization
* adopt XRT boolean lookup tables
* learn how to migrate recursive into iterative


2017 Jan : PSROC presentation, CHEP proceedings
-------------------------------------------------

* CHEP meeting proceedings bulk of the writing  
* start looking at GPU CSG implementation
* PSROC presentation
* PHP


2016 Dec : g4gun, CSG research
----------------------------------

* Paris trip, review
* g4gun 
* CHEP proceedings 
* GPU CSG research 

2016 Nov : G4/Opticks optical physics chisq minimization
---------------------------------------------------------

* scatter debug
* groupvel debug 
* high volume histo chisq numpy comparisons machinery 

2016 Oct : G4/Opticks optical physics chisq minimization
-----------------------------------------------------------

* CHEP meeting 
* DYB optical physics including reemission teleported into cfg4
* CRecorder - for tracing the G4 propagations in Opticks photon record format 
* reemission continuation handling, so G4 recorded propagations can be directly compared to opticks ones
* step-by-step comparisons within the propagations
* tlaser testing 
* tconcentric chisq guided iteration 

2016 Sep : mostly G4/Opticks interop
----------------------------------------

* encapsulate Geant4 into CG4
* multievent handling rejig, looks to be mostly done in optixrap/OEvent.cc
* intro OKMgr and OKG4Mgr the slimmed down replacements for the old App
* Integrated Geant4/Opticks running allowing G4GUN steps to be directly Opticks GPU propagated
* OptiX buffer control worked out for multi-event running, using buffer control flags system  

2016 Aug : OpticksEvent handling, high level app restructure along lines of dependency
-----------------------------------------------------------------------------------------

* migration to OptiX 4.0.0 prompts adoption of buffer control system
* texture handling reworked for 400
* adopt cleaner OpticksEvent layout, with better containment
* add OpticksMode (interop,compute,cfg4) to persisted OpticksEvent metadata
* fix bizarre swarming photon visualization from noise in compressed buffer 
* adjust genstep handling to work with natural (mixed) Scintillation and Cerenkov gensteps
* start app simplification refactoring with low hanging fruit of splitting up classes along 
  lines of dependency - intro OpticksHub (beneath viz, hostside config,geometry,event) 
  and OpticksViz 

* With eye towards future support for fully integrated but layered(for dendency flexibility)
  Opticks/G4 running  

* take sledge hammer to the monolith App, pulling the pieces into separate classes, by dependency
* rework for simultaneous Opticks, G4 simulation - OpticksEvent pairs held in OpticksHub
* integration genstep handoff form G4 to Opticks

2016 Jul : porting to Windows and Linux, Linux interop debug
----------------------------------------------------------------

* migrate logging from boostlog to PLOG, as works better on windows - it also turns out to be better overall
* learning windows symbol export API approachs 
* succeed to get all non-CUDA/Thrust/OptiX packages to compile/run with windows VS2015
* migrate Opticks from env into new opticks repository, mercurial history manipulations
  allowed to bring over the relevant env history into opticks repo
* porting to Linux and multi-user environment in prep for SDU Summer school
* documenting Opticks and organizing the analysis scripts in prep for school
* inconclusive attempts to address Linux interop buffer overwrite issue

2016 Jun : porting to Windows
----------------------------------

* replacing GCache with OpticksResource for wider applicability 
* port externals to Windows/MSYS2/MINGW64
* move to using new repo opticksdata for sharing inputs  
* windows port stymied by g4 not supporting MSYS2/MINGW64  
* rejig to get glew, glfw, imgui, openmesh built and installed on windows with VS2015
* boost too

2016 May : CTests, CFG4 GDML handling, non-GPU photon indexing
------------------------------------------------------------------

* shifts
* getting more CTests to pass 
* bringing more packages into CMake superbuild
* add CGDMLDetector
* workaround lack of material MPT in vintage GDML, using G4DAE info 
* integrating with G4 using CG4 
* CPU Indexer and Sparse, for non-GPU node indexing
* rework event data handling into OpticksEvent

2016 Apr : build structure make to CMake superbuild, spawn Opticks repo
---------------------------------------------------------------------------

* GTC
* factoring usage of OptiX to provide functionality on non-CUDA/OptiX capable nodes
* CMake superbuild with CTests 
* external get/build/install scripts
* prep for spawning Opticks repository 

2016 Mar : Opticks/G4 PMT matching, GPU textures, making movie 
------------------------------------------------------------------

* resolved PMT skimmer BR BR vs BR BT issue - turned out to be Opticks TIR bug
* PmtInBox step-by-step record distribution chi2 comparison 
* rejig material/surface/boundary buffer layout to match OptiX tex2d float4 textures, with wavelength samples and float4 at the tip of the array serialization
* Dayabay presentation
* screen capture movie making 
* GTC presentation

2016 Feb : partitioned analytic geometry, compositing raytrace and rasterized viz
-----------------------------------------------------------------------------------

* create analytic geometry description of Dayabay PMT 
* PMTInBox debugging
* compositing OptiX raytrace with OpenGL rasterized


2016 Jan : Bookmarks, viewpoint animation, presentations
--------------------------------------------------------------------

* rework Bookmarks, split off state handling into NState
* add InterpolatedView for viewpoint animation 
* JUNO meeting presentation 
* PSROC meeting presentation 


2015 : First year of Opticks, based on NVIDIA OptiX
-----------------------------------------------------

**Year Executive Summary**

Develop Opticks based on the NVIDIA OptiX ray tracing framework, replacing Chroma.
Achieve match between Opticks and Geant4 for simple geometries with speedup 
factor of 200x with a mobile GPU. Performance factor expected to exceed 1000x 
with multi-GPU workstations.  

**Year Summary**

* realize lack of multi-GPU is showstopper for Chroma 
* find that NVIDIA OptiX ray tracing framework exposes accelerated geometry intersection 
* develop Opticks (~15 C++ packages: GGeo, AssimpWrap, OptiXRap, ThrustRap, OGLRap,...) 
  built around NVIDIA OptiX to replace Chroma : effectively 
  recreating part of the Geant4 context on the GPU 
* port Geant4 optical physics into Opticks
* achieve match between Opticks and Geant4 for simple geometries, 
  with speedup factor of 200x with laptop GPU with only 384 cores


2015 Dec : matching against theory for prism, rainbow, 200x performance with 384 cores
------------------------------------------------------------------------------------------

* prism test with Plankian light source using GPU texture
* rainbow comparisons against expectation : achieve Geant4/Opticks match with rainbow geometry
* cfg4, new package for comparison against standalone geant4
* cfg4 G4StepPoint recording - creating opticks format photon/step/history records with cfg4-
* Opticks/Geant4 rainbow scatter matching achieved
* enable loading of photons/records into ggv, in pricipal enables visualizing both Opticks and G4 cfg4- generated/propagated events on non-CUDA machines
* revive compute mode reveals 200x faster performance than Geant4 with only 384 CUDA cores 

2015 Nov : refactor for dynamic boundaries, Fresnel reflection matching, PMT uncoincidence
---------------------------------------------------------------------------------------------

* overhaul material/surface/boundary handling to allow dynamic boundary creation post geocache
  (ie geometry configurable from commandline)
* implement dynamic test geometry creation controlled by commandline argument, using "--test" option 
* npy analysis for Fresnel reflection testing
* adopt more rational PMT partitioning surfaces (not a direct translation)

2015 Oct : meshfixing, instanced identity, start analytic partitioning
--------------------------------------------------------------------------

* vertex deduping as standard  
* IAV and OAV mesh surgery
* sensor handling
* identity with instancing
* develop analytic PMT approach : via detdesc parsing and geometrical partitioning
* flexible boundary creation

2015 Sep : thrust for GPU resident photons, OpenMesh for meshfixing
--------------------------------------------------------------------

* use interop Thrust/CUDA/OptiX to make **photons fully GPU resident**, eliminating overheads
* finally(?) nail majority of CUDA/Thrust/OpenGL/OptiX interop issues
* add Torch for testing
* investigate bad material for upwards going photons, find cause is bad geometry
* uncover issue with DYB cleaved meshes, develop fix using OpenMesh

2015 Aug : big geometry handling with Instancing
--------------------------------------------------

* OptiX instancing 
* intro BBox standins
* Thrust interop

2015 Jul : photon index, propagation histories, Linux port
-----------------------------------------------------------

* photon indexing with Thrust
* verifying ThrustIndex by comparison against the much slower SequenceNPY
* auto-finding repeated geometry assemblies by progeny transform/mesh-index digests in GTreeCheck
* interim Linux compatibility working with Tao
* 4-GPU machine testing with Tao
* OpenGL/OptiX instancing 
* trying to get JUNO (big) geometry to work with instancing 
* computeTest timings for Juno Scintillation as vary CUDA core counts

2015 Jun : develop compressed photon record, learn Thrust 
------------------------------------------------------------

* Cerenkov and Scintillation generated photons match to Geant4 achieved within OptiX machinery
* implement Fresnel reflection/refraction with OptiX

* develop highly compressed photon records
* ViewNPY machinery for OpenGL uploading 
* get animation working 
* add GOpticalSurface, for transporting surface props thru Assimp/AssimpWrap into GGeo
* learning Thrust
* OptiX 3.8 , CUDA 7.0 update 


2015 May : GPU textures for materials, geocache, ImGui
---------------------------------------------------------

* bring NPY persistency to GGeo : introducing the geocache
* implement geocache loading to avoid XML parsing on every launch 
  (turned out to be a luxury for DayaBay [saving only a few seconds per launch], 
   but 6 months hence it is a necessity for JUNO [saving several minutes for every launch])
* GSubstanceLib infrastructure
* start bringing materials to GPU via textures
* material code translation in Lookup
* reemission handling, inverse CDF texture creation
* Cerenkov and Scintillation generated photons match to Geant4 achieved within OptiX machinery
* pick ImGui immediate mode GUI renderer
* GUI adoption by the oglrap classes
* prepare presentation 

  * Why not Chroma ? Progress report on migrating to OptiX 
  * http://simoncblyth.bitbucket.io/env/presentation/optical_photon_simulation_with_nvidia_optix.html

2015 April 
------------

* reuse NumpyServer infrastructure for UDP messaging allowing live reconfig of objects 
  with boost::program_option text parsing 
* add quaternion Trackball for interactive control
* avoid duplication with OptiXRap
* arrange OptiX output buffer to be a PBO which is rendered as texture by OpenGL
* create OpenGL visualization package: OGLRap (Prog/Shdr infrastructure) and OptiXEngine ray tracer
* OptiXEngine starting point for propagation, previously focussed on OptiX ray tracing 
* ported Cerenkov generation from Chroma to OptiX

2015 March 
-----------

* encounter OptiX/cuRAND resource issue, workaround using pure CUDA to initialize and persist state
* fail to find suitable C++ higher level OpenGL package, start own oglrap- on top of GLFW, GLEW
* integrate ZMQ messaging with NPY serialization using Boost.ASIO ASIO-ZMQ to create NumpyServer


2015 February 
----------------

* fork Assimp https://github.com/simoncblyth/assimp/commits/master
* benchmarks with using CUDA_VISIBLE_DEVICES to control how many K20m GPUs are used
* fork Assimp for Opticks geometry loading
* test OptiX scaling with IHEP GPU machine
* great GGeo package, intermediary geometry model
* experiment with GPU textures for interpolated material property access 

2015 January 
-------------

* https://bitbucket.org/simoncblyth/env/src/2373bb7245ca3c1b8fb06718d4add402805eab93/presentation/gpu_accelerated_geant4_simulation.txt?fileviewer=file-view-default
* https://simoncblyth.bitbucket.io/env/presentation/gpu_accelerated_geant4_simulation.html

  * G4 Geometry model implications 
  * G4DAE Geometry Exporter
  * G4DAEChroma bridge

* realize lack of multi-GPU support is showstopper for Chroma
* find NVIDIA OptiX, initial tests suggest drastically 50x faster than Chroma
* first look at OptiX immediately after making the above presentation
* fork Assimp for geometry loading into GGeo model
* succeed to strike geometry with Assimp and OptiX


2014 : Year of G4DAEChroma : Geant4 to Chroma runtime bridge
----------------------------------------------------------------

**Year Executive Summary**

Get G4DAE exported geometries into Chroma and integrate Geant4 
and Chroma event data via G4DAEChroma runtime bridge.  

**Year Summary**

* Get Chroma to operate with G4DAE exported geometries. 
* Develop G4DAEView visualization using CUDA/OpenGL interoperation techniques
  and OpenGL shaders for geometry and photon visualization.
* Develop G4DAEChroma runtime bridge interfacing Geant4 with external optical photon propagation.
* Realize that photon transport is too large an overhead, so implement GPU Scintillation/Cerenkov
  generation within Chroma based in transported gensteps

**December 2014**

* realize photon transport has too much overhead, "gensteps" are born 
* implement Cerenkov and Scintillation step transport and photon generation on GPU 

**October/November 2014**

* develop G4DAEChroma (photon transport over ZMQ): Geant4 to Chroma runtime bridge 

**September 2014**

* present G4DAE geometry exporter at: 19th Geant4 Collaboration Meeting, Okinawa, Sept 2014

**August 2014**

* export Daya Bay PMT identifiers
* develop non-graphical propagator

**June/July 2014**

* create GLSL shader visualizations of photon propagations 
* reemission debug 

**May 2014**

* develop ChromaZMQRoot approach to transporting photons from NuWa to Chroma 

**Mar-Apr 2014**

* forked Chroma, adding G4DAE integration and efficient interop buffers
* develop g4daeview geometry viewer (based on pyopengl, glumpy)  

**Jan-Feb 2014**

* December 16th 2013 : purchase Macbook Pro laptop GPU: NVIDIA GeForce GT 750M 
  (in Hong Kong while on trip for DayaBay shifts) 
* integrate G4DAE geometry with Chroma 


2013 Aug-Dec : Initial look, G4DAE geometry exporter 
-----------------------------------------------------

Develop G4DAE Geant4 exporter that liberates tesselated G4 geometries
into COLLADA DAE files, including all material and surface properties.

* study Geant4 and Chroma optical photon propagation
* develop C++ Geant4 geometry exporter : G4DAE 
* experiment with geometry visualizations (webgl, meshlab)

December 2013 (G4DAE visualization 2nd try: meshlab)
-------------------------------------------------------

* meshlab- hijacked for COLLADA viewing
* meshlab COLLADA import terribly slow, and meshlab code is a real mess 
* forked meshlab https://bitbucket.org/simoncblyth/meshlab
* investigate openscenegraph- colladadom- osg-
  (clearly decided meshlab far to messy to be a basis for anything)

November 2013 (G4DAE visualization 1st try: webgl)
----------------------------------------------------

* webgl threejs daeserver.py 

Status report coins G4DAE, were validating G4DAE against VRML2

* https://bitbucket.org/simoncblyth/env/src/9f0c188a8bb2042eb9ad58d95dadf9338e08c634/muon_simulation/nov2013/nov2013_gpu_nuwa.txt?fileviewer=file-view-default

Oct 2013 (G4DAE approach born)
--------------------------------

* translate Geant4 volume tree into COLLADA DAE
* webpy server of DAE subtrees

Sept 2013
----------

* sqlite3 based debugging of VRML exports 
* try reality player VRML viewer
* end Sept, start looking into GDML and COLLADA pycollada-
 
Although VRML was a dead end, it provided the G4Polyhedron 
triangulation approach used later in G4DAE.

Sep 24 2013
~~~~~~~~~~~~~

The only real progress so far is with the geometry aspect
where I have made Geant4 exports of VRML2 and GDML
versions of the Dayabay geometry and examined how those
exporters operate. From that experience, I think that
development of a Geant4 Collada exporter (a common 3D file format)
is the most convenient way to proceed in order to
extract the Chroma needed triangles+materials from Geant4.
For developing the new exporter, I need to learn the relevant
parts of the Collada format and can borrow much code
from the VRML2 and GDML exporters.

August 2013 (geometry exporter study)
---------------------------------------

* Geant4 Muon simulation profiling, fast-
* studing Geant4 and Geant4/Chroma integration
* looking into Geant4 exporters and visualization
* study meshlab-
* trying VRML exports
* try blender
* study Chroma operation

* https://bitbucket.org/simoncblyth/env/commits/e7cb3c9353775de29bade841b171f7a7682cbe9c


July 2013 (surveying landscape)
-----------------------------------

Looked into muon simulation optimization techniques

* photon weighting




Notes
----------

Early years copied here from okc-vi there is more detail over there than here.


Updating 
----------

review presentations 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update and open the internal index of presentations with *index.sh* (env/bin/index.sh)

* http://localhost/env/presentation/index.html
* http://simoncblyth.bitbucket.io/env/presentation/index.html
* to update the descriptions appearing in the index page, update the metadata description fields in the .txt sources, 
  no need to update the html of the presentations unless you find bugs that warrant "explanations from the future"


review commit messages across multiple repos on bitbucket and github plus JUNO svn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use *git-;git-month n* (from env-) to review commits for the numbered month, 
negative n eg -12 for December of last year.
To see diff details of a commit listed by git-month::

    git log -n1 -p ab5f1feb3

Select repesentative/informative commit messages for inclusion into monthly 
progress notes above. 

For SVN see svn-offline-blyth using::

   svn log -v --search $USER


`notes-progress` summaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This progress text is parsed by `bin/progress.py` in preparation of ``notes-progress`` summaries, 
to work with that parser follow some rules:

1. title lines have a colon after the date, suppress a title by using semi-colon instead
2. other lines have no colons
3. bullet lines to be included in the summary should be in bold




