Progress
=========

Use *hg-month n* to review commits for the numbered month, 
negative n eg -12 for December of last year.

Dont use colons in section text for easy title grepping.



2015 May : GPU textures for materials, geocache, ImGui
----------------------------------------------------------

* start bringing materials to GPU via textures
* introduce the geocache
* material code translation in Lookup
* adopt ImGui

2015 Jun : develop compressed photon record, learn Thrust 
------------------------------------------------------------

* develop highly compressed photon records
* ViewNPY machinery for OpenGL uploading 
* get animation working 
* add GOpticalSurface, for transporting surface props thru Assimp/AssimpWrap into GGeo
* learning Thrust
* OptiX 3.8 , CUDA 7.0 update 

2015 Jul : photon index, propagation histories, Linux port
-----------------------------------------------------------

* photon indexing with Thrust
* verifying ThrustIndex by comparison against the much slower SequenceNPY
* auto-finding repeated geometry assemblies by progeny transform/mesh-index digests in GTreeCheck
* interim Linux compatibility working with Tao
* 4-GPU machine testing with Tao
* OpenGL instancing 
* trying to get JUNO geometry to work
* computeTest timings for Juno Scintillation as vary CUDA core counts

2015 Aug : big geometry handling with Instancing
--------------------------------------------------

* OptiX instancing 
* intro BBox standins
* Thrust interop

2015 Sep : thrust for GPU resident photons, OpenMesh for meshfixing
--------------------------------------------------------------------

* use interop Thrust/CUDA/OptiX to make photons fully GPU resident, eliminating overheads
* add Torch for testing
* investigate bad material for upwards going photons, find cause is bad geometry
* integrate OpenMesh to enable mesh fixing


2015 Oct : meshfixing, instanced identity, start analytic partitioning
--------------------------------------------------------------------------

* vertex deduping as standard  
* IAV and OAV mesh surgery
* sensor handling
* identity with instancing
* analytic geometry description of DYB PMT via detdesc parsing and geometrical partitioning
* flexible boundary creation

2015 Nov : refactor for dynamic boundaries, Fresnel reflection matching, PMT uncoincidence
---------------------------------------------------------------------------------------------

* overhaul material/surface/boundary handling to allow dynamic boundary creation post geocache
* implement dynamic test geometry creation controlled by commandline argument, using "--test" option 
* npy analysis for Fresnel reflection testing
* adopt more rational PMT partitioning surfaces (not a direct translation)

2015 Dec : matching against theory for prism, rainbow
----------------------------------------------------------

* prism test with Plankian light source using GPU texture
* rainbow comparisons against expectation
* cfg4, new package for comparison against standalone geant4
* cfg4 G4StepPoint recording - creating opticks format photon/step/history records with cfg4-
* Opticks/Geant4 rainbow scatter matching achieved
* enable loading of photons/records into ggv, in pricipal enables visualizing both Opticks and G4 cfg4- generated/propagated events on non-CUDA machines
* begin revival of compute mode

2016 Jan : Bookmarks, viewpoint animation, presentations
--------------------------------------------------------------------

* rework Bookmarks, split off state handling into NState
* add InterpolatedView for viewpoint animation 
* JUNO meeting presentation 
* PSROC meeting presentation 

2016 Feb : partitioned analytic geometry, compositing raytrace and rasterized viz
-----------------------------------------------------------------------------------

* create analytic geometry description of Dayabay PMT 
* PMTInBox debugging
* compositing OptiX raytrace with OpenGL rasterized

2016 Mar : Opticks/G4 PMT matching, GPU textures, making movie 
------------------------------------------------------------------

* resolved PMT skimmer BR BR vs BR BT issue - turned out to be Opticks TIR bug
* PmtInBox step-by-step record distribution chi2 comparison 
* rejig material/surface/boundary buffer layout to match OptiX tex2d float4 textures, with wavelength samples and float4 at the tip of the array serialization
* Dayabay presentation
* screen capture movie making 
* GTC presentation

2016 Apr : build structure make to CMake superbuild, spawn Opticks repo
---------------------------------------------------------------------------

* GTC
* factoring usage of OptiX to provide functionality on non-CUDA/OptiX capable nodes
* CMake superbuild with CTests 
* external get/build/install scripts
* prep for spawning Opticks repository 

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

2016 Jun : porting to Windows
----------------------------------

* replacing GCache with OpticksResource for wider applicability 
* port externals to Windows/MSYS2/MINGW64
* move to using new repo opticksdata for sharing inputs  
* windows port stymied by g4 not supporting MSYS2/MINGW64  
* rejig to get glew, glfw, imgui, openmesh built and installed on windows with VS2015
* boost too

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


2016 Sep : mostly G4/Opticks interop
----------------------------------------

* encapsulate Geant4 into CG4
* multievent handling rejig, looks to be mostly done in optixrap/OEvent.cc
* intro OKMgr and OKG4Mgr the slimmed down replacements for the old App
* Integrated Geant4/Opticks running allowing G4GUN steps to be directly Opticks GPU propagated
* OptiX buffer control worked out for multi-event running, using buffer control flags system  


2016 Oct : G4/Opticks optical physics chisq minimization
-----------------------------------------------------------

* CHEP meeting 
* DYB optical physics including reemission teleported into cfg4
* CRecorder - for tracing the G4 propagations in Opticks photon record format 
* reemission continuation handling, so G4 recorded propagations can be directly compared to opticks ones
* step-by-step comparisons within the propagations
* tlaser testing 
* tconcentric chisq guided iteration 

2016 Nov : G4/Opticks optical physics chisq minimization
---------------------------------------------------------

* scatter debug
* groupvel debug 
* high volume histo chisq numpy comparisons machinery 

2016 Dec : g4gun, CSG research
----------------------------------

* Paris trip, review
* g4gun 
* CHEP proceedings 
* GPU CSG research 

2017 Jan : presentations, proceedings, holidays
-------------------------------------------------

* CHEP meeting proceedings bulk of the writing  
* start looking at GPU CSG implementation
* PSROC presentation
* PHP

2017 Feb : GPU CSG raytracing prototyping
-------------------------------------------

* prototyping GPU CSG in python
* Ulyanov iterative CSG paper pseudocode leads me astray
* GPU binary tree serialization
* adopt XRT boolean lookup tables
* learn how to migate recursive into iterative

2017 Mar : GPU CSG raytracing implementation, SDF modelling, MC and DCS polygonization of CSG trees 
-----------------------------------------------------------------------------------------------------

* moving CSG python prototype to CUDA
* reiteration, tree gymnastics
* CSG stacks in CUDA
* fix a real painful rare bug in tree reiteration  
* OpticksCSG unification of type shape codes

* learn geometry modelling with implicit functions, SDFs
* start adding polygonization of CSG trees using SDF isosurface extraction
* integrate marching cubes, MC
* integrate dual contouring sample DCS, detour into getting Octree operational in acceptably performant,
  painful at the time, by got real experience of z-order curves, multi-res and morton codes


2017 Apr : better polygonization with IM, applying GPU CSG to detdesc and GDML, adding primitives
----------------------------------------------------------------------------------------------------

* integrate implicit mesher IM over a couple of days - much faster than MC or DCS 
  as uses continuation approach and produces prettier meshes
* boot DCS out of Opticks into optional external 
* start adding transform handling to the CSG tree
* add scaling transform support, debug normal transforms
* fix implicit assumption of normalized ray directions bug in sphere intersection 
* introduce python CSG geometry description into tboolean 
* remove CSG tree height limitation by adoption of bit twiddling postorder, 
  benefiting from morton code experience gained whilst debugging DCS Octree construction

* implement ncylinder
* implement nzsphere

* attempts to use unbounded and open geometry as CSG sub-objects drives home 
  the theory behind CSG - S means SOLID, endcaps are not optional 

* conclude polygonization fails for cathode and base are a limitation of current poly techniques, 
  need new approach to work with thin volumes, find candidate env-;csgparametric-

* complete conversion of detdesc PMT into NCSG (no uncoincide yet)

* conclude topdown detdesc parse too painful, jump ship to GDML
* GDML parse turns out to be much easier
* implement GDML tree querying to select general subtrees 

  



