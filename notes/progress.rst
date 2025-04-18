Progress
=========


.. contents:: Table of Contents : https://bitbucket.org/simoncblyth/opticks/src/master/notes/progress.rst
    :depth: 3


**Tips for making monthly summaries + next presentation**

* https://bitbucket.org/simoncblyth/opticks/src/master/notes/progress.rst

1. review commit messages month by month. Although these progress notes 
   mostly cover Opticks it is still necessary to review all the main repositories 
   to get the full picture::

     ~/o/month.sh -12  # Dec last year  
     ~/j/month.sh -12
     ~/n/month.sh -12
     ~/e/month.sh -12
    
     SDIR=$JUNOTOP/junosw ~/o/month.sh 5   ## jo  

   * **select small-ish fraction of informative/representative commit messages** 
   * include the selected commit messages below : intention is to give broad strokes overview (not details)
   * include the hash only where it is particularly informative (eg `git l  a81983c4a^-1` )
   * compose month titles if possible
   

2. review presentations month by month, find them with presentation-index

   * include links to presentations in below timeline, with highlight slide titles
   * cherry pick sets of slides for next presentation

3. while doing the above reviews. compile a list of topics and check 
   that the vast majority of commit messages and presentation pages 
   can fit under the topics : if not add more topics or re-scope the topics

   * for communication purposes do not want too many topics, aim for ~8, 
     think about how they are related to each other 



2025 Feb
---------

* o : 02/20 : switch OFF DEBUG_TAG from sysrap/CMakeLists.txt avoiding three QSimTest_ALL.sh fails from stagr usage

2025 Jan : testing the migrations revealed reversions, now fixed
-------------------------------------------------------------------

* o : 01/09 : fix reversion of no hits array on B side following multi-launch impl as was trying to get photons from topfold before the gather, fix by getting from fold

2024 Dec : migrate from XORWOW to Philox removing need for curandState files, add out-of-core to QSim::simulate -> Opticks unlimited by VRAM or state files
------------------------------------------------------------------------------------------------------------------------------------------------------------

* o : 12/17 : adjust QSim::simulate profile stamps for multi-launch running
* o : 12/13 : complete initial migration from curand XORWOW to Philox  


2024 Nov : generalize translation to use listnode with unchanged source geometry 
-------------------------------------------------------------------------------------

* o : 11/05 : enable geometry translation to create smaller trees with listnode using sn::CreateSmallerTreeWithListNode rather than requiring G4MultiUnion in the G4 geometry, to avoid G4 voxelization SEGV


2024 Oct : performance test comparing 1st and 3rd gen RTX with full JUNO geometry, conference+workshop reports
----------------------------------------------------------------------------------------------------------------

* https://simoncblyth.bitbucket.io/env/presentation/opticks_20241025_montreal_nEXO_light_simulations_workshop.html
* https://simoncblyth.bitbucket.io/env/presentation/opticks_20241021_krakow_chep2024.html

* o : 10/15 : md5sum checks show that the dud QCurandState were all zeros, plus that these states are consistent between the CUDA versions in use
* o : 10/14 : investigate 3 in 10M photons issue of lpmtid being > 17612, avoid the CUDA_Exception 
* o : 10/13 : update the progress notes 


2024 Sep : fixed intersect issue with CUDA 12.4 OptiX 8 compat : RTX 3rd gen test
----------------------------------------------------------------------------------

* o : 09/30 : enable skipahead as standard, configured with OPTICKS_EVENT_SKIPAHEAD. Switch to NPX::ArrayFromData from NPX::Make
* o : 09/14 : add more efficient CombinedArray serialize/import to S4MaterialPropertyVector.h and std::vector of std::array serialize/import to NPX.h 
* o : 09/12 : avoid need to keep changing mode in elv.sh using emm.sh that is symbolic linked to elv.sh were the mode is set according to the script stem
* o : 09/10 : succeed to get ELV selected dynamic geometry to work with force triangulated solids using postcache SScene to SScene sub-selection
* o : 09/09 : preparations to allow force triangulated geometry to follow ELV postcache selection to allow dynamic geometry speed testing
* o : 09/05 : enable rendering to run in Release build to avoid test fails
* o : 09/04 : confirmed capture of WITH_HEISENBUG in CSG/csg_intersect_leaf.h with an acceptable fix not requiring any magic printf 
* o : 09/04 : heisenbug resisting arrest : within intersect_leaf an earlier magic printf also works
* o : 09/02 : dump debugging CSG issue with optix 7.5 and cuda 12.4 : adding lots of dump debug to csg_intersect headers scares away the bug, but switching off the debug and it comes back : bizarre
* o : 09/02 : CUDA level debug of the intersects shows no difference between 11.7 and 12.4 with OptiX 7.5 on TITAN_RTX and RTX_5000_Ada so resort to PIDX debug within OptiX raytrace


2024 Aug : shakedown integrated triangles, Ada test reveals CUDA 12.4 CSG boolean issue 
-----------------------------------------------------------------------------------------

* o : 08/30 : notes on CUDA 12.4 Ada CSG boolean issue
* o : 08/27 : shakedown forced triangulation, see notes/issues/flexible_forced_triangulation.rst
* o : 08/26 : 9cf40f6c0 - first untested nearly full cut at flexible forced triangulation across all levels : stree.h CSGFoundry and SBT


2024 Jul : updates for Geant4 11.2 
--------------------------------------

* http://localhost/env/presentation/opticks_20240702_kaiping_status_and_plan.html

  * n-ary CSG Compound "List-Nodes" => Much Smaller CSG trees
  * 3x3x3 grid of MultiUnion/list-node each with 7x7x7=343 Orb
  * FastenerAcrylicConstruction  

* o : 07/12 : change skin surface vector to map G4VERSION_NUMBER branch to 1122 



2024 Jun : add user control for triangulated geom : apply to guidetube torii
-------------------------------------------------------------------------------


* http://localhost/env/presentation/opticks_20240606_ihep_panel_30min.html
 
  * simulation unlimited by optical photons => greater understanding => more fruit


* o : 06/27 : initial try at compiling to optixir binary instead of ptx text
* o : 06/25 : add OpenGL snapshots using glReadPixels to SGLFW.h
* o : 06/24 : add sgeomtools.h based on G4GeomTools to provide torus bbox accounting for the phi range
* o : 06/23 : arrange for number key frame hopping to interpret SHIFT modifier as offset of 10, so can now have 20 bookmark frames
* o : 06/16 : handle listnode in stree::get_frame_remainder
* o : 06/16 : provide way to download render pixels and flip the vertical prior to annotation and saving, plus start on interactive GPU side flip with traceyflip, add MOI targeted frame hop from M key
* o : 06/11 : c4855b1e3 - start adding forced triangulation solids at stree::factorize stage 

2024 May : listnode translation from simple G4MultiUnion operational
------------------------------------------------------------------------

* o : 05/16 : anaviz for listnode converted from G4MultiUnion now working, performantly rendering solid of 7x7x7=343 constituent Orbs that would be impossible without listnode
* o : 05/15 : filling out G4MultiUnion sn.h CSGPrim/CSGNode listnode full conversion
* o : 05/15 : first impl of sn(listnode) conversion to CSGPrim/CSGNode mostly in CSGImport::importPrim
* o : 05/15 : get examples/UseOpticksCUDA/go.sh to work with CUDA 11.7
* o : 05/14 : review state of listnode impl, looks like primary lack is translation
* o : 05/14 : as different pkgs need to resolve GEOM in different ways put the resolution into the corresponding ctest runners rather than overloading GEOM.sh
* o : 05/09 : tri/ana flexible geometry now working to zeroth order
* o : 05/06 : start tri/ana generalization SBT setup using union of CustomPrim and Trimesh in the HitGroupData 


2024 Apr : triangle + analytic integration at OptiX and OpenGL levels
-----------------------------------------------------------------------

* implement interactive geometry navigation (header-only approach for fast cycle)
* http://localhost/env/presentation/opticks_20240418_ihep_epd_seminar_story_of_opticks.html


* o : 04/30 : 2adf7fc2c - rejig CSGOptix IAS creation to make it work WITH_SOPTIX_ACCEL, tri/ana integration reached to needing hitgroup branching 
* o : 04/29 : use the unified SOPTIX_BuildInput to handle Triangles and CPA within CSGOptiX/SBT.h for now hidden behind WITH_SOPTIX_ACCEL
* o : 04/29 : unify SOPTIX_BuildInput_IA.h SOPTIX_BuildInput_CPA.h SOPTIX_BuildInput_Mesh.h as specializations of SOPTIX_BuildInput.h for common handling
* o : 04/29 : remove context arg from SOPTIX_MeshGroup by deferring SOPTIX_Accel gas creation in order to conform closer to CSGOptiX/SBT API
* o : 04/28 : adopt more vertical API SOPTIX_MeshGroup::Create going from CPU side SMeshGroup to GPU side SOPTIX_MeshGroup in one step, to facilitate tri/ana integration
* o : 04/28 : move to SIMG.h STTF.h from former SIMG.hh STTF.hh removing ttf instance from SLOG.cc enabling header-only usage of SIMG.h STTF.h functionality
* o : 04/25 : 1e09ea291 - add external device pixel functionality to CSGOptiX.cc, enabling addition of CSGOptiXRenderInteractiveTest providing interactive CSGOptiX analytic rendering with WASDQE+mouse navigation
* o : 04/24 : move context handling to SOPTIX_Context.h freeing SOPTIX.h for top level coordination
* o : 04/24 : a81983c4a - tidy and review OpenGL SGLFW machinery
* o : 04/23 : review triangulated geometry machinery in syrap/SOPTIX.rst
* o : 04/18 : add notes/issues/G4CXTest_raindrop_shows_Geant4_Process_Reorder_doesnt_fix_velocity_after_reflection_in_Geant4_1120.rst
* o : 04/12 : add frame jumping by number key and camera type toggle
* o : 04/12 : add WASDQE 3D navigation with mouse arcball control of q_eyerot quaternion
* o : 04/11 : fix three issues : perspective raytrace/raster inconsistency, raytrace quaternion slewing due to not using proper inverse, raster mixup due to omitting GL_DEPTH_TEST setup
* o : 04/10 : use SBitSet.h for SGLM.h VIZMASK control used from sysrap/tests/SGLFW_SOPTIX_Scene_test.sh
* o : 04/04 : comparing UseGivenVelocity_KLUDGE in examples/Geant4/OpticalApp/OpticalAppTest.sh and g4cx/tests/G4CXTest_raindrop.sh
* o : 04/03 : add recording and point flags to examples/Geant4/OpticalApp using copy of minimal numpy writer np.h 
* o : 04/01 : notes from review of opticks presentations  (`git l 43e2cbbef^-1`)


::

    notes/objectives.rst
    notes/story_of_opticks_review.rst



2024 Mar : header-only SGLFW.h revival of triangulated OpenGL geometry viz, revive OptiX builtin triangle ray trace render
----------------------------------------------------------------------------------------------------------------------------

* implement OpenGL rasterized viz + OptiX triangle geometry 

* o : 03/28 : investigate velocity after reflection (TIR or otherwise), find that UseGivenVelocity keeps that working as well as refraction
* o : 03/27 : Merged PR from Yuxiang for enhancements to lookup based ART calc
* o : 03/26 : add InverseModelView IMV matrix calc in updateComposite and use that in updateEyeBasis to make the ray trace sensitive to the look rotation quaternion
* o : 03/26 : interactive CUDA-OpenGL interop optix ray trace of builtin triangle geometry working to some extent in sysrap/tests/SGLFW_SOPTIX_Scene_test.sh
* o : 03/26 : fixed bug in SOPTIX_Scene::init_IAS that was preventing IAS hits 
* o : 03/25 : complete triangulated header-only OptiX render sysrap/tests/SOPTIX_Scene_test.sh now runs, but getting all misses
* o : 03/25 : add minimal header only sppm.h for pre-OpenGL test of SOPTIX triangulated render
* o : 03/21 : encapsulate triangleArray buildInput and GAS creation into SOPTIX_Mesh.h 
* o : 03/20 : separate off SGLFW_Scene.h, start SCUDA_Mesh.h following pattern of SGLFW_Mesh.h, header only SCU.h  
* o : 03/19 : rename SGLFW_Render.h to SGLFW_Mesh.h following decoupling from SGLFW_Program.h
* o : 03/18 : fixed Linux instancing no-show, twas missing getAttribLocation from SGLFW_Program::enableVertexAttribArray_OfTransforms
* o : 03/17 : add SMesh::Concatenate for vtx/tri/nrm concat, used from SScene
* o : 03/14 : splitting SGLFW_Program.h and SGLFW_CUDA.h from SGLFW.h and better state control gets flipping between pipelines to work 
* o : 03/11 : fixed OpenGL viz bug, it was issue of vPos vNrm attribute enabling getting swapped, as they act on the active GL_ARRAY_BUFFER when invoked
* o : 03/10 : add lookRotation control using quaternion based SGLM_Arcball.h following Ken Shoemake original Arcball publication 
* o : 03/08 : expt with glm quaternions for Arcball impl
* o : 03/05 : SGLM.h tests review reveals some potential causes of mis-behaviour such as that seen with UseGeometryShader but needs rasterized comparison before can be actionable
* o : 03/04 : working out how to combine triangulated with analytic optix geometry, review some OpenGL GeometryShader rendering : want view/position navigation functionality that is reusable in all combinations : OpenGL/OptiX/interop
* o : 03/01 : add U4Mesh::MakeFold created meshes for all solids to stree created with U4Tree



2024 Feb : CPU memory leak find+fix, lookahead to triangle geometry with OptiX 7
--------------------------------------------------------------------------------------------

* http://localhost/env/presentation/opticks_20240227_zhejiang_seminar.html

  * Optical photons limit many simulations => lots of interest in Opticks 


* http://localhost/env/presentation/opticks_20240224_offline_software_review.html

  * Status of known issues : most leaks now fixed
  * [2] A:B Chi2 comparison of optical propagation history frequencies
  * [3] Pure Optical TorchGenstep 20 evt scan : 0.1M to 100M photons
  * Optimizing separate "Release" build in addition to "Debug" build
  * Absolute Comparison with ancient Opticks Measurements ?
  * Yuxiang Hu : Gamma Event at CD center : Comparison of JUNOSW with JUNOSW+Opticks


* o : 02/29 : update examples/UseOptiX7GeometryModular to work with OptiX 7.5 while thinking about reviving triangulated and interactive graphics
* o : 02/29 : switch to SEvt::getLocalHit from the old SEvt::getLocalHit_LEAKY impl
* o : 02/29 : note that intended new implementation of SEvt::getLocalHit avoids most of the off-by-one sensor identifier complications by using the CPU side double precision transforms that as never uploaded dont need to offset the identifier
* o : 02/19 : avoiding leaking transforms for every hit in sframe.h and CSGTarget.h reduces leak from 2300 kb to 800 kb in U4HitTest with 1750 hits, on Darwin
* o : 02/01 : add sysrap/tests/sleak.sh in the style of sreport.sh but simpler as just focussing on leak checking 


2024 Jan : GPU memory leak find+fix
-----------------------------------------------------


* o : 01/24 :  notes on fixing the 14kb/launch VRAM leak due to use of separate CUDA stream for each launch, plus change non-controllable per launch logging to be under SEvt__MINIMAL control
* o : 01/22 : implement running from a sequence of input gensteps such that cxs_min_igs.sh can redo the pure Opticks GPU optical propagation for gensteps persisted from a prior Geant4+Opticks eg okjob/jok-tds job
* o : 01/02 : add NPX.h ArrayFromDiscoMapUnordered handling of int,int unordered_map



2023 Dec : multi-event profiling and leak finding infrastructure, intro Release build, first JUNO+Opticks release, 3 test scripts 
------------------------------------------------------------------------------------------------------------------------------------

* http://localhost/env/presentation/opticks_20231219_using_junosw_plus_opticks_release.html

  * ~/o/cxs_min.sh  ## 2.2M hits from 10M photon TorchGenstep, 3.1 seconds 
  * First Pre-Release has lots of rough edges


* http://localhost/env/presentation/opticks_20231211_profile.html

  * Introduce Three Opticks test scripts 
  * Optimizing separate "Release" build in addition to "Debug" build
  * sreport.{sh,cc,py} : Opticks Event metadata reports and plots
  * Debug   : 0.341 seconds per million photons,    34s for 100M photons 
  * Release : 0.314 seconds per million photons,    31s for 100M photons  
  * ~2x ancient with old PMT model
  * Amdahls "Law" : Expected Speedup Limited by Serial Processing 
  * How much parellelized speedup actually useful to overall speedup?


* o : 12/29 : add mock lookup function to test the GPU texture 
* o : 12/17 : quantify leak GB/s with linefit, reduce smonitor logging
* o : 12/09 : examine preprocessor flattened CSGOptiX/CSGOptiX7.cu with preprocessor.sh to look for inadvertent use of printf and doubles as suggested by opticks-ptx for Release PTX showing a few of those left 
* o : 12/01 : add run level profile recording
* o : 12/01-07 : enhance multi event profiling and leak checking

2023 Nov
----------

* o : 11/25-30 : enhance profiling and reporting machinery eg NPFold::LoadNoData
* o : 11/06 : bump the CXX dialect to c++17 now that are shifting to OptiX 7.5 CUDA 11.7
* o : 11/03 : revisit tests : all sysrap passing
* o : 11/01 : improve chi2 interpretation reporting by QCF : confirm fix A/B g4cx/tests/G4CXTest_raindrop.sh difference by using Opticks defaults that correspond closer to Geant4 


2023 Oct
---------

* o : 10/31 : remove U4VolumeMaker::PVF as PMTFASTSIM now fully replaced by PMTSIM, improve error handling in U4VolumeMaker::PV, add a starter script g4cx/tests/G4CXTest_hello.sh for using G4CXTest with users gdml 
* o : 10/12 : revive examples/UseGeometryShader using more controlled sysrap/tests/sphoton_test.cc generation of the record.npy array


2023 Sept
-----------

* http://localhost/env/presentation/opticks_20230907_release.html

  * problem solids
  * 3inch fix
  * Geometry Translation now using minimal intermediate model
  * degenerate PMT apex virtual wrapper issue 


* o : 09/29 : document stamp_test.sh as a demo of local/remote workflow
* o : 09/20->25 : covid 
* o : 09/19 : add U4Mesh.h bringing over parts of the old X4Mesh.cc for polygonized viz of Geant4 solids with pyvista
* o : 09/04 : leaping to new workflow, removing old packages from om-subs--all and hiding use of old package headers behind WITH_GGEO in G4CXOpticks
* o : 09/04 : remove the CSG_stree_Convert approach as CSGImport now almost complete

2023 August
------------

* o : 08/27 : review CSGFoundry old/new diffs remaining, shows two left : boundary index and subNum on compound root nodes
* o : 08/27 : fix old workflow prim aabb bug, that incorrectly inflated some bbox due to inclusion of all zero bbox from operator nodes that was not skipped as an unset bbox : with similar new workflow fix the prim bbox are now matching between geometry workflows 
* o : 08/17 : sn.h nodes now precisely match snd.hh as demonstrated for full geometry with U4TreeCreateTest.sh 
* o : 08/15 : bring most snd.hh features over to the more flexible sn.
* o : 08/12 : CSGFoundry::CreateFromSim as no point operating from stree alone as SSim info required for operational geometry
* o : 08/08 : CSG/tests/csg_intersect_leaf_test.sh shows intersect_leaf normals onto sphere with transforms as used by PMT Ellipsoid are not normalized
* o : 08/04 : start g4cx/tests/G4CXAppTest.sh for standalone bi-simulation
* o : 08/01 : complete cleanup of G4CXOpticks::simulate moving event handling down to QSim and SEvt levels, add selective saving to SEvt::save



2023 July : mat+sur+bnd+optical translation into new workflow, special surface enum, qpmt.h special surface impl, MOCK_TEXTURE
----------------------------------------------------------------------------------------------------------------------------------

* http://localhost/env/presentation/opticks_20230726_kaiping_software_review.html

  * Opticks used to find JUNOSW bugs, many of them... 
  * using Opticks improves CPU simulation too !!


* o : 07/19 : expand mocking further, such that QSim_MockTest::propagate can now run qsim::propagate without CUDA 
* o : 07/18 : expand MOCK_TEXTURE/MOCK_CUDA coverage into QBnd QTex
* o : 07/17 : get CPU QProp_test.sh MOCK_CURAND version of the GPU QPropTest.sh to work 
* o : 07/15 : first cut at qsim::propagate_at_surface_CustomART by reusing qsim::propagate_at_boundary with override of TransCoeff using theTransmittance 
* o : 07/15 : disable isur from absorbers without RINDEX as pointless and confusing, for example isur on a Vacuum///Steel border
* o : 07/14 : confirmed fix, now that old and new workflows agree on sensors are getting expected CSGFoundry inst 4th column
* o : 07/13 : look into going direct to CSGFoundry from stree in CSG/tests/CSG_stree_Convert_test.sh
* o : 07/13 : found smoking gun in GSurfaceLib the change to PMT geometry means isSensor no longer giving true,  as the LPMT bnd surfaces now do not have EFFICIENCY prop
* o : 07/12 : investigate unexpected sensor_id in CF inst array, needed by QPMT for lpmtid 
* o : 07/11 : QSimTest shakedown, avoid SBnd bnd name assert 
* o : 07/10 : integrating SPMT/QPMT with QSim
* o : 07/10 : GGeo::convertSim update for SSim new plumbing
* o : 07/08 : avoid the OLD prefix by using separate NPFold called GGeo for the old workflow arrays, split off standard array names into smat.h
* o : 07/07 : encapsulate scint icdf prep into U4Scint.h used from U4Tree::initScint
* o : 07/06 : prep for changing optical array to contain smatsur.h enum 
* o : 07/06 : try changing X4/GGeo workflow to match skinsurfaces by LV pointer instead of the LV name to avoid notes/issues/old_workflow_finds_extra_lSteel_skin_surface.rst
* o : 07/06 : investigate bnd difference with unexpected Water/StrutAcrylicOpSurface/StrutAcrylicOpSurface/Steel
* o : 07/04 : review surface handling, attempt to recreate oldsur without GGeo/X4 in U4Surface::MakeStandardArray
* o : 07/03 : try PhysicsTable based approach to handling Geant4 Water RAYLEIGH from RINDEX special casing, used from U4Tree::initRayleigh
* o : 07/02 : remove use of Opticks bash/CMake machinery from examples/UseCustom4/go.sh as Custom4 is upstream from Opticks and should work off the CMAKE_PREFIX_PATH without any need for Opticks
* o : 07/02 : UseCustom4 version checking. Notes on c4 updating from v0.1.4 to v0.1.5

2023 June : CSGOptiX ELV render scanning, SPMT,QPMT  
--------------------------------------------------------------

* o : 06/30 : attempt direct from stree/NPFold sstandard::bnd sstandard::mat sstandard::sur creation without GGeo/X4
* o : 06/25 : integrating Custom4 Stack TMM calc with qudarap/QPMT.hh/qpmt.h using full PMT lpmtid info from SPMT.h 
* o : 06/24 : rationalize SPMT.h stack::calc avoiding 2nd stackNormal instance, accomodate new ART layout
* o : 06/19 : extending QPMT.hh QPMTTest.cc to use full SPMT.hh PMT info
* o : 06/17 : SPMT.h summarizing the PMTSimParamData NPFold into a few arrays for upload to GPU with QPMT.hh and use with qpmt.h
* o : 06/15 : bring multi token substitution to U::Resolve
* o : 06/14 : getting g4cx/tests/G4CXOpticks_setGeometry_Test.sh to convert GEOM FewPMT into CSGFoundry for use from the cxs_min.sh/cxt_min.sh/cxr_min.sh and subsequently for small geometry interation for Custom4 GPU equivalent development
* o : 06/12 : add TMAX flexibility and increase the default to avoid far cutoff in wide views

* 06/11 : Presentation, Qingdao, SDU, Workshop

  * http://localhost/env/presentation/opticks_20230611_qingdao_sdu_workshop.html
  * http://simoncblyth.bitbucket.io/env/presentation/opticks_20230611_qingdao_sdu_workshop.html

* o : 06/06 : reviving the ELV render results table
* o : 06/04 : add cudaSetDevice to Ctx::Ctx as envvar approach is somehow not working
* o : 06/02 : SCVD::ConfigureVisibleDevices needed for CVD envvar control of CUDA_VISIBLE_DEVICES 
* o : 06/02 : rejig SGLM.h to avoid kludge double update call and add ESCALE/escale glm::mat4 for matrix consistency
* o : 06/01 : setup standalone SGLM_set_frame_test.sh to duplicate the SGLM.h/sframe.h view mechanics done by CSGOptiX::RenderMain

2023 May
---------

* jo : 05/25 : Finally MR 180 is merged : PMT Geometry pivot 

  * https://simoncblyth.bitbucket.io/env/presentation/opticks_20230525_MR180_timestamp_analysis.html

* 05/25 : Presentation

  * Opticks + JUNO : MR180 Timestamp Analysis 
  * http://localhost/env/presentation/opticks_20230525_MR180_timestamp_analysis.html
  * http://simoncblyth.bitbucket.io/env/presentation/opticks_20230525_MR180_timestamp_analysis.html

* o : 05/19 : try using U4Touchable::ImmediateReplicaNumber from U4Recorder::UserSteppingAction_Optical for SD step points to set the sphoton iindex : possibly a fast way to get the PMT copyNo  without any ReplicaDepth searching
* o : 05/18 : SProfile.h simple timestamp collection and persisting struct, using to profile junoSD_PMT_v2::ProcessHits
* o : 05/15 : standalone N=0,1 timestamp analysis plots 
* o : 05/13 : record BeginOfEvent EndOfEvent time stamps into photon metadata, start analysis of all the time stamps 
* o : 05/12 : misc notes while reviewing for presentation, stamp.h for simple epoch stamping

* 05/08 : CHEP Presentation

  * Opticks : GPU Optical Photon Simulation via NVIDIA® OptiX™ 7, NVIDIA® CUDA™
  * http://localhost/env/presentation/opticks_20230508_chep.html
  * http://simoncblyth.bitbucket.io/env/presentation/opticks_20230508_chep.html


2023 April
-----------

* 04/28 : Presentation

  * Opticks + JUNO : More junoPMTOpticalModel Issues + Validation of Custom4 C4OpBoundaryProcess Fix
  * http://localhost/env/presentation/opticks_20230428_More_junoPMTOpticalModel_issues_and_Validation_of_CustomG4OpBoundaryProcess_fix.html
  * http://simoncblyth.bitbucket.io/env/presentation/opticks_20230428_More_junoPMTOpticalModel_issues_and_Validation_of_CustomG4OpBoundaryProcess_fix.html

* j : 04/19 : notes on getting the merge to work without conflicts using dry run test merges in temporary clone
* jo : 04/17 : MR 180 is requested : PMT Geometry pivot

  * https://code.ihep.ac.cn/JUNO/offline/junosw/-/merge_requests/180


* o : 04/15 : split SEvt::transformInputPhoton from SEvt::addFrameGenstep and do that from SEvt::setFrame as the transformed input photons are needed earlier than SEvt::BeginOfEvent
* j : 04/12 : notes on fixing the opticksMode:0 vs 2 difference : its was due to clearing of interaction lengths for alignment being left switched on 
* o : 04/03 : initial try at U4Navigator based simtrace working to some extent within simple standalone geom
* o : 04/02 : explore use of G4Navigator for a more general Geant4 only simtrace implementation

2023 March
-----------


* o : 03/30 : get uc4packed from the spho label into current_aux for each step point
* o : 03/27 : factor off history chi2 comparison into opticks.ana.qcf QCF QU
* o : 03/24 : switch from PMTSIM to CUSTOM4 for the custom boundary process C4OpBoundaryProcess and associated headers
* o : 03/22 : investigate how to handle custom boundary process deps from opticks and junosw : looks like will have to split off mini-package
* o : 03/21 : tidy U4SimulateTest U4Recorder in preparation for using the recorder from an AnaMgr within monolith 
* o : 03/13 : testing junoPMTOpticalModel::ModelTriggerSimple_ with dist1 > EPSILON 2e-4 with onepmt line test avoids double tigger issue, brings N=0/1  history chi2 into match
* o : 03/09 : support for ModelTrigger_Debug
* o : 03/03 : simple low dependency approach to A-B history comparison in u4/tests/U4SimulateTest_cf.py and add NNVT fake step point detection
* o : 03/02 : working out how to skip fakes with U4Recorder to allow A-B comparison between unnatural and natural PMT geometry

2023 Feb 
---------

* o : 02/27 : generalize U4VolumeMaker to allow testing with multiple PMT types from PMTSim
* o : 02/23 : rejig CustomART to facilitate switching between Traditional-Detection-at-photocathode-POM and MultiFilm-photons-in-PMT-POM 
* o : 02/20 : snd sndtree updates for sn.h, higher level s_pool::serialize s_pool::import
* o : 02/18 : pull s_pool.h out from sn.h to avoid duplication of serialize/import machinery
* o : 02/17 : using deepcopy succeeds to make sn.h pruning squeaky clean with absolutely zero node leaks
* o : 02/17 : try not leaking nodes in sn tree manipulations like pruning in order to maintain an active node map that can use to serialize
* o : 02/16 : comparing transforms reveals that they all match between A and B but 93:solidSJReceiverFastern and 99:uni1 which are balanced/unbalanced differ in the ordering of the transforms : somehow transforms get shuffled, is primitive order changed by the balancing
* o : 02/16 : CSGFoundryAB.sh down to 74/8179 discrepant tran/itra that are tangled with lack of tree balancing for lvid 93:solidSJReceiverFastern 99:uni1
* o : 02/14 : simplify snd/scsg reducing overlap between them and add inverse csg transform handling to stree,snd trying to duplicate itra for the CSGImport 
* o : 02/10 : start comparing CSGFoundry from CSGImport of stree and old way via GGeo, find lvid 93 99 are balanced in old but not new 
* o : 02/08 : maybe can use simple pointer based minimal binary tree node sn.h to do the dirty tree population and pruning prior to bringing into the persistable snd.hh


* 02/06 Presentation

  * Opticks + JUNO : PMT Geometry + Optical Model Progress 
  * http://simoncblyth.bitbucket.io/env/presentation/opticks_20230206_JUNO_PMT_Geometry_and_Optical_Model_Progress.html
  * http://localhost/env/presentation/opticks_20230206_JUNO_PMT_Geometry_and_Optical_Model_Progress.html

* jo : 02/03 
 
  * MR 126 is merged : https://code.ihep.ac.cn/JUNO/offline/junosw/-/merge_requests/126

* o : 02/01 : work out way of defining inorder traversal for n-ary tree, use that in snd::render_r writing to scanvas.h

2023 Jan : NPFold/NPX map serialize, low dep PMT data branch, U4Material/U4Surface/U4Tree/U4Solid/snode/snd
-----------------------------------------------------------------------------------------------------------------

Work split into two:

1. preparing low dependency PMT data access for use by CustomG4OpBoundaryProcess (prepping MR 126)

   * Opticks+JUNO blocked 01/17 (Tue ~2 weeks ago) awaiting merge request to be granted 
 
2. transition to Opticks direct geometry translation (massive code reduction is close)
   
* o : 01/28 : debug snd.hh scsg.hh failure to set parent, fixed by reserving the vectors in scsg::init
* o : 01/27 : U4Solid::init_Ellipsoid now U4TreeCreateTest.sh gets thru all JUNO solids, Polycone Ellipsoid need testsing and ZNudge
* o : 01/27 : U4Polycone.h requires snd.hh ZNudge mechanics, try using CSG_CONTIGUOUS snd::Compound for polycone instead of binary tree as X4Solid does
* o : 01/27 : adopt general n-ary tree handling used with snode.h for snd.hh too, switch to int ref returns for snd statics, add U4Solid::init_Sphere
* o : 01/26 : add Tubs and Cons, find complex snd::Boolean not following l == r+1, how to nc/fc handle that ? 
* o : 01/26 : add snd.h persisting with referenced pools, plus generalize to non-boolean tree using fc/nc first_child/num_child
* o : 01/25 : building out U4Solid.h
* o : 01/24 : collect skin and border surfaces together as needed for the boundary surface index approach 
* o : 01/23 : extend U4Surface used from U4Tree and C4 
* o : 01/21 : new C4 package (short for CSG_U4) for direct from Geant4 to CSG geometry conversion expts
* o : 01/20 : U4PMTAccessorTest.cc testing PMT accessor external to j/PMTFastSim
* j : 01/20 : comparing IPMTAccessor scans from PMTAccessor and JPMT show max 1e-15 deviations
* j : 01/20 : setup to compare PMTAccessor.h with JPMT.h profiting from PMTSimParamData persisting functionality

* o : 01/19 : start PMTAccessor.h destined for monolith residence, but developed outside for fast dev cycle

* o : 01/18 : start simplifying the standalone j/Layr/JPMT.h API used by u4/CustomART.h in order to converge the standalone and full APIs such that they can both be used with u4/CustomART.h
* j : 01/18 : making the standalone JPMT.h API closer to that needed for the full non-standalone API such that u4/CustomART.h can work with both of the APIis


* jo : 01/17 : (Tue before CNY) : branch blyth-66-low-dependency-PMT-data-access is ready for merge as it addresses the problem outlined in issue 66
* jo : 01/17 

  * Make MR 126 : https://code.ihep.ac.cn/JUNO/offline/junosw/-/merge_requests/126


* jo : 01/12 -> 01/17 : WIP PMTSimDataSvc branch 
* o : 01/13 : G4CXOpticks__SaveGeometry_DIR envvar control for G4CXOpticks::SaveGeometry as need to do the save later than setGeometry when have SSim additions
* o : 01/10 : enhancements to allow NPFold.h persisting of jo:Simulation/SimSvc/PMTSimParamSvc/src/PMTSimParamData.h
* o : 01/09 : make NP.hh decl and impl ordering consistent for ease of navigation, add NP::ArrayFromVec NP::ArrayFromMap
* j : 01/05 : brief look at reading root files without ROOT, conclude too much effort for the problem of PMT info from .root as can use a more cunning approach for that


2022 Dec : Simplify junoPMTOpticalModel (MultiLayer TMM) using standalone testing 
-----------------------------------------------------------------------------------

* 12/21 : NP::LoadCategoryArrayFromTxtFile NP::CategoryArrayFromString for enum arrays

* 12/20 : presentation

  * Opticks + JUNO : junoPMTOpticalModel FastSim issues and proposed fix using a CustomG4OpBoundaryProcess
  * http://localhost/env/presentation/opticks_20221220_junoPMTOpticalModel_FastSim_issues_and_CustomG4OpBoundaryProcess_fix.html
  * http://simoncblyth.bitbucket.io/env/presentation/opticks_20221220_junoPMTOpticalModel_FastSim_issues_and_CustomG4OpBoundaryProcess_fix.html


* 12/16 : pull CustomART.h CustomStatus.h out of CustomBoundary.h : rationalize theCustomStatus handling and presentation in preparation for switching from CustomBoundary.h to CustomART.h making more use of standard G4OpBoundaryProcess mom,pol changes
* 12/16 : sboundary_test_brewster.sh sboundary_test_critical.sh : plots comparing polarizations before and after TIR and Brewster angle ref
* 12/15 : try to do less in CustomART by reusing the mom/pol impl of G4OpBoundaryProcess::DielectricDielectric
* 12/15 : illustrating Brewsters angle polarization using sysrap/tests/sboundary_test.sh showing color wheel polarization directions before and after reflect or transmit
* 12/15 : make many G4CXOpticks methods private, to simplify usage : suggestions for Hans CaTS in notes/issues/Hans_QSim_segv_with_CaTS.rst
* 12/13 : bring over the new polarization from sboundary.h into sysrap/tests/stmm_vs_sboundary_test.cc
* 12/13 : drawing more parallels between stmm.h and sboundary.h calcs in order to correctly get reflect and transmit polarizations in stmm.h context
* 12/12 : comparing two layer stmm.h with sboundary.h based on qsim::propagate_to_boundary, matched TransCoeff
* 12/10 : thinking about how to bring CustomBoundary.h to GPU, start looking into mom and pol vectors after the TMM stack
* 12/08 : more vectorized (NumPy) way to get the seqhis histories 
* 12/08 : make U4Touchable::ReplicaNumber implementation comprehensible, collect G4 ReplicaNumber into sphoton.h iindex
* 12/07 : generalize Geant4 volume/solid intersect plotting to any level of transforms using U4Tree/stree in u4/tests/U4PMTFastSimGeomTest BECOMES U4SimtraceTest.cc
* 12/06 : avoid duplication and simplify by moving jps/N4Volume.hh jfs/P4Volume.hh down to common header-only sysrap/SVolume.h
* 12/05 : j/PMTFastSim/junoPMTOpticalModel_vs_CustomBoundaryART_propagation_time_discrepancy.rst
* 12/05 : pull CustomBoundary.h out of InstrumentedG4OpBoundaryProcess to make it more palatable 
* 12/04 : logging all consumption for big-bouncer with both N=0,1 geometries : DECIDE THAT ALIGNING DIFFERENT GEOM WHILE POSSIBLE PHOTON-BY-PHOTON IS KINDA POINTLESS 
* 12/04 : add envvar control of Absorption and Scattering in U4Physics, but cannot use for big-bouncer as the different consumption makes that no longer a big bouncer
* 12/03 : add SEvt::aux for collecting point-by-point debug info, currently SOpBoundaryProcess::get_U0 
* 12/02 : prep machinery to do step-by-step SEvt__UU_BURN to try to keep consumption aligned between the with-fakes and natural geometry 
* 12/02 : flipping the normal convention helps to give expected refraction, but now need to keep random consumption aligned between the old geometry with fake same material volumes and new simple geometry with no fakes 
* 12/01 : comparing junoPMTOpticalModel::Refract with InstrumentedG4OpBoundaryProcess::CustomART : initial shakedown bugs


2022 Nov : standalone fastsim checks, pivot to InstrumentedG4OpBoundaryProcess::CustomART$
---------------------------------------------------------------------------------------------

* over in j: developed Layr/Layr.h single header TMM 

* 11/30 : bringing over junoPMTOpticalModel into InstrumentedG4OpBoundaryProcess::CustomART
* 11/29 : start pivot to customizing u4/InstrumentedG4OpBoundaryProcess as seems FastSim cannot handle very simple geometry without fakes
* 11/29 : extend sseq.h to store NSEQ:2 64 bit elements, so sseq.h now handles maxbounce of NSEQ*16 = 32 without wraparound/overwriting issues
* 11/26 : change spho label gn field to uchar4, as need to pass FastSim ARTD flg via trackinfo for the U4Recorder to work with multiple PMTs
* 11/25 : increase the bounce limit, add extra record plotting to xxv.sh
* 11/25 : review U4Recorder+SEvt, add SEvt::resumePhoton in attempt to handle FastSim/SlowSim transition detected by fSuspend track status in U4Recorder
* 11/23 : save/restore when labelling in U4Recorder::PreUserTrackingAction_Optical succeeds to allow geant4 rerunning of single photons without precooked randoms by storing the g4state MixMaxRng into an NP array managed within SEvt
* 11/22 : U4Recorder::saveOrLoadStates attempting to save and restore g4 random states, for pure optical single selected photon rerun
* 11/22 : NP::MakeSelectCopy for array masking, eg to rerun a single generated photon
* 11/20 : PMTFastSim integration in GeoChain for translate.sh and extg4 for xxv.sh, expand storch.h adding T_LINE
* 11/18 : 3D plotting ModelTrigger yes/no positions : Getting familiar with FastSim in junoPMTOpticalModel
* 11/18 : 2nd implementation of Catmull Rom spline, factoring off weights reduces the interpolation to a single matrix multiply for each segment


* 11/17 : presentation 

  * Opticks+JUNO : PMT Mask Bugs, GPU Multilayer TMM 
  * http://simoncblyth.bitbucket.io/env/presentation/opticks_20221117_mask_debug_and_tmm.html
  * http://localhost/env/presentation/opticks_20221117_mask_debug_and_tmm.html

* 11/17 : try Catmull-Rom spline around a circle : in principal it looks like OptiX 7.1+ curve could handle guidetube 
* 11/15 : fast sim debug using U4PMTFastSimTest.cc 
* 11/15 : add access to volumes from PMTFastSim via U4VolumeMaker::PV, use from u4/tests/U4PMTFastSimTest.cc
* 11/12 : fix lots of CSG test fails, overall down to 25/507 fails
* 11/12 : start reviving opticks-t tests, remove opticks-t- check on relic installcache dir and OPTICKS_KEY envvar 
* 11/11 : add SEvt::numphoton_collected SEvt::GetNumPhotonCollected SEvt::getNumPhotonCollected to avoid looping over all gensteps to get the running total, after have confirmed equivalence
* 11/10 : QPMTTest on device interpolation now working, after arranging for the last column integer annotation to survive narrowing by doing the narrowing when 3D 
* 11/10 : arrange for NP::MakeNarrow to preserve last column integer annotation when a metadata switch is enabled
* 11/05 : move geometry persisting earlier in G4CXOpticks, add GEOM example_pet to bin/GEOM_.sh, notes on bordersurface issue 
* 11/03 : try to remove OPTICKS_KEY dependency
* 11/03 : add ana/tests/check.sh ana/test/check.py to demonstrate basic use of ana photon history debugging machinery
* 11/01 : QProp::Make3D allowing to scrunch up higher dimensions eg from JPMTProp into standard 3, use from qudarap/tests/QPMTPropTest.cc

2022 Oct : plog/SLOG better integration, debug scintillation Birks issue, CPU/GPU complex TMM
-------------------------------------------------------------------------------------------------

* 10/29 : NPFold::load with multiple rel
* 10/29 : improve access into tree of subfold of arrays using NPFold::find_array 
* 10/29 : parsing deranged property txt file format in NP::LoadFromString and using that in NPFold::load_dir to recursively load directories of property txt files into trees of NPFold
* 10/28 : NP::ArrayFromString generalize to handle real world property text files, NP::get_named_value accessor for single column key:value prop files
* 10/28 : SProp.h machinery for loading directories of property text files into NPFold
* 10/28 : split into NP::ArrayFromTxtFile NP::ArrayFromString as both useful 
* 10/27 : Add NP::ArrayFromTxt NP::ReadKV array and const property from txt accessors, skip the geocache_code_version_pass assert
* 10/27 : change PIP::CreateModule_debugLevel to a value that works with OptiX 7.5 as well as 7.0
* 10/25 : low level _sizeof methods needed by https://github.com/simoncblyth/j/blob/main/PMTFastSim/LayrTest.sh
* 10/24 : develop pattern for std::complex/thrust::complex arithmetic with common nvcc/gcc source via the C++ using declaration
* 10/20 : om special case directories names PMTFastSim informing the build that these have sources in j repo
* 10/20 : try surrounding all use of OpenSSL 3.0 deprecated MD5 API to quell -Wdeprecated-declarations compilation warnings reported by Hans 
* 10/19 : in sdigest.h try to suppress OpenSSH 3.0 deprecated MD5 API compilation warnings  
* 10/19 : attempting a header only OKConf.h so STTF.hh and SIMG.hh can work header only, but it runs into SLOG.hh complications 
* 10/18 : incorporate x4,u4 changes from Hans for Geant4 1100 
* 10/18 : notes for cuda::std::complex
* 10/18 : prep to look into FastSim details
* 10/14 : add more CEHIGH regions to illuminate PMT inner corners
* 10/14 : a few more OptiX 7.5 API changes
* 10/13 : try generalization against OptiX 700 -> 750 API change with BI::getBuildInputCPA 
* 10/12 : move some of QCurandState down into SCurandState, aiming to tie together SEvt maxima with the number of curandState loaded
* 10/12 : adjust cehigh extra resolution genstep grid to look for overlaps insitu at PMT bottom
* 10/11 : tidy and document the Simtrace 2D cross-section intersect and plotting machinery
* 10/07 : some more fields in u4/U4Scintillation_Debug.hh 
* 10/07 : add SOpticksResource::GDMLPathFromGEOM used from G4CXOpticks::setGeometry
* 10/06 : update opticks-prepare-installation to use qudarap-prepare-installation which is based on qudarap/QCurandState
* 10/06 : QCurandStateTest to replace the old cuRANDWrapperTest
* 10/06 : issue reported by Ami : notes/issues/opticks-prepare-installation-needs-updating-from-cudarap-to-QUDARap-binary.rst start developing new workflow curandState preparation to avoid need for mixed workflows
* 10/05 : investigate LS property warnings during translation, bending of property domain meaning prevents GPropertyMap table presentation and causes lots of domain warnings
* 10/05 : U4Debug class to simplify comparisons, realize that opticksMode 1 steps should not be matching opticksMode 0 as is found, it is modes 0 and 3 that should have exact step match : they do currently
* 10/03 : document update of my plog fork https://github.com/simoncblyth/plog to the upstream latest, mainly for the new PLOG_LOCAL functionality that makes it possible to use full Opticks style logging controls within integrated packages 
* 10/02 : LOG_IF update the rest of the active packages to work with latest plog without dangling else warnings
* 10/01 : rest of the active projects, rename PLOG to SLOG : needed for updating to newer PLOG external 

2022 Sep : thin CYLINDER and CONE fixes required for new JUNO geom 
---------------------------------------------------------------------

* 09/30 : expt with plog int template argument to try to use opticks PLOG in across shared libs of external projects that do not use -fvisibility=hidden
* 09/29 : start examples/UseFindOpticks to look into mis-behaviour of PLOG LEVEL logging from an external library
* 09/28 : context info for U4Scintillation_Debug, remove index prefix
* 09/21 : standardize the geomlist to make it easier to work with multiple geometries, useful for checking hama solids just like have done with nnvt
* 09/17 : adopt new CSG_CONE implementation avoiding apex glancer MISS and parallel ray quadratic precision loss issues
* 09/16 : CSGSimtraceSample.sh for running small simtrace arrays, eg obtained from python selections of spurious rays
* 09/16 : G4Polycone cylinder + cone union, sprinkle of spurious appear to all have rays that when extended would go close to the cone apex
* 09/15 : move prior cylinder imp to CSG_OLDCYLINDER and promote CSG_ALTCYLINDER to CSG_CYLINDER, the new simpler cylinder imp avoids spurious intersects observed with thin cylinders like nmskTailOuterITube 
* 09/13 : implement a simpler less flops CSG_ALTCYLINDER/intersect_leaf_altcylinder that perhaps improves numerical robustness and speed 
* 09/12 : investigating axial ray intersect precision loss with very thin cylinders, try simpler approach as should be more robust
* 09/12 : change X4Solid thin cylinder as disc criteria to 0.1 mm from 1 mm : so the PMT mask sub-mm lips are translated as cylinder, look into lip spill intersects from vertical/horizontal rays : probably v.thin cylinder axial special casing problem 
* 09/11 : found probable cause of mask thin lip spurious intersects : a mistranslation of the hz 0.3 and 0.65 thin tubs as disc when cylinder needed 
* 09/05 : CSG/ct.sh CSGSimtraceTest.cc for CPU running of CUDA csg intersect code with standard simtrace approach
* 09/01 : x4t.sh now working, presenting X4Simtrace G4VSolid intersects using same simtrace approach as gxt.sh

2022 Aug  : U4Tree/stree instancing, find balancing can cause missing, Simtrace check PMT and Mask
------------------------------------------------------------------------------------------------------

* 08/30 : add CEHIGH_0/1/2/3 for adding additional gensteps to provide high resolution regions of a simtrace 
* 08/30 : X4Solid::convertEllipsoid add safety to upper and lower placeholder cuts where there is no such cut intended, to address the rare zsphere apex bug
* 08/29 : investigate futher MISS-ing near apex intersects, probably from handling when no upper cap
* 08/29 :  notes on fixing a rare zsphere MISS for rays expected to intersect close to apex : notes/issues/unexpected_zsphere_miss_from_inside_for_rays_that_would_be_expected_to_intersect_close_to_apex.rst 
* 08/27 : CSGSimtraceRerun.cc as used by CSG/nmskSolidMask.sh for highly detailed CSGRecord look at CSG intersect algorithm
* 08/26 : GeoChain single solid translations of j/PMTSim solids : nmskSolidMaskVirtual, nmskSolidMask, nmskSolidMaskTail and added GEOM handling using CFBaseFromGEOM
* 08/26 : morton bracketing, try to configure gxt.sh to run from GeoChain translated single solid geometry nmskSolidMaskVirtual 
* 08/21 : shave 21s off CSGFoundry::upload by only finding unique gas, as the others not needed 
* 08/20 : as stree is now held by SSim relocate persisted folders accordingly
* 08/19 : add stree::labelFactorSubtrees so U4Tree::identifySensitiveInstances can traverse just the global remainder nodes to find them there as well as within the instanced factor subtrees
* 08/16 : passing sensor_identifier all the way up to U4hit level using sphit.h which is populated by SEvt::getLocalHit using info from sframe.h
* 08/15 : transform debug by populating an stree.h during GGeo creation in X4PhysicalVolume::convertStructure
* 08/14 : correct off-by-one inconsistency in sensor_index in CSG_GGeo_Convert::addInstances
* 08/14 : more sqat4.h identity updates, pass stree into CSG_GGeo_Convert::Translate done by G4CXOpticks::setGeometry
* 08/13 : prepare for getting sensor_id sensor_index into CSG_GGeo created CSGFoundry inst at the CSG_GGeo stage as a transitional sensor solution prior to U4Tree/stree becoming the mainline route to create CSGFoundry 
* 08/08 : add GGeo::save_to_dir implemented by changing idpath for usage from G4CXOpticks::saveGeometry
* 08/06 : add gdxml to the standard om-subs--all package list
* 08/04 : conclude that U4Tree.h/stree.h minimal approach to factorization is matching the GGeo/GInstancer factorization, with improved precision and better capability for retaining mapping across the factorization which is helped stree.h simplicity and persistability


2022 July : B focus, start full geom random alignment
-------------------------------------------------------

**B : Fully instrumented Geant4 reached plateau, ready for A-B iteration, random alignment + comparison**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* https://bitbucket.org/simoncblyth/opticks/commits/?page=7

* 07/29 : lightweight geometry translation expts, serializing n-ary tree and forming subtree digests

  * U4Tree.h U4TreeTest.cc

* 07/27 : rejig SEvt accessors, heavy:gather lightweight:get 
* 07/26 : local frame sphoton.h for new workflow hits orchestrated by SEvt.hh using SGeo.hh and sframe.h

  * notes/issues/joined_up_thinking_geometry_translation.rst 

* 07/25 : preparations for JUNO offline integration of new workflow, stran.h Tran::photon_transform

  * sphoton::transform and look at targetted transform collection in u4/U4Tree u4/U4Transform

* 07/24 : move SEvt inside G4CXOpticks to simplify integrated use of G4CXOpticks

* 07/24 moving away from OPTICKS_KEY

  * QCerenkov was assuming saved GGeo with IDPath defined and access to IDPath/GScintillatorLib/LS_ori/RINDEX.npy : try without
  * try simply skipping GGeo::save when no idpath set, eg when no OPTICKS_KEY

* 07/23 : CSG/tests/CSGFoundryAB.sh notes/issues/ellipsoid_transform_compare_two_geometries.rst  
* 07/23 : notes/issues/review_geometry_translation.rst 
* 07/22 : avoid overlapping, the instanced geometry is not showing the ellipsoid transform problem
* 07/22 : pull SPlaceCylinder.h SPlaceSphere.h SPlaceRing.h out of SPlace.h 
* 07/21 : SPlace.h AroundCylinder and AroundSphere now creating instance transforms for testing, works OK for placing and orienting arrows in pyvista plotting
* 07/20 : a2b vector rotation matrix machinery to prepare test instanced geometry to try to reproduce the missed transform issue

  * stran.h Tran::MakeRotateA2B creating a transform matrix that rotates from one vector to another 

* 07/19 : using GDMLSub to select single PV of the PMT hatbox from full GDML and wrap it fails to reproduce the ellipsoid transform issue
* 07/19 : machinery for running from full GDML, but selecting a PV by solid name and ordinal and wrapping it in small geometry test (U4Volume::FindPVSub)

  * notes/issues/full_geom_missing_ellipsoid_transform_again.rst 

* 07/17 : check hama_body_log, dont see the issue (of ellipsoid without its scale transform)
* 07/17 : widen the line of input photons for DownXZ1000 for better check of PMT intersects
* 07/15 : decouple variable definition from export for : OPTICKS_INPUT_PHOTON OPTICKS_INPUT_PHOTON_FRAME 
* 07/13 : colored package list RST tables/pages using bin/stats.sh : code stats
* 07/13 : A-B : investigate extra BT in B cf A : looks like U4Recorder needs microStep suppression together with random rewinds to stay aligned 

  * notes/issues/ab_full_geom_extra_BT.rst 
  * notes/issues/ab_full_geom.rst 

* 07/12 : B : export A_FOLD in u4s.sh to allow U4RecorderTest.cc to load the sframe.npy from the A side so input photons can be transformed the same in A and B
* 07/11 : A+B : SEvt machinery for transforming input photons into instance frame using OPTICKS_INPUT_PHOTON_FRAME 
* 07/11 : B : use BoxOfScintillator for fast turnaround tagging of reemission random consumption
* 07/11 : B : tag consumption from DielectricMetal ChooseReflection DoReflection LambertianRand
* 07/11 : B : use SRandom.h protocol base to U4Random to allow SEvt::addTag to check cursor vs slots : enabling untagged consumption to be detected at the next SEvt::AddTag 
* 07/10 : B : switch to manual random consumption tagging as Linux SBacktrace::Summary misses crucial frames making auto tagging problematic
* 07/10 : B : reorganize the process Shims to not use inline imps to see if it changes the Linux SBacktrace::Summary for U4Stack::Classify
* 07/09 : B :  U4VolumeMaker::PVG for GDML reading using SOpticksResource::GDMLPath resolution using _GDMLPath envvar trick 
* 07/08 : A : comparing precision of intersect z position for TO BT SD and TO BR SA shows nothing special about BR, just small deviations get amplified by reflection
* 07/08 : A : more modularization of simtrace plotting to avoid too much duplication for new g4cx/tests/G4CXSimtraceTest.py gx level simtrace plotting with gxt.sh
* 07/07 : B : pull U4Step::MockOpticksBoundaryIdentity and U4CF out of U4Recorder
* 07/04 : A+B : remove 12 old pkgs from standard build list om-subs--all 
* 07/03 : A : working through initial issues with gx/tests/G4CXSimulateTest.cc
* 07/01 : B : start on generalizing U4VolumeMaker to work with jps:PMTSim provided volumes
* 07/01 : A : confirmed fix for AB and SC (absorb and scatter) deviations by using KLUDGE_FASTMATH_LOGF in qsim.h
* 07/01 : A : confirmed suspicion that bulk of AB/SC position aligned deviation is not from float/double but rather from -use_fast_math logf which is __logf vs full float or double logf : try KLUDGE_FASTMATH_LOGF to reduce the deviation in the u > 0.998f region 


2022 June : B + A-B focus : new workflow validation : U4Recorder, U4Random, more A+B sharing, enum based systematic random alignment in new workflow
------------------------------------------------------------------------------------------------------------------------------------------------------

**A-B : new systematic enum approach to random aligned simulation**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A+B : maximise shared code between A and B**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**B : Initiate U4Recorder : Geant4 simulation with full Opticks SEvt instrumentation**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 06/30 : A : finding the randoms leading to SC or AB shows they are all very close to 1. which leads to float/double difference as -log(u) is small, close to 1-u but in air for example the absorbtion length is large eg 1e7
* 06/30 : B : investigate effect of float log rather than double log in ShimG4OpAbsoption and ShimG4OpRayleigh
* 06/29 : A-B : upped sample to 1M : only 1/1M is not history aligned : no surprises with the 500/1M with > 0.1 deviations, start prep for PMTSim test geometry from U4VolumeMaker 
* 06/29 : A-B : use ana/p.py:cuss CountUniqueSubSelection for more systematic look at the 17/10k > 0.1 deviants : 11 of which are tangent skimmers
* 06/28 : A-B : 10k cxs_rainbow turn out to be history aligned including scattering and absorption immediately, found explanations for all deviants investigated so far
* 06/27 : A-B : start rayleigh scatter random align by increasing stats and geometry extents
* 06/23 : B : U4Process::ClearNumberOfInteractionLengthLeft at tail of U4Recorder::UserSteppingAction_Optical makes Geant4 consumption regular, so most of the history should be possible to align to, but these is difference in the tail due to NoRINDEX hack termination
* 06/20 : A+B : stag.h machinery for tagging random consumption, aiming to be usable from both simulations 
* 06/20 : A-B : devise a systematic simstream callsite approach to doing the alignment starting with an enumeration of curand_uniform consumption callsites
* 06/18 : B : arrange non-tmp directory for precooked randoms and use by default with U4Random, mechanics seem working : but so far not aligning history
* 06/17 : B : start using the NP::slice to create U4 material props from the bnd array 
* 06/16 : A+B : pull SBnd.h out of QBnd.hh to facilitate usage from U4, SBnd::getPropertyGroup to pull the standard bnd sets of 8 properties out of the bnd array 
* 06/16 : A-B : try using trans_length cutoff for judging normal incidence in attempt to get double and float calcs to special case more consistently 
* 06/15 : A : start adding prd to sevent.h to allow full quad2 isect collection for debugging the normal incidence decision issue, and other isect related issues
* 06/15 : A-B :  confirmed the cause of polarization difference is from the Geant4 double precision normal incidence judgement only matching the Opticks float judgement something like half the time
* 06/14 : A : reinstate DEBUG_PIDX to investigate polz nan 
* 06/14 : A+B : SEventConfig::SetStandardFullDebug to make it easier for separate executables to use same config, using this for notes/issues/U4RecorderTest_cf_CXRaindropTest.rst 
* 06/11 : B : use rainbow geometry with U4VolumeMaker::RaindropRockAirWater to expand checking with U4RecorderTest
* 06/11 : B : simplify U4 by moving material related methods to U4Material
* 06/09 : A+B : reemission bookkeeping : scrubbing BULK_ABSORB and setting BULK_REEMIT in SEvt::rjoinPhoton
* 06/07 : A+B : move hostside sevent.h instance primary residence from QEvent down to SEvt for common access from Geant4 and Opticks branches
* 06/07 : A+B : move aside the old sevent.h to sxyz.h in prep for the qudarap/qevent.h migration down to sysrap/sevent.h
* 06/06 : B : basic structure of U4Recorder looking much simpler than old way, hope the full implementation with flags etc.. can stay as simple 
* 06/03 : B : U4RecorderTest dumping explorations whilst reviewing the old CFG4 way and looking for simplifications 
* 06/02 : B : start setup of U4RecorderTest for notes/issues/reimplement-G4OpticksRecorder-CManager-for-new-workflow.rst
* 06/02 : A+B : prep to move event array persisting from QEvent down to SEvt using SCompProvider protocol so SEvt can invoke QEvent getComponent to do the downloads 
* 06/01 : A : remove the nasty mixed CFBase kludge now that have moved the virtual Water///Water skips to translation time instead of load time


2022 May : A focus : CUDA qsim.h continued, shallow stack, more A+B sharing, g4cx started, cherry picking as remove packages  
-------------------------------------------------------------------------------------------------------------------------------

**A : New worflow pure CUDA simulation, reached plateau**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* 05/31 : A : ELVSelection succeeds to skip the virtual jackets as visible in simtrace plotting but observe prim mis-naming at python level, presumably because the python naming is based on the copied geocache which is unchanged by dynamic prim selection
* 05/31 : A : implement notes/issues/namelist-based-elv-skip-string.rst for skipping virtual Water///Water PMT wrapper surfaces
* 05/30 : A : set default PropagateEpsilon to 0.05
* 05/28 : A+B : pyvista screenshot is proving finicky, yielding blank renders, so avoid issue using macos level mpcap.sh pvcap.sh sfcap.sh from env/bin
* 05/27 : A : simtrace geometry and cxsim hits together starting to work, needs perpendicular simtrace shift to hit control plus view is finnicky
* 05/26 : A : get the simtrace python analysis machinery to run with sframe instead of the old gridspec and gsmeta, simtrace shakedown of new frame genstep 
* 05/25 : A : rearrange CSGOptiXSimtraceTest genstep creation and metadata persisting using sframe as the central element
* 05/25 : A : incorporate Yuxiangs multifilm developments 
* 05/23 : A : G4CX logging hookup, debug CSGFoundry::Load
* 05/22 : A+B : move genstep collection entirely down to SEvt so workflow for all types of genstep can be the same
* 05/21 : B : try genstep collection approach in U4 that avoids baring soul using translation unit local static functions and variables to keep Opticks types out of the U4 header
* 05/20 : A : use SSim centralized input array management to simplify QSim::UploadComponents and make it more extensible plus eliminate duplicated setup code
* 05/18 : A : G4CXOpticks::setGeometry methods to start from every level of geometry, check snap in SRG_RENDER mode 
* 05/17 : A+B : migrate xercesc dependent cfg4/CGDMLKludge into new gdxml package, as JUNO GDML still needs kludging
* 05/17 : A : start bringing together top level package and interface class for new workflow g4cx/G4CXOpticks prior to more direct CSGFoundry from G4 translation 
* 05/16 : A : try simpler genstep collection in sysrap/SEvt.h 
* 05/12 : A+B : SDir.h for directory listing without using boost::fs  (brap/boost avoidance)
* 05/11 : A : get hemisphere_s/p/x_polarized and propagate_at_boundary_s/p/x_polarized to work again 
* 05/10 : A : modularize qsim further into qbnd qbase to avoid chicken-egg dependency issue in setup of qscint and qcerenkov
* 05/09 : A : pull qscint.h out of qsim.h for better encapsulation and clarity 
* 05/06 : A : remove unused and hence confusing template params, pull qcerenkov out of qsim
* 05/05 : A+B : layout for scerenkov.h following pattern of storch.h for filling out from ocu/cerenkovstep.h
* 05/02 : A+B : prepare SGLM for comparison with Composition to see if SGLM can take over from Composition within CSGOptiX (optickscore avoidance)


2022 April : A focus : qudarap/qsim.h QSimTest :  bringing simulation to plain CUDA (no OptiX)
--------------------------------------------------------------------------------------------------

**A : Plain CUDA simulation, OptiX use segregated**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 04/30 : A : QEvent::save and QEvent standardization, with sseq.h for encapsulated seqhis recording 
* 04/29 : A : merged in new PMT optical model
* 04/29 : A+B : standardize wavelength domain compression to use center-extent form for consistency with other domains and common handling, fix uchar4/char4 bug
* 04/26 : A+B : examples/UseGeometryShader : standalone flying point viz working, start flexible centralization of OpenGL/GLFW mechanincs into header-only-imp SGLFW.hh
* 04/25 : A : CXRaindopTest now providing photon histories, need compressed recording to push to higher stats for the history table
* 04/25 : A : integrating torch into QSim for on device GENTORCH QSimTest, relocate basis types storch.h scurand.d down from QUDARap to SysRap for SEvent genstep creation
* 04/22 : A : mocking curand_uniform with s_mock_curand.h enables CPU testing of some qsim.h methods in qsim_test.cc, textures not so easy to mock
* 04/21 : A : CSGMaker::makeBoxedSphere for raindrop geometry
* 04/20 : A : QBnd::Add using NP::itembytes to extract surfaces and materials from the boundary array plus QBnd::GetPerfectValues for things like perfectAbsorbSurface
* 04/12 : A : start integrating QEvent/qevent with QSim/qsim
* 04/12 : A : move QSeed functionality into QEvent and SEvent for clarity of control
* 04/08 : A : hit handling, encapsulating stream compation into SU.hh SU.cu tests/SUTest.cc 
* 04/05 : A : mock_propagate step-by-step photon recording debug 
* 04/05 : A : prepare for qsim::mock_propagate testing, switch from qprd to quad2 for easy loading of mock_prd 
* 04/04 : A : qudarap reflect_diffuse reflect_specular 


2022 March : A geometry focus :  AltXJfixtureConstruction, Dynamic Prim selection, QSimTest
-----------------------------------------------------------------------------------------------

* https://simoncblyth.bitbucket.io/env/presentation/opticks_20220329_progress_towards_production.html
* http://localhost/env/presentation/opticks_20220329_progress_towards_production.html

**Physics : QSimTest, BoundaryStandalone, RayleighStandalone**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 03/17 : switch gears from geometry issues back to non-geometry simulation bringing into QUDARap

  * fill_state, rayleigh_scatter, propagate_to_boundary, propagate_at_boundary : brought over to new qsim.h workflow 

    * maybe simpler to not sign the boundary, instead determining that at raygen level after trace
    * G4OpBoundaryProcess_MOCK to enable standalone boundary process testing where the surface normal is set externally

  * QSimTest : fine grained testing

    * random aligned comparison qsim-vs-bst : 1-in-a-million level match for S/P/X-polarization/normal-incidence/TIR
    * P-polarized random aligned comparison of propagate_at_boundary shows 1 in a million TransCoeff cut edger just like S-polarized 
    * persist the qstate with QState, swap the refractive indices to check TotalInternalReflection

  * Geant4 MOCK environment setup tests:

    * bst opticks/examples/Geant4/BoundaryStandalone
    * opticks/examples/Geant4/RayleighStandalone 

**Geometry :  AltXJfixtureConstruction, Prim speed scan using Dynamic Prim Selection**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* preparation for deployment of AltXJfixtureConstruction using CSG_CONTIGUOUS to replace slow and spurious isect afflicted solid
* implemented dynamic CSG prim selection in CSGCopy : needed for investigating geometry slowdown 

  * *working on this due to x3 slowdown in JUNO geometry observed compared with Dec 2021*  
  * equivalent to being able to dynamically control which G4VSolid instances are present in the geometry
  * have long been able to dynamically control the higher level compound solids, but 
    now that bottlenecks are in the global remainder compound solid zero needed finer level control
    in order to see what is causing the slowdowns in global geometry.  
  * now can do that at the lower level of Prim 
  * ELV enabled-logical-volume index selection by CSGCopy applied to the loaded CSGFoundry 

* transform bug motived developing additional volume level testing machinery and simplified geometry switching 

  * X4VolumeMaker for creation for test PV LV volumes for debugging and switch to using that with g4ok/G4OKVolumeTest.sh 

* Act on feedback from NVIDIA engineer at UK GPU Hackathon, with no performance difference (probably geom is otherwise bottlenecked) 

  * WITH_PRD pointer packing : nice for code organization
  * try CSGOptiX::initStack PIP::configureStack following optixPathTracer example 

* Created repo to share CSGFoundry geometry via tarballs https://github.com/simoncblyth/cfbase

  *  geocache-create-cfbase-tarball

* get subNum subOffset CSG generalization thru geometry translation

* XJfixtureConstruction, balanced tree incompatibility issue, CSG list nodes: CSG_CONTIGUOUS, CSG_DISCONTIGUOUS, CSG_OVERLAP

  * https://simoncblyth.bitbucket.io/env/presentation/opticks_20220307_fixed_global_leaf_placement_issue.html
  * http://localhost/env/presentation/opticks_20220307_fixed_global_leaf_placement_issue.html 


2022 February  : CSG list-node generalization to avoid large trees due to discovery of balanced tree incompatibility 
----------------------------------------------------------------------------------------------------------------------

* LHCb RICH geometry into CSG model for UK GPU Hackathon

  * https://simoncblyth.bitbucket.io/env/presentation/opticks_20220227_LHCbRich_UK_GPU_HACKATHON.html
  * http://localhost/env/presentation/opticks_20220227_LHCbRich_UK_GPU_HACKATHON.html
      
* CSG_DISCONTIGUOUS : leaf list with simple nearest ENTER/EXIT imp

  * added new compound node implemented in CSG/csg_intersect_node.h:intersect_node_discontiguous 
  * TODO: test use within CSG trees

* CSG_OVERLAP : a multi-INTERSECTION equivalent of the CSG_CONTIGUOUS multi-UNION
   
  * added new compound node implemented in CSG/csg_intersect_node.h:intersect_node_overlap
    based on farthest_enter and nearest_exit 
  * list based : so it can mop up intersection nodes into a compound node 
  * https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/OverlapBoxSphere.rst
  * :doc:`/notes/issues/OverlapBoxSphere`
  * TODO: test the compound prim can work in CSG tree 
  * TODO: think about intersecting with complemented (and unbounded phicut/thetacut/plane nodes) : 
    can CSG_OVERLAP be made to work with such leaves ?
  * potentially be used for general sphere combining intersects  

* thoughts on UK GPU hackathon

  * :doc:`/docs/geometry_testing`
  * https://bitbucket.org/simoncblyth/opticks/src/master/docs/geometry_testing.rst 

* CSG_CONTIGUOUS multiunion : trying to replace large trees with instead small trees with some large compound nodes

  * TODO: try to apply to XJFixtureConstruction : gather suitable union leaves to mop up into CSG_CONTIGUOUS   
  * TODO: detect suitable raw(unbalanced) G4BooleanSolid trees suitable for use of CSG_CONTIGUOUS 
    
    * see X4SolidMaker::AltXJfixtureConstruction the last G4UnionSolid in a sequence of them which 
      fulfill the topological requirements should have the "CSG_CONTIGUOUS" marker within its name

      * need to inhibit balancing for such trees
      * could explictly use G4MultiUnion in source geometry, see X4Solid::convertMultiUnion
      * can branch based on the solid name marker within X4Solid::convertUnionSolid


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

      * perhaps implementing CSG_OVERLAP that does for intersections what CSG_CONTIGUOUS  
        does for unions would allow implementing the general sphere directly with planes and cones 
        rather than with pairs-of-planes and pairs-of-cones 



* 02/01 : look into primitive ordering of balanced trees, simple cases do not change primitive order
* 02/01 : find with BoxFourBoxUnion the issues is not due to a change in primitive traversal order with the balanced tree, so it must be from the changed CSG structure


2022 January : JUNO trunk geom overlap checks, XJfixtureConstruction : atypical spurious, tree balancing implicated     
----------------------------------------------------------------------------------------------------------------------

* 01/31 : confirmed that switching off tree balancing avoids interior constituent boundary spurious intersects as that guarantees no disjoint-union-ness as the postorder tree grows


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



**2021 : Very Short Summary JUNO Opticks Progress** 

From scratch development of a shared GPU+CPU geometry model enabling 
state-of-the-art NVIDIA OptiX 7 ray tracing of CSG based detector geometries, 
flattened into a two-level structure for optimal performance harnessing ray trace 
dedicated NVIDIA GPU hardware. Development was guided by frequent consultation with NVIDIA engineers. 

JUNO Opticks development, validation and performance testing revealed issues with PMT 
and Fastener geometry, Cerenkov photon generation and PMT parameter services.
This has led to improved geometry modelling, Cerenkov numerical integration
and sampling and PMT services resulting in substantial improvements to the correctness
and performance of the JUNO Geant4 and Opticks based simulations.

**2021 : Medium Length (600 word) Summary : Broad headings progress**

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
    

2021 Dec : Apply ZSolid to cutting PMTs,  Geo Speed Check, work with LHCb RICH people on phicut/thetacut primitive
---------------------------------------------------------------------------------------------------------------------

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


2021 Nov : Develop Z-cutting G4VSolid that actually cuts the CSG tree, Geant4 2D cross-sections with (Q->X4)IntersectSolidTest, (Q->X4)IntersectVolumeTest 
------------------------------------------------------------------------------------------------------------------------------------------------------------

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



2021 Oct : QUDARap : QTex, QCerenkov : new workflow simulation atoms, JUNO Fastenener void subtraction reveals CSG limitation, Geant4 1100 property debug
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


2021 Aug : Cerenkov S2, QRng, QBuf, integrating the new workflow packages
----------------------------------------------------------------------------

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


2021 March : NVIDIA OptiX 7 expts in OptiXTest, curand skipahead, Start CPU/GPU CSG Model Development
-------------------------------------------------------------------------------------------------------

* http://simoncblyth.bitbucket.io/env/presentation/opticks_detector_geometry_caf_mar2021.html
* http://localhost/env/presentation/opticks_detector_geometry_caf_mar2021.html

  * detailed look at Opticks geometry approach (prior to OptiX7 CSG developments, but IAS/GAS mentioned) 


* http://simoncblyth.bitbucket.io/env/presentation/lz_opticks_optix7_20210315.html
* http://localhost/env/presentation/lz_opticks_optix7_20210315.html
 
  * resolve the compound GAS issue, by switching to using singe BI containing all AABB
  * intersect_node.h allowing CPU testing  
  * run into identity limitations


**OptiXTest : 2021/03/11 -> 2021/05/07**

* https://github.com/simoncblyth/OptiXTest/commits/main
* Geo, Grid, IAS, GAS, Shape, Foundry, Ctx, BI, PIP, PrimSpec

**Opticks repo**

* curand skipahead
* check for CUDA capable GPU before opticks-full-prepare 
* always save origin.gdml into geocache to try to avoid fails of tests that need GDML when running from geocache created live
* standalone-ish L4CerenkovTest exercising the branches of L4Cerenkov::GetAverageNumberOfPhotons and plotting NumPhotons vs BetaInverse with branches distinguished


2021 Feb : First development using new NVIDIA OptiX 7 API
----------------------------------------------------------

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



2021 Jan : Geant4 1070 Followup several bugs reported,  Learning OptiX 7 API
---------------------------------------------------------------------------------

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




