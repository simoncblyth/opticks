releases-and-versioning
===========================

Opticks repositories on Bitbucket and Github
-----------------------------------------------

Day-to-day Opticks development uses the bitbucket repository

* https://bitbucket.org/simoncblyth/opticks/commits/

Less frequenty Opticks is pushed to the github repository.

* https://github.com/simoncblyth/opticks



How to download a snapshot .tar.gz or .zip
---------------------------------------------

Git "snapshot" tags are occasionally made and pushed to
both bitbucket and github.

Snapshot .tar.gz/.zip for each tag can be downloaded from GitHub,
see the below page to find the URLs.

* https://github.com/simoncblyth/opticks/tags

For example download the v0.2.2 zip with::

    curl -L -O https://github.com/simoncblyth/opticks/archive/refs/tags/v0.2.2.zip

Note that the tarball does not include git information, it simply provides
a snapshot of the state of the repository at a particular commit that has been
marked by having an associated tag.


How to checkout snapshot tag into a new branch
------------------------------------------------

An alternative way to use a tag starting from a clone clone

::

    git fetch origin
    git fetch --all --tags               # fetch from upstream
    git tag                              # list the tags
    git checkout tags/v0.2.2 -b v022     # create branch for a tag and checkout into it
    git branch                           # list branches
    git checkout master                  # return to master with the latest




git clone https://github.com/simoncblyth/opticks.git
cd opticks
git checkout v0.2.2


Release Notes
----------------

* start from "git lg -n20" and summarize useful commit messages worthy of mention



v0.5.6 2025/10/29 : fix PMT inconsistency bug, file writing changes for production invokation, planning libcurl client 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* add LOGOVER control to cxs_min.sh to test invokation from non-standard LOGDIR, change default directory for run.npy run_meta.txt saving to invoking directory beside log and SProf.txt
* avoid issue of inconsistent MPMT 0-or-600 using SPMT_Num/pmtNum in SPMT.h from PMTParamData
* rc check must be immediately after the executable, handle GEOM_CFBaseFromGEOM in GEOM bash function
* change SProf control envvar to SProf__WRITE_INDEX which when 0/1/2/3/... writes to eg SProf_00000.txt , default of -1 inhibits writing
* add debug for copyno_pmtid_consistent assert in SPMT::init_lcqs
* remove direct QSim usage within G4CXOpticks, instead use QSim only via CSGOptiX from high level G4CXOpticks, motive is to simplify swap out of CSGOptiX with non-GPU client
* add T for tera/trillion to integer spec, change all narrow int spec methods in favor of int64_t ones
* idea how to proceed with libcurl based client, optionally replace CSGOptiX within G4CXOpticks with something like s_CURL_CSGOptiX.h fulfilling s_CSGOptiX.h protocol that CSGOptiX also fulfils
* apply cuda13gcc15_2.patch from Nicola fixing ULONGLONG4 LONGLONG4 defines needed for CUDA 13



v0.5.5 2025/10/23 : removed 32 bit limits on photons per event, improved profiling for production, improved geometry digest, python pointcloud overlap 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* start CUDA 13 following suggestions by new user Nicola
* removed max photons per event limitation of 4.29 billion, tested with 8.25 billion yielding 1.5 billion hits

  * change pidx to unsigned long long and increase PIDX sentinel from unsigned-1 4.29 billion to X40 (0x1<<40)-1 which is more than a trillion
  * add sphoton::set_index sphoton::get_index increasing bits from 32 to 32+8=40 for the index, extending maximum from 4.29 billion to more than a trillion
  * add lower level sdigest::BufRaw_ for faster digest based dupe checking
  * additions for NP for ~/j/hit_digest/hit_digest.sh checking billion hit array for dupes using CUDA to speed up comparisons
  * dont find duplicated hits, taking 22 min for a billion hits, 2 min for digests, 20 min for merging
  * change photon_idx to unsigned long long from unsigned to avoid duplicated photons in simulations exceeding 4.29 billion photons
  * review CUDA hit handling, while consider how to do hit binning using PMT local theta-phi made more relevant now for checking of event with 800M hit from X32 4.29 billion photon event
  * move orient bit in sphoton.h from orient_idx to orient_iindex to dedicate full 32 bits to the index
  * add raw digest API to sdigest.h and check if thats more efficient for dupe finding
  * basic hit checking able to handle 600M hits in 10mins
  * avoidance of 32bit clocking as push to simulation beyond 2.14 billion, testing in cxs_min.sh with G3
  * fix unsigned clocking of offset_bytes in NP::Concatenate causing tail zeros in very large hit arrays of 600M entries
  * improve VRAM OOM logging, with warning about inappropriate OPTICKS_EVENT_MODE

* profiling in production improvements

  * improve sreport code annotations
  * rejig the profile timestamp collection to use separate SProf.txt output instead of the kludge of appending to the run_meta.txt, add optional meta field to SProf

* improved geometry digest in preparation for server/client geometry consistency check

  * add SSim::AnnotateFrame which populates two new metadata fields in sframe.h : tree and dynamic, invoke the SSim::AnnotateFrame from CSGFoundry::getFrameE
  * add stree::get_tree_digest and stree::make_tree_digest which return the standard and dynamic geometry digests of the root node subtree, ie all nodes in the tree
  * add SBitSet::serialize to allow ELV bitset info to be included with sdigest.h digests

* python overlap checking using raytraced pointclouds with normals and KDTree to speedup closest pairing between clouds

  * configure logging with timestamps to measure the point cloud overlap time in cxt_min.py
  * improve CFBASE_ALT handling from C++ and python to allow transient usage of a dynamic ELV geometry consisently from C++ creator executable and python analysis of overlaps
  * cxt_min.sh simtrace plotting : add BOXSEL to restrict extent of point clouds to allow faster proximity/overlap checking
  * use scipy.spatial.KDTree to find nearest neighbors between point cloud intersects from two solids, then use intersect normals to give signed distance to find points that penetrate into the other cloud of points
  * add SFrameGenstep::Make_CEGS_NPY_Genstep where NPY array of (n,3) points in CEGS local frame are used to form n-gensteps, useful for investigating suspected overlap geometry positions


v0.5.4 2025/09/29 : some triangulation automation, handling for unsupported cut tubs, U4Polycone some generalization for new JUNO geometry, start impl of service-client with nanobind
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* add stree::desc_triangulate reporting which solids are auto and force triangulated
* initial impl of stree::is_auto_triangulate aiming to avoid the need to manually maintain a list of solids that must be triangulated, plus review how difficult instanced-triangulated impl will be (28 hours ago) <Simon C Blyth>
* U4Polycone.h handle polycone with non-zero rInner for some of the sections only, eg Mug shape without handle
* add u4/tests/StrutMakerTest.sh for testing some new geometry
* get magic callbacks to work with numpy header travelling in the HTTP POST body, see ~/np/tests/np_curl_test/np_curl_test.sh
* add eventID index to the high level API, the FastAPI/CSGOptiXService now succeeding to simulate
* get CSGOptiXServiceTest.sh to start to work - testing the high level NP API needed for service-client running
* add high level NP-centric QSim::simulate API for use from CSGOptiXService.h
* review genstep collection and hit handling, conclude that forming sensor frame local hits should happen on client side to avoid doubling hit size
* rework input genstep handling, adding flexibility to enable testing the high level NP-centric API, see CSGOptiXService.h
* rework input genstep preparation for flexibility to support higher level NP-centric API testing
* resolved runtime nanobind type error across python C++ barrier by copying the array frombuffer constructed from the FastAPI request data
* add multipart/form-data uploading to NP_CURL as thats more efficient than urlencoding for large binary uploads
* first impl of NP_CURL.h providing libcurl based array upload/download over HTTP PUT request/response enabling remote array transformation, tested with FastAPI ASGI python webserver endpoint
* initial try at getting complex solid WaterDistributer with MultiUnion within MultiUnion with G4CutTubs that is destined for triangles to convert using CSG_NOTSUPPORTED sn::Notsupported placeholder



v0.5.3 2025/08/27 : shakedown new PMT types WP_ATM_LPMT WP_WAL_PMT, fixes for two longstanding viz issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* add stree::save_desc which writes .txt files for listed desc methods to stree/desc directory which is created when stree::save is invoked
* overhaul extending to PMT types WP_ATM_LPMT WP_WAL_PMT
* add stree__FREQ_CUT envvar control of the instancing criteria
* adhoc workaround longstanding issue of non-reproducible initial viewpoint by ignoring the first few GLFW cursor_moved events
* fix longstanding issue of whacky up direction in vizualization, caused by SGLM::UP.w being incorrectly defaulted to 1.f when should be 0.f for a direction, not a position
* add EXTENT_FUDGE control that when set to 10 for example improves the viz interface when targetting volumes with small extent
* add stree::desc_node_elvid that dumps all snode which have lvid listed within comma delimited ELVID envvar
* add stree::desc_rem/tri/nds using stree::desc_lvid_unique to provide intelligible dumping even with many thousands of snode
* sstamp::FormatLog aka U:FormatLog for NPFold::load lite timestamp logging


v0.5.2 2025/08/19 : intersect precision refinement option plus add handling for new WP PMT type handling via SPMT.h s_pmt.h
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* start adding support for three more PMT types WP_ATM_LPMT WP_ATM_MPMT WP_WAL_PMT
* fixed updating of https://simoncblyth.github.io
* follow suggestion of plexoos to use first device when multiple devices are visible with warning to use CUDA_VISIBLE_DEVICES envvar to quell the warning
* add example of counting pmt categories to SPMTAccessor_test.cc
* add sseq_index_ab__desc_HISWID envvar control for seqhis history width used by sseq_index_test.sh, tidy trailing whitespace in addtag.sh
* expand SRecord::getSimtraceAtTime to working with multiple arange/linspace times specified in OPTICKS_INPUT_PHOTON_RECORD_TIME eg [0.1:88.8:-444],
  use that from CSGOptiX/cxt_precision.sh to estimate intersect precision as function of ray trace distance


v0.5.1 2025/07/17 : record animation as debug tool, add OPTICKS_PROPAGATE_REFINE improving precision of intersect, make sdevice.bin persisting no longer default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* rationalize scontext.h sdevice.h moving all file specifics into sdevice.h and making the sdevice.bin persisting no longer the default
* untested integrate SRecord::getPhotonAtTime by step point interpolation for input time with input photon running, used by providing OPTICKS_INPUT_PHOTON ending with record.npy
* add SRecord::getPhotonAtTime providing photons interpolated from the record array step points
* add small cylinder within the big one for long ray intersect precision test
* add U4VolumeMaker::BigWaterPool as test geometry for long range intersection, kick U4VolumeMakerTest.sh into shape
* add OPTICKS_PROPAGATE_REFINE control to optionally enable refinement of the optixTrace calls when the intersect distance exceeds 99 percent of OPTICKS_PROPAGATE_REFINE_DISTANCE
* add simple, and so far untested, refined trace technique that repeats optixTrace from closer_ray_origin following Ingo Wald, Ch34 of Ray Tracing Gems II
* logging level control SRecord__level SGen__level
* add sevt.py SAB.q_and selecting photon indices based on provided histories in A and B, for example allowing selection of input photons with histories differing in the number of BT between A and B
* add NP::load_data_where used from NP::LoadSlice using a where array eg /tmp/w54.npy that is optionally sliced itself eg /tmp/w54.npy[0:10]
* enhance AFOLD/BFOLD evt plotting on top of the cxt_min simtrace geometry, to see more clearly the WP_PMT BZERO
* extend sseq_array::create_selection to handling a comma delimited string to select the OR of multiple photon histories
* add SemiCircle input photons
* add SideZX input photons to check WP_PMT from side, shows no problem with ipc chi2
* touch control that accepts date-time string to reset CUR backwards to accept earlier screenshots
* add SGen.h SGLFW_Gen.h and shader gl/gen_line_strip for rendering gensteps, not yet working
* fix numeric_limits min is zero ranging bug
* scerenkov::MinMaxPost for genstep extent and time range
* move view config into VUE.sh with VUE bash function to avoid duplication between the renderers
* generalize sphoton::ChangeTimeInsitu to handling float and double photon arrays fixing OPTICKS_INPUT_PHOTON_CHANGE_TIME
* add OPTICKS_INPUT_PHOTON_CHANGE_TIME to change time of all input photons
* use EVT MOI ELV SDR CUR bash functions and optional .sh scripts to reduce duplication between the renderer scripts cxr_min.sh SGLFW_SOPTIX_Scene_test.sh
* add rec_line_strip shader
* add stree::get_frame_from_coords for a frame targetting a global position, change miss zdepth for SOPTIX.cu in order to see event records in front of the miss bkg
* switch to controlling enabled LV via file /home/blyth/.opticks/GEOM/ELV.sh
* add soname to SScene to allow ELV selection within SScene::Load
* move ELV mechanics down to SGeoConfig from CSGFoundry so can use from SScene
* add smath.py with rotateUz impl used to add CircleXZ input photons



v0.5.0 2025/07/02 : improved install cleanliness, add missing gl shaders and python modules to install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* SWITCHED BACK TO STANDARD CUSTOM4 EXTERNAL

* clean build/install with om-prefix-clean opticks-full revealed some python modules missing from install, include them
* change om-prefix-clean to deleting all dirs under prefix other than el9_amd64_gcc11 which is used for test expansion of all tarballs, so after om-prefix-clean now need opticks-full
* change bin/oktar.py to include OpenGL shaders in release tarball
* install ssst.sh as alias for SGLFW_SOPTIX_Scene_test.sh
* add raw translation frame handling for input photons, convenient for global frame


v0.4.9 2025/07/01 : fix muon render kink animation artifact, fix WP PMT qescale giving WP_PMT A:B match
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* RELEASE WAS BUILT AGAINST NON STANDARD DEBUG CUSTOM4 PREFIX WHICH CAUSED CI CMAKE BUILD ERROR

  * NON STANDARD PREFIX : /data1/blyth/local/custom4_Debug/0.1.8/include/Custom4
  * ~/j/oj_cmake_error/oj_cmake_error.rst

* fix another SPMT.h qescale contiguousidx/oldcontiguousidx bug for WP PMT using SPMT::get_pmtid_qescale that A:B matches WP PMT hits
* expand DEBUG_PIDX dumping into qpmt.h
* fix SRecord.h time and position ranges with sphoton::MinMaxPost by excluding unfilled zeros from mn/mx
* fix future kinked muon render bug by excluding zero as a valid time in the rec_flying_point shader
* eliminate the old mixed geom+event sysrap/SGLFW_Event.h in favor of event only sysrap/SGLFW_Evt.h
* add QSim__ALLOC control to dump VRAM allocation salloc.h table before launch
* make pvplt_viewpoint EYE LOOK UP use m2w target transform such that the inputs can remain local with GLOBAL=1 global frame plotting
* move setting of U4Tree into U4Recorder to lower level from U4Tree::initRecorder : this needed for U4Simtrace.h identity of intersects


v0.4.8 2025/06/25 SProcessHits_EPH.h improve handling of large values and legibility of desc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* confirmed fix for muon crash issue in OJ Opticks+JUNOSW, was caused by non-optical particles
  crossing sensitive detectors


v0.4.7 2025/06/25 : fix qe_scale contiguous/oldcontiguous issue getting S_PMT EC/EX to A:B match
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* confirmed fix for qe_scale contiguous/oldcontiguous issue which gets S_PMT EC/EX to match between A and B



v0.4.6 2025/06/24 : within WITH_CUSTOM4 working on WP PMT and SPMT hit matching, plus add EPSILON0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* within WITH_CUSTOM4 try using SPMT qe to replace SD:SURFACE_DETECT with EC:EFFICIENCY_COLLECT/EX:EFFICIENCY_CULL, but currently getting 25% more EC than B side
* bring s_qeshape and s_qescale to GPU with QPMTTest checks
* add ssys::getenviron ssys::countenv ssys::is_under_ctest and use ssys::is_under_ctest detection from SGLFW_SOPTIX_Scene_test to avoid popping up interactive window during ctest running
* add s_qescale for the 25600 S_PMT to SPMT.h
* add X25 to RainXZ input photons to better target S_PMT, add cxs_min.sh input_photon_s_pmt for faster A dev cycle than ipc InputPhotonCheck A:B testing
* adjust s_pmt function names to use pmtid for CD_LPMT+WP_PMT+SPMT and lpmtid used for CD_LPMT + WP_PMT
* add seqhis history slice selection to SRecord::Load used from cxr_min.sh via AFOLD_RECORD_SLICE
* change SRecord::Load to take folder argument rather than path to facilitate seq.npy loading to allow seqhis selection
* moving the ProcessHits EPH flag change from SD to EC/EX into U4Recorder::UserSteppingAction_Optical gets EC/EX into both sides
* switch flag to EC/EX from former SD on A side, requiring OpticksPhoton.h enum reordering to avoid FFS(flag) exceeding 4 bits for EC
* update QPMTTest.sh for WP PMT, enable hits onto WP PMT by allowing qsim::propagate_at_surface_CustomART to proceed with such lpmtid
* rework pmt indexing distinguishing lpmtid and lpmtidx to support WP PMT info together with CD_LPMT, add s_pmt.h to reduce duplication
* add NP::LoadSlice for handling very large arrays by loading only slice specified items using std::ifstream::seekg
* new name NP::LoadThenSlice instead of NP::LoadSlice to make it clear that a full load is done before doing the slicing
* fix sctx.h qsim.h reversion effecting debug arrays from a few days ago : sctx.h needs ctx.idx to be the zero based index but ctx.pidx needs to be absolute
* add SEventConfig::AllocEstimate using salloc.h, aiming to get auto-max-slot-sizing based on VRAM to account for debug arrays
* add SEvt__SAVE_NOTHING control that in OPTICKS_EVENT_MODE of Minimal or Nothing disables SEvt directory creation and saving of run metadata


* use OPTICKS_PROPAGATE_EPSILON0 after OPTICKS_PROPAGATE_EPSILON0_MASK default TO,CK,SI,SC,RE plus use OPTICKS_MAX_TIME truncation together with OPTICKS_MAX_BOUNC

  *  setting OPTICKS_PROPAGATE_EPSILON0 to a smaller value (eg zero) than OPTICKS_PROPAGATE_EPSILON can potentially avoid geometry leaks
     when scatter/generation/reemission happens within OPTICKS_PROPAGATE_EPSILON of boundaries

* add SEventConfig controls OPTICKS_PROPAGATE_EPSILON0 OPTICKS_PROPAGATE_EPSILON0_MASK to enable different epsilon after eg scattering, also add OPTICKS_MAX_TIME renaming old domain settings
* suppress NPFold saving when the fold only contains metadata unless NPFold::set_allowonlymeta_r is used


v0.4.5 2025/06/13 : Theta dependent CE culling on GPU working with qpmt::get_lpmtid_ATQC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* reimpl NPFold::concat less strictly to enable concat of hits when launches are sliced finely resulting in some subfold not having hits
* change ctx.idx to the global photon_idx from the local within the launch idx for more meaningful PIDX dumping
* collect metadata regarding the optixpath mtime into SEvt run metadata from CSGOptiX::initMeta

  * stale optixpath found to be the cause of the muon CUDA crash reported by Haosen, eg "CRASH=1 cxs_min.sh"

* make QSim::simulate handle zero gensteps
* add QSim::MaybeSaveIGS to enable fast cycle input genstep debug of eventID that cause CUDA launch crashes
* use ProcessHits EPH info to change finalPhoton SD flags into EC/EX EFFICIENCY_COLLECT/EFFICIENCY_CULL
* make CE over costh available to qsim.h using cecosth_prop enabling get_lpmtid_stackspec_ce as alternative to get_lpmtid_stackspec_ce_acosf
* change to qpmt::get_lpmtid_ATQC returning absorption,transmission,qe,ce as need to do separate collectionEfficiency throw
* fix NP::FromNumpyString


v0.4.4 2025/06/08
~~~~~~~~~~~~~~~~~~

* switch to collection efficiency scaling using qpmt::get_lpmtid_ARTE_ce from qsim::propagate_at_surface_CustomART, add ce tests to QPMTTest.sh
* revive QPMTTest.sh and add cetheta GPU interpolation test
* add lower level track API to U4Recorder.hh that may enable sharing of Geant4 track info between Opticks and other usage


v0.4.3 2025/05/30
~~~~~~~~~~~~~~~~~~~

* bring SGLFW_SOPTIX_Scene_test.sh into release
* start getting B side simtrace to work with U4Recorder__EndOfRunAction_Simtrace using U4Navigator.h U4Simtrace.h
* enhance A side simtrace analysis cxt_min.sh
* add globalPrimIdx to Binding.h OptiX geometry for debugging
* integrate record rendering with geometry rendering
* move navigation functionality like frame hop and interface control from mains into SGLM.h SGLFW.h
* bring SRecordInfo.h into use


v0.4.2 2025/05/15
~~~~~~~~~~~~~~~~~~

* avoid the slow bash function opticks-setup-find-geant4-prefix when Geant4 env is already present
* remove OPTICKS_MAX_BOUNCE bounce limit instead use inherent SEventConfig::RecordLimit from sseq::SLOTS
* add RandomSpherical1M to input_photons
* add serialization of the full sseq_index AB table into single array with names with the seqhis strings
* create unversioned InputPhotons.tar for deployment to /cvmfs/opticks.ihep.ac.cn/.opticks/InputPhotons
* remove the confusing Default EventMode, set actual default OPTICKS_EVENT_MODE to Minimal, increase MaxBounceDefault from 9 to 31
* add qcf_ab.f90 f2py approach that is more than twice as fast as numpy qcf.py approach but thats nowhere near the CPP approach used by sysrap/sseq_index.h



Snapshot Tags History
----------------------

+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| tag     | OVN | date       | Notes                                                                                                                              |
+=========+=====+============+====================================================================================================================================+
| v0.5.6  | 56  | 2025/10/29 | fix PMT inconsistency bug, file writing changes for production invokation, planning libcurl client                                 |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.5.5  | 55  | 2025/10/23 | removed 32 bit limits on photons per event, improved profiling for production, improved geometry digest, python pointcloud overlap |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.5.4  | 54  | 2025/09/29 | triangulation automation, handle unsupported, U4Polycone generalization for new JUNO geometry, start service-client                |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.5.3  | 53  | 2025/08/27 | shakedown new PMT types WP_ATM_LPMT WP_WAL_PMT, fixes for two longstanding viz issues                                              |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.5.2  | 52  | 2025/08/19 | intersect precision refinement option plus add handling for new WP PMT type handling via SPMT.h s_pmt.h                            |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.5.1  | 51  | 2025/07/17 | record animation as debug, OPTICKS_PROPAGATE_REFINE intersect precision, sdevice.bin persisting no longer default                  |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.5.0  | 50  | 2025/07/02 | improved install cleanliness, add missing gl shaders and python modules to install                                                 |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.4.9  | 49  | 2025/07/01 | fix muon render kink animation artifact, fix WP PMT qescale giving WP_PMT A:B match                                                |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.4.8  | 48  | 2025/06/25 | SProcessHits_EPH.h improve handling of large values and legibility of desc                                                         |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.4.7  | 47  | 2025/06/25 | fix qe_scale contiguous/oldcontiguous issue getting S_PMT EC/EX to A:B match                                                       |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.4.6  | 46  | 2025/06/24 | Within WITH_CUSTOM4 working on WP PMT and SPMT hit matching, plus add EPSILON0                                                     |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.4.5  | 45  | 2025/06/13 | Theta dependent CE culling with qpmt::get_lpmtid_ATQC becoming usable                                                              |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.4.4  | 44  | 2025/06/08 | add collection efficiency scaling from qpmt::get_lpmtid_ARTE_ce, add separate label U4Recorder API                                 |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.4.3  | 43  | 2025/05/30 | integrate OpenGL event record rendering with geometry render, globalPrimIdx added to Binding.h, cxt_min.sh enhance                 |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.4.2  | 42  | 2025/05/15 | remove OPTICKS_MAX_BOUNCE limit, increase default OPTICKS_MAX_BOUNCE from 9 to 31, skip slow find-geant4-prefix                    |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.4.1  | 41  | 2025/04/28 | fix WITH_CUSTOM4 regression and outdated jpmt access in G4CXTest                                                                   |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.4.0  | 40  | 2025/04/24 | last failing release test + avoid some slow tests                                                                                  |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.3.9  | 39  | 2025/04/23 | geom access standardization to enable release ctests                                                                               |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.3.8  | 38  | 2025/04/22 | leap to CMake CUDA LANGUAGE for multi CUDA_ARCHITECTURES compilation                                                               |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.3.7  | 37  | 2025/04/21 | change compute capability target of ptx to 70 to support older GPU                                                                 |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.3.6  | 36  | 2025/04/16 | start getting scripts like cxr_min.sh G4CXTest_raindrop.sh to work from release                                                    |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.3.5  | 35  | 2025/04/06 | okdist tarball standardize labelling, some simtrace revival                                                                        |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.3.4  | 34  | 2025/04/02 | wayland viz fix, handle no CUDA device detected with opticksMode 1                                                                 |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.3.3  | 33  | 2025/03/17 | try to hide non-zero rc in bashrc from the set -e used by gitlab-ci                                                                |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.3.2  | 32  | 2025/03/17 | okdist-- installed tree fixes                                                                                                      |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.3.1  | 31  | 2025/01/11 | fixes BR/BT reversion in v0.3.0                                                                                                    |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.3.0  | 30  | 2025/01/08 | many changes, including jump to Philox RNG + addition of out-of-core running                                                       |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.2.7  | 27  | 2024/02/01 | tag requested by Hans, just for some convenience OpticksPhoton methods                                                             |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.2.6  | 26  | 2024/01/25 | fix VRAM leak by using default CUDA stream for every launch                                                                        |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.2.5  | 25  | 2023/12/19 | fix off-by-one sensor identifier bug                                                                                               |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.2.4  | 24  | 2023/12/18 | fix for tests installation                                                                                                         |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.2.3  | 23  | 2023/12/18 | Addition of smonitor GPU memory monitoring, explicit reset API in QSim and G4CX                                                    |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.2.2  | 22  | 2023/12/14 | Addition of profiling machinery, introduce Release build, fix CK generation bug                                                    |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.2.1  | 21  | 2023/10/20 | Fix stale dependencies issue reported by Hans, remove opticksaux from externals                                                    |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+
| v0.2.0  | 20  | 2023/10/12 | Resume tagging after 2 years of changes : huge change from prior release                                                           |
+---------+-----+------------+------------------------------------------------------------------------------------------------------------------------------------+

OVN: OPTICKS_VERSION_NUMBER

For a record of ancient tags see the "Snapshot pre-History" section at the end of this page.


Workflow for adding "snapshot" tag to github and bitbucket
------------------------------------------------------------

Follow the workflow documented within the "~/opticks/addtag.sh" script



OpticksVersionNumber.hh from OKConf package
------------------------------------------------

::

    epsilon:opticks blyth$ tail -15 okconf/OpticksVersionNumber.hh
    #pragma once

    /**
    OpticksVersionNumber
    =====================

    Definition of version integer

    **/


    #define OPTICKS_VERSION_NUMBER 10



Using **OPTICKS_VERSION_NUMBER**  to navigate API changes
----------------------------------------------------------

::

    epsilon:opticks blyth$ cat sysrap/tests/SOpticksVersionNumberTest.cc

    #include <cstdio>
    #include "OpticksVersionNumber.hh"

    int main()
    {
    #if OPTICKS_VERSION_NUMBER < 10
        printf("OPTICKS_VERSION_NUMBER < 10 \n");
    #elif OPTICKS_VERSION_NUMBER == 10
        printf("OPTICKS_VERSION_NUMBER == 10 \n");
    #elif OPTICKS_VERSION_NUMBER > 10
        printf("OPTICKS_VERSION_NUMBER > 10 \n");
    #else
        printf("OPTICKS_VERSION_NUMBER unexpected \n");
    #endif
        return 0 ;
    }


OKConf/tests related to versioning
---------------------------------------

OpticksVersionNumberTest converts the macro into a string::

    epsilon:okconf blyth$ cat tests/OpticksVersionNumberTest.cc
    #include <cstdio>
    #include "OpticksVersionNumber.hh"

    #define xstr(s) str(s)
    #define str(s) #s

    int main()
    {
        printf("%s\n",xstr(OPTICKS_VERSION_NUMBER));
        return 0 ;
    }


The exeutable enables bash scripts to access the version::

    epsilon:opticks blyth$ ver=$(OpticksVersionNumberTest)
    epsilon:opticks blyth$ echo $ver
    10


OKConfTest dumps version integers using static functions such as  OKConf::OpticksVersionInteger()::

    epsilon:opticks blyth$ OKConfTest
    OKConf::Dump
                      OKConf::OpticksVersionInteger() 10
                       OKConf::OpticksInstallPrefix() /usr/local/opticks
                            OKConf::CMAKE_CXX_FLAGS()  -fvisibility=hidden -fvisibility-inlines-hidden -fdiagnostics-show-option -Wall -Wno-unused-function -Wno-unused-private-field -Wno-shadow
                         OKConf::CUDAVersionInteger() 9010
                   OKConf::ComputeCapabilityInteger() 30
                            OKConf::OptiXInstallDir() /usr/local/optix
                         OKCONF_OPTIX_VERSION_INTEGER 50001
                        OKConf::OptiXVersionInteger() 50001
                         OKCONF_OPTIX_VERSION_MAJOR   5
                          OKConf::OptiXVersionMajor() 5
                         OKCONF_OPTIX_VERSION_MINOR   0
                          OKConf::OptiXVersionMinor() 0
                         OKCONF_OPTIX_VERSION_MICRO   1
                          OKConf::OptiXVersionMicro() 1
                       OKConf::Geant4VersionInteger() 1042
                       OKConf::ShaderDir()            /usr/local/opticks/gl

     OKConf::Check() 0



Git tags
-----------

List tags with "git tag" or "git tag -l"::

    epsilon:opticks blyth$ git tag -l
    v0.0.0-rc1
    v0.0.0-rc2
    v0.0.0-rc3
    v0.1.0-rc1
    v0.1.0-rc2




Snapshot pre-History
----------------------

* *NB : IT WOULD BE VERY UNWISE TO ATTEMPT TO USE ANY OF THESE ANCIENT SNAPSHOTS*

+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+
| date       | tag     | OPTICKS_VERSION_NUMBER  | GEOCACHE_CODE_VERSION      | Notes                                                                           |
+============+=========+=========================+============================+=================================================================================+
| 2021/08/28 | v0.1.1  | 11                      | 14                         | Fermilab Geant4 team request, severe Cerenkov Wavelength bug found, DO NOT USE  |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+
| 2021/08/30 | v0.1.2  | 12                      | 14                         | Fixed Cerenkov wavelength bug                                                   |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+
| 2021/09/02 | v0.1.3  | 13                      | 14                         | Fixed minor CManager bug                                                        |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+
| 2021/09/24 | v0.1.4  | 14                      | 14                         | Changes for Geant4 1100 beta, 4 cfg4 test fails remain, needing G4 GDML read fix|
|            |         |                         |                            | see notes/issues/Geant4_1100_GDML_AddProperty_error.rst                         |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+
| 2021/09/30 | v0.1.5  | 15                      | 14                         | All use of G4PhysicsVector::SetSpline removed due to Geant4 API change,         |
|            |         |                         |                            | see notes/issues/Geant4_Soon_SetSpline_change.rst                               |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+
| 2021/10/06 | v0.1.6  | 16                      | 14                         | More updates for Geant4 API in flux and fixing test fails,                      |
|            |         |                         |                            | see notes/issues/Geant4_Soon_GetMinLowEdgeEnergy.rst                            |
+------------+---------+-------------------------+----------------------------+---------------------------------------------------------------------------------+





