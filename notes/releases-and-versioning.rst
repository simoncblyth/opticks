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

+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| date       | tag     | OPTICKS_VERSION_NUMBER  | Notes                                                                                                               |
+============+=========+=========================+=====================================================================================================================+
| 2025/06/25 | v0.4.8  | 48                      | SProcessHits_EPH.h improve handling of large values and legibility of desc                                          |  
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/06/25 | v0.4.7  | 47                      | fix qe_scale contiguous/oldcontiguous issue getting S_PMT EC/EX to A:B match                                        |  
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/06/24 | v0.4.6  | 46                      | Within WITH_CUSTOM4 working on WP PMT and SPMT hit matching, plus add EPSILON0                                      |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/06/13 | v0.4.5  | 45                      | Theta dependent CE culling with qpmt::get_lpmtid_ATQC becoming usable                                               |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/06/08 | v0.4.4  | 44                      | add collection efficiency scaling from qpmt::get_lpmtid_ARTE_ce, add separate label U4Recorder API                  |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/05/30 | v0.4.3  | 43                      | integrate OpenGL event record rendering with geometry render, globalPrimIdx added to Binding.h, cxt_min.sh enhance  |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/05/15 | v0.4.2  | 42                      | remove OPTICKS_MAX_BOUNCE limit, increase default OPTICKS_MAX_BOUNCE from 9 to 31, skip slow find-geant4-prefix     |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/04/28 | v0.4.1  | 41                      | fix WITH_CUSTOM4 regression and outdated jpmt access in G4CXTest                                                    |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/04/24 | v0.4.0  | 40                      | last failing release test + avoid some slow tests                                                                   |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/04/23 | v0.3.9  | 39                      | geom access standardization to enable release ctests                                                                |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/04/22 | v0.3.8  | 38                      | leap to CMake CUDA LANGUAGE for multi CUDA_ARCHITECTURES compilation                                                |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/04/21 | v0.3.7  | 37                      | change compute capability target of ptx to 70 to support older GPU                                                  |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/04/16 | v0.3.6  | 36                      | start getting scripts like cxr_min.sh G4CXTest_raindrop.sh to work from release                                     |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/04/06 | v0.3.5  | 35                      | okdist tarball standardize labelling, some simtrace revival                                                         |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/04/02 | v0.3.4  | 34                      | wayland viz fix, handle no CUDA device detected with opticksMode 1                                                  |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/03/17 | v0.3.3  | 33                      | try to hide non-zero rc in bashrc from the set -e used by gitlab-ci                                                 |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/03/17 | v0.3.2  | 32                      | okdist-- installed tree fixes                                                                                       |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/01/11 | v0.3.1  | 31                      | fixes BR/BT reversion in v0.3.0                                                                                     |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2025/01/08 | v0.3.0  | 30                      | many changes, including jump to Philox RNG + addition of out-of-core running                                        |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2024/02/01 | v0.2.7  | 27                      | tag requested by Hans, just for some convenience OpticksPhoton methods                                              |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2024/01/25 | v0.2.6  | 26                      | fix VRAM leak by using default CUDA stream for every launch                                                         |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2023/12/19 | v0.2.5  | 25                      | fix off-by-one sensor identifier bug                                                                                |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2023/12/18 | v0.2.4  | 24                      | fix for tests installation                                                                                          |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2023/12/18 | v0.2.3  | 23                      | Addition of smonitor GPU memory monitoring, explicit reset API in QSim and G4CX                                     |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2023/12/14 | v0.2.2  | 22                      | Addition of profiling machinery, introduce Release build, fix CK generation bug                                     |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2023/10/20 | v0.2.1  | 21                      | Fix stale dependencies issue reported by Hans, remove opticksaux from externals                                     |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+
| 2023/10/12 | v0.2.0  | 20                      | Resume tagging after 2 years of changes : huge change from prior release                                            |
+------------+---------+-------------------------+---------------------------------------------------------------------------------------------------------------------+

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





