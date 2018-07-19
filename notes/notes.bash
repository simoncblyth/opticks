notes-rel(){      echo notes ; }
notes-src(){      echo notes/notes.bash ; }
notes-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(notes-src)} ; }
notes-vi(){       vi $(notes-source) ; }
notes-usage(){ cat << \EOU

notes- : Opticks Notes
=====================================

Most notes on the research that goes into Opticks are 
in env- repo, see opticksdev- for an index of these.
However some notes related to Opticks progress are kept here.


Monthly Progress 
------------------

::

    2015 May : GPU textures for materials, geocache, ImGui
    2015 Jun : develop compressed photon record, learn Thrust 
    2015 Jul : photon index, propagation histories, Linux port
    2015 Aug : big geometry handling with Instancing
    2015 Sep : thrust for GPU resident photons, OpenMesh for meshfixing
    2015 Oct : meshfixing, instanced identity, start analytic partitioning
    2015 Nov : refactor for dynamic boundaries, Fresnel reflection matching, PMT uncoincidence
    2015 Dec : matching against theory for prism, rainbow
    2016 Jan : Bookmarks, viewpoint animation, presentations
    2016 Feb : partitioned analytic geometry, compositing raytrace and rasterized viz
    2016 Mar : Opticks/G4 PMT matching, GPU textures, making movie 
    2016 Apr : build structure make to CMake superbuild, spawn Opticks repo

    /usr/local/workflow/admin/reps/latex/ntu-report-may-2016.pdf

    2016 May : CTests, CFG4 GDML handling, non-GPU photon indexing
    2016 Jun : porting to Windows
    2016 Jul : porting to Windows and Linux, Linux interop debug
    2016 Aug : OpticksEvent handling, high level app restructure along lines of dependency
    2016 Sep : mostly G4/Opticks interop
    2016 Oct : G4/Opticks optical physics chisq minimization
    2016 Nov : G4/Opticks optical physics chisq minimization
    2016 Dec : g4gun, CSG research

    /usr/local/workflow/admin/reps/latex/ntu-report-dec-2016.pdf

    2017 Jan : presentations, proceedings, holidays
    2017 Feb : GPU CSG raytracing prototyping
    2017 Mar : GPU CSG raytracing implementation, SDF modelling, MC and DCS polygonization of CSG trees 
    2017 Apr : better polygonization with IM, applying GPU CSG to detdesc and GDML, adding primitives



2015 May - 2016 Apr :  Opticks development, G4 matching with simple test geometries
--------------------------------------------------------------------------------------


2016 May - 2017 Apr :  Project Infrastructure, Optical Physics Matching, Geometry Modelling 
-------------------------------------------------------------------------------------------------------------------



* Project Infrastructure

  * refactoring for dynamic testing  
  * porting : code quality
  * CTesting
  * CMake-ing 

* Optical Physics Validation 

  * matching G4/Opticks optical physics within analytic geometries

* Geometry Modelling to avoid tessellation approximations

  * GPU CSG prototyping and development
  * SDF modelling
  * polygonization





EOU
}


notes-env(){
    olocal-
    opticks-
}

notes-dir(){  echo $(opticks-home)/notes ; }
notes-cd(){   cd $(notes-dir); }

notes-progress-(){ echo $(notes-dir)/progress.rst ; }
notes-progress(){ cat $(notes-progress-) | progress.py ; }
notes-edit(){  vi $(notes-progress-) ; }


