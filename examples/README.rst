Examples of Opticks Usage, especially the CMake machinery  
=============================================================

The examples have go.sh scripts which perform the normal 
config, make, install using a fresh build dir every time.
This is a convenient when the focus is on the CMake machinery. 

UsePLog
    simple executable using PLog via imported interface target 
    from cmake/Modules/FindPLog.cmake

 
UseGLM
   (plain vanilla modern CMake with target import/export)

   * find_package(GLM) succeeds due to cmake/Modules/FindGLM.cmake which 
     exports the Opticks::GLM interface only target 
     (GLM is just headers) 

   * vending targets as of CMake 3.5 takes lots of boilerplate



UseUseGLM
   (plain vanilla modern CMake)

   * find_package(UseGLM) succeed due to the exported target written by UseGLM

   * nice and simple : plain CMake is fine when consuming 
     targets only without vending.  

   * Note that only need to be concerned with direct dependency on UseGLM.
     The dependency of UseGLM on GLM comes along with the imported target with no effort.

   * Also note that the dependency transmission is working across package boundaries, 
     ie UseGLM and UseUseGLM are not tied together in a single CMake build, 
     they are entirely separate projects.



UseGLMViaBCM
    (modern CMake assisted by BCM)

   * much less boilerplate, intent is clear

   * forced(?) to place headers into an include folder in source dir 
   * forced(?) in install at prefix/include rather than current opticks location prefix/include/name/

   * HMM: NOT KEEN ON CO-MINGLING HEADERS OF ALL PROJECTS IN A SINGLE include DIR, OR ON HAVING 
     THE HASSLE OF SEPARATE include DIR for sources 
   

UseUseGLMViaBCM
    
   * similar observations to UseUseGLM


UseSysRap
   now using import of targets exported by sysrap/CMakeLists.txt  


UseNPY(needs-revisit)
    old first attempt using raw inclusion of exported targets with 
    the non-standard approach of cmake -DOPTICKS_CONFIG_DIR=/usr/local/opticks/config
    rather than using the standard find_package mechanism 

    * exporting was done in opticksnpy/CMakeLists.txt
      BUT: this approach is jumping into the thicket of the dependency tree.  Better to 
      get experience out in the leaves, before tackling the interior of the bush.  






