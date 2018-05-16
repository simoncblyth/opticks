bcm-src(){      echo externals/bcm.bash ; }
bcm-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(bcm-src)} ; }
bcm-vi(){       vi $(bcm-source) ; }
bcm-usage(){ cat << EOU

BCM : Boost CMake Modules
===========================

This provides cmake modules that can be re-used by boost and other
dependencies. It provides modules to reduce the boilerplate for installing,
versioning, setting up package config, and creating tests.

* https://github.com/boost-cmake/bcm
* https://readthedocs.org/projects/bcm/
* http://bcm.readthedocs.io/en/latest/

Very clear explanation describing a standalone CMake setup for building boost_filesystem

* http://bcm.readthedocs.io/en/latest/src/Building.html#building-standalone-with-cmake

bcm_auto_export
----------------

* https://github.com/boost-cmake/bcm/blob/master/share/bcm/cmake/BCMExport.cmake

FUNCTIONS
-----------

bcm--
   clones bcm from github, configures, "builds" and installs the share/bcm/cmake/ 
   directory contents such as BCMDeploy.cmake 
   beneath opticks-prefix (typically /usr/local/opticks) 

   share/bcm/cmake/BCMDeploy.cmake



EOU
}

bcm-env(){  olocal- ; opticks- ; }
bcm-view-(){ ls -1 $(opticks-prefix)/share/bcm/cmake/* ; }
bcm-view(){ vim -R $($FUNCNAME-) ; }
bcm-info(){ cat << EOI

    url  : $(bcm-url)
    dist : $(bcm-dist)
    base : $(bcm-base)
    dir  : $(bcm-dir)
    bdir : $(bcm-bdir)
    idir : $(bcm-idir)


EOI
}



bcm-url(){ echo http://github.com/simoncblyth/bcm.git ; }   

bcm-base(){   echo $(opticks-prefix)/externals/bcm ; }
bcm-prefix(){ echo $(opticks-prefix) ; }
bcm-idir(){   echo $(bcm-prefix) ; }

bcm-dir(){  echo $(bcm-base)/bcm ; }
bcm-bdir(){ echo $(bcm-base)/bcm.build ; }

bcm-ecd(){  cd $(bcm-edir); }
bcm-cd(){   cd $(bcm-dir)/$1 ; }
bcm-bcd(){  cd $(bcm-bdir); }
bcm-icd(){  cd $(bcm-idir); }

bcm-get(){
   local iwd=$PWD
   local dir=$(dirname $(bcm-dir)) &&  mkdir -p $dir && cd $dir
   if [ ! -d "bcm" ]; then 
       git clone $(bcm-url)
   fi 
   cd $iwd
}

bcm-wipe(){
  local bdir=$(bcm-bdir)
  rm -rf $bdir 
}

bcm-cmake(){
  local iwd=$PWD
  local bdir=$(bcm-bdir)
  mkdir -p $bdir
  bcm-bcd
  cmake $(bcm-dir) -DCMAKE_INSTALL_PREFIX=$(bcm-prefix) 
  cd $iwd
}

bcm-configure()
{
   bcm-wipe
   bcm-cmake $*
}

bcm-make(){
  local iwd=$PWD
  bcm-bcd
  cmake --build . --target ${1:-install}
  cd $iwd
}

bcm--(){
  bcm-get 
  bcm-cmake
  bcm-make install
}


