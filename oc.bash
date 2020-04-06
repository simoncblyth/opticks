oc-source(){ echo $BASH_SOURCE ; }
oc-vi(){ vi $(oc-source) ; }
oc-env(){  olocal- ; opticks- ; }
oc-usage(){ cat << EOU

OC : Opticks Config Based on pkg-config
===========================================================

* NB this aspires to becoming a standalone opticks-config script, 
  so keep use of other scripts to a minimum


TODO 
-----

Avoid manual edits of::

   externals/lib/pkgconfig/assimp.pc


Requirements for pkg-config hookup
-------------------------------------

1. all packages (internal and external) need a .pc file to be 
   installed into lib/pkgconfig or externals/lib/pkgconfig : often 
   the simplest way to do that is via the bash functions that install
   the package 
  

2. CMake machinery needs to be informed for externals by addition 
   of the INTERFACE_PKG_CONFIG_NAME property to found targets, this 
   is most conveniently done in cmake/modules/FindName.cmake 

   The name corresponding to pkg-config pc file eg glm.pc, assimp.pc 

::

     39     set_target_properties(${_tgt} PROPERTIES
     40         INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIR}"
     41         INTERFACE_PKG_CONFIG_NAME "glm"
     42     )


   Despite pkg-config being usable without CMake the Opticks 
   build remains CMake based and most of the .pc files are generated
   by the BCMPkgConfig CMake machinery. It is because of this that 
   the CMake build needs to the INTERFACE_PKG_CONFIG_NAME for 
   inclusion into the generated pc files.

   After adding that property, need to rebuild and install 
   packages that use the dependency in order to re-generate the pc files.


Typical Usage
----------------

::

    opticks-
    oc-

    pkg=OpenMeshRap

    gcc -c $sdir/Use$pkg.cc $(oc-cflags $pkg)
    gcc Use$pkg.o -o Use$pkg $(oc-libs $pkg) 
    LD_LIBRARY_PATH=$(oc-libpath $pkg) ./Use$pkg



FIXED : define-prefix is scrubbing the CUDA include dir ?
-----------------------------------------------------------

define-prefix assumes the prefix can be obtained from the 
location of the /usr/local/opticks/libs/pkgconfig/optickscuda.pc 

/usr/local/opticks/externals/lib/pkgconfig/optickscuda.pc
/usr/local/opticks/xlib/pkgconfig/optickscuda.pc::

    prefix=/usr/local/cuda
    includedir=${prefix}/include
    libdir=${prefix}/lib

    Name: CUDA
    Description: 
    Version: 9.1 
    Libs: -L${libdir} -lcudart -lcurand
    Cflags: -I${includedir}

::

    epsilon:UseCUDARapNoCMake blyth$ oc-pkg-config optickscuda --cflags
    -I/usr/local/cuda/include

    epsilon:UseCUDARapNoCMake blyth$ oc-pkg-config optickscuda --cflags --define-prefix
    -I/usr/local/opticks/include

Solution is to not use the variables (prefix etc..) 
for packages that are not going to be part of the distribution.  
This prevents --define-prefix from having any effect.
  




    
EOU
} 


#oc-pkg-config(){ PKG_CONFIG_PATH=$(opticks-prefix)/lib/pkgconfig:$(opticks-prefix)/externals/lib/pkgconfig pkg-config $* ; }
oc-pkg-config(){ PKG_CONFIG_PATH=$(opticks-prefix)/lib/pkgconfig:$(opticks-prefix)/xlib/pkgconfig pkg-config $* ; }
oc-libpath(){ echo $(opticks-prefix)/lib:$(opticks-prefix)/externals/lib ; }


# "public" interface
oc-cflags(){ oc-pkg-config $1 --cflags --define-prefix ; }
oc-libs(){   oc-pkg-config $1 --libs   --define-prefix ; }
oc-dump(){   oc-pkg-config-dump $1 ; }



oc-pkg-config-find(){
   local pkg=${1:-NPY}
   local lpkg=$(echo $pkg | tr [A-Z] [a-z])

   local ipc=$(opticks-prefix)/lib/pkgconfig/${lpkg}.pc
   local xpc=$(opticks-prefix)/xlib/pkgconfig/${lpkg}.pc

   if [ -f "$ipc" ]; then
      echo $ipc
   elif [ -f "$xpc" ]; then
      echo $xpc
   else
      echo $FUNCNAME failed for pkg $pkg ipc $ipc xpc $xpc
   fi 
}

oc-pkg-config-dump(){
   local pkg=${1:-NPY}
   local opt
   $FUNCNAME-opts- | while read opt ; do 
       cmd="oc-pkg-config $pkg $opt"  
       printf "\n\n# %s\n\n"  "$cmd"
       $cmd | tr " " "\n"
   done
}

oc-pkg-config-dump-opts-(){ cat << EOC
--print-requires --define-prefix
--cflags 
--cflags --define-prefix
--libs 
--libs --define-prefix
--cflags-only-I --define-prefix
EOC
}

oc-pkg-config-check-dirs(){
   local pkg=${1:-NPY}
   local line 
   local dir
   local exists 
   oc-pkg-config $pkg --cflags-only-I --define-prefix | tr " " "\n" | while read line ; do 
     dir=${line:2} 
     [ -d "$dir" ] && exists="Y" || exists="N"  
     printf " %s : %s \n" $exists $dir 
   done 
}



