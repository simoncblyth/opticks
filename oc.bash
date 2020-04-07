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

   /usr/local/opticks/externals/lib/pkgconfig/assimp.pc
   /usr/local/opticks/externals/lib/pkgconfig/glfw3.pc
   /usr/local/opticks/externals/lib/pkgconfig/glew.pc

The edits move the "externals" from the prefix into the libdir and includedir.

The reason for this is because are using pkg-config with --define-prefix 
in order to work in a relocatable way for distributions.


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



Hmm always using define-prefix means have to get rid of prefix var in below ?
------------------------------------------------------------------------------

* suppose use of macports xerces-c means have to add /opt/local/lib/pkgconfig 
  to PKG_CONFIG_PATH so --define-prefix then can operate correctly without changes
  to the pc

* hmm but its problematic from a cross platform point of view


::

    epsilon:UseOpticksXercesC blyth$ cat /opt/local/lib/pkgconfig/xerces-c.pc
    prefix=/opt/local
    exec_prefix=${prefix}
    libdir=${exec_prefix}/lib
    includedir=${prefix}/include

    Name: Xerces-C++
    Description: Validating XML parser library for C++
    Version: 3.2.1
    Libs: -L${libdir} -lxerces-c
    Libs.private: -lcurl
    Cflags: -I${includedir}




FIXED : define-prefix is scrubbing the CUDA include dir ?
-----------------------------------------------------------

define-prefix assumes the prefix can be obtained from the grandparent
dir of the /usr/local/opticks/xlib/pkgconfig/optickscuda.pc 
which yeilds /usr/local/opticks. This replaces the prefix variable 
if there is one defined in the pc file.

/usr/local/opticks/xlib/pkgconfig/optickscuda.pc::

    prefix=/usr/local/cuda
    includedir=${prefix}/include
    libdir=${prefix}/lib

    Name: CUDA
    Description: 
    Version: 9.1 
    Libs: -L${libdir} -lcudart -lcurand
    Cflags: -I${includedir}

The result is the wrong prefix.::

    epsilon:UseCUDARapNoCMake blyth$ oc-pkg-config optickscuda --cflags
    -I/usr/local/cuda/include

    epsilon:UseCUDARapNoCMake blyth$ oc-pkg-config optickscuda --cflags --define-prefix
    -I/usr/local/opticks/include

One solution is to exclude the prefix variable in pc files
of packages that are not going to be part of the distribution.  
This prevents --define-prefix from having any effect.
    
EOU
} 


oc-libdir0-default(){ cat << EOD
$(opticks-prefix)/lib
$(opticks-prefix)/externals/lib 
EOD
}

oc-libdir0-(){ oc-pkg-config $* --variable=libdir --define-prefix ; } ## hmm relies on pc variable libdir
oc-libdir0(){ 
   if [ $# == 0 ]; then 
      oc-libdir0-default
   else
      oc-libdir0-  $* | tr " " "\n" | sort | uniq  
   fi 
}

oc-libdir-(){ oc-pkg-config $* --libs-only-L --define-prefix | tr -- "-L" " " ; }
oc-libdir(){  oc-libdir- $* | tr " " "\n" | sort | uniq ; }

oc-libpath(){ local dirs=$(oc-libdir $*) ; echo $dirs | tr " " ":" ; }

oc-cflags-(){ oc-pkg-config $* --cflags --define-prefix ; }

# "public" interface
oc-cflags(){ echo $(oc-cflags- $*) -std=c++11 ; }

oc-libs(){   oc-pkg-config $* --libs   --define-prefix ; }
oc-libsl(){  oc-pkg-config $* --libs-only-L --define-prefix ; }
oc-deps(){   oc-pkg-config $* --print-requires  ; }

oc-dump(){   oc-pkg-config-dump $* ; }
oc-check(){  oc-pkg-config-check-dirs $* ; }
oc-find(){   oc-pkg-config-find $* ; }
oc-cat(){    
   local pc=$(oc-find $*) 
   [ -n "$pc" -a -f "$pc" ] && cat $pc
}

# "private" interface
oc-pkg-config-path--(){ cat << EOP
$(opticks-prefix)/lib/pkgconfig
$(opticks-prefix)/xlib/pkgconfig
/opt/local/lib/pkgconfig
EOP
}
oc-pkg-config-path-(){
   $FUNCNAME- | while read dir ; do
      [ -d "$dir" ] && echo $dir
   done
}
oc-pkg-config-path(){ echo $(oc-pkg-config-path-) | tr " " ":" ; }



oc-pkg-config(){ PKG_CONFIG_PATH=$(oc-pkg-config-path) pkg-config $* ; }

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
      echo $FUNCNAME failed for pkg $pkg ipc $ipc xpc $xpc 1>&2
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

