oks-vi(){ vi $(opticks-home)/bin/oks.bash ; }
oks-env(){ echo -n ; }
oks-usage(){  cat << \EOU

Opticks Development Functions
===============================

General usage functions belong in opticks- 
development functions belong here.


oks-i
    edit FindX.cmake for internals     
oks-x
    edit FindX.cmake for xternals     
oks-o
    edit FindX.cmake for others

oks-bash
    edit bash scripts for interal projects 

oks-txt
    edit CMakeLists.txt for interal projects 


issues
---------

pthreads cmake issue
~~~~~~~~~~~~~~~~~~~~~~~


* https://github.com/casacore/casacore/issues/104

Even after excluding boost component "thread" cmake still looks for pthread.h and pthread_create::

    -- Boost version: 1.57.0
    -- Found the following Boost libraries:
    --   system
    --   program_options
    --   filesystem
    --   regex
    -- Configuring SysRap
    -- Configuring BoostRap
    -- Configuring NPY
    -- Configuring OpticksCore
    -- Configuring GGeo
    -- Configuring AssimpRap
    -- Configuring OpenMeshRap
    -- Configuring OpticksGeometry
    -- Configuring OGLRap
    -- Looking for pthread.h
    -- Looking for pthread.h - found
    -- Looking for pthread_create
    -- Looking for pthread_create - found
    -- Found Threads: TRUE  






EOU
}


oks-dirs(){  cat << EOL
sysrap
boostrap
opticksnpy
optickscore
ggeo
assimprap
openmeshrap
opticksgeo
oglrap
cudarap
thrustrap
optixrap
opticksop
opticksgl
ggeoview
cfg4
EOL
}

oks-xnames(){ cat << EOX
boost
glm
plog
gleq
glfw
glew
imgui
assimp
openmesh
cuda
thrust
optix
xercesc
g4
zmq
asiozmq
EOX
}

oks-internals(){  cat << EOI
SysRap
BoostRap
NPY
OpticksCore
GGeo
AssimpRap
OpenMeshRap
OpticksGeo
OGLRap
CUDARap
ThrustRap
OptiXRap
OpticksOp
OpticksGL
GGeoView
CfG4
EOI
}
oks-xternals(){  cat << EOX
OpticksBoost
Assimp
OpenMesh
GLM
GLEW
GLEQ
GLFW
ImGui
EnvXercesC
G4DAE
ZMQ
AsioZMQ
EOX
}
oks-other(){  cat << EOO
OpenVR
CNPY
NuWaCLHEP
NuWaGeant4
cJSON
RapSqlite
SQLite3
ChromaPhotonList
G4DAEChroma
NuWaDataModel
ChromaGeant4CLHEP
CLHEP
ROOT
ZMQRoot
EOO
}




oks-tags()
{
   local iwd=$PWD

   opticks-scd

   local hh

   local dir
   oks-dirs | while read dir ; do
       local hh=$(ls -1  $dir/*_API_EXPORT.hh 2>/dev/null)
       local proj=$(dirname $hh)
       local name=$(basename $hh)
       local utag=${name/_API_EXPORT.hh}
       local tag=$(echo $utag | tr "A-Z" "a-z" )
       printf "%20s %30s \n" $proj $utag 
   done

   cd $iwd
}


oks-find-cmake-(){ 
  local f
  local base=$(opticks-home)/CMake/Modules
  local name
  oks-${1} | while read f 
  do
     name=$base/Find${f}.cmake
     [ -f "$name" ] && echo $name
  done 
}

oks-i(){ vi $(oks-find-cmake- internals) ; }
oks-x(){ vi $(oks-find-cmake- xternals) ; }
oks-o(){ vi $(oks-find-cmake- other) ; }

oks-edit(){  opticks-scd ; vi opticks.bash $(oks-bash-list) CMakeLists.txt $(oks-txt-list) ; } 
oks-txt(){   opticks-scd ; vi CMakeLists.txt $(oks-txt-list) ; }
oks-bash(){  opticks-scd ; vi opticks.bash $(oks-bash-list) ; }
oks-tests(){ opticks-scd ; vi $(oks-tests-list) ; } 


oks-name(){ echo Opticks ; }
oks-sln(){ echo $(opticks-bdir)/$(opticks-name).sln ; }
oks-slnw(){  vs- ; echo $(vs-wp $(opticks-sln)) ; }
oks-vs(){ 
   # hmm vs- is from env-
   vs-
   local sln=$1
   [ -z "$sln" ] && sln=$(opticks-sln) 
   local slnw=$(vs-wp $sln)

    cat << EOC
# sln  $sln
# slnw $slnw
# copy/paste into powershell v2 OR just use opticks-vs Powershell function
vs-export 
devenv /useenv $slnw
EOC

}


oks-backup-externals()
{
   # backup externals incase clouds are inaccessible
   # and need to use opticks-nuclear to test complete builds
   cd $LOCAL_BASE
   [ ! -d "opticks_backup" ] && return 
   cp -R opticks/externals opticks_backup/
} 



oks-txt-list(){
  local dir
  oks-dirs | while read dir 
  do
      echo $dir/CMakeLists.txt
  done
}
oks-tests-list(){
  local dir
  local name
  oks-dirs | while read dir 
  do
      name=$dir/tests/CMakeLists.txt
      [ -f "$name" ] && echo $name
  done

}
oks-bash-list(){
  local dir
  local home=$(opticks-home)
  oks-dirs | while read dir 
  do
      ## project folders should have only one .bash excluding any *dev.bash
      local rel=$(ls -1 $home/$dir/*.bash 2>/dev/null | grep -v dev.bash) 

      if [ ! -z "$rel" -a -f "$rel" ]; 
      then
          echo $rel
      else
          echo MISSING $rel
      fi
  done
}
oks-grep()
{
   local iwd=$PWD
   local msg="=== $FUNCNAME : "
   opticks-
   local dir 
   local base=$(opticks-home)
   oks-dirs | while read dir 
   do
      local subdirs="${base}/${dir} ${base}/${dir}/tests"
      local sub
      for sub in $subdirs 
      do
         if [ -d "$sub" ]; then 
            cd $sub
            #echo $msg $sub
            grep $* $PWD/*.*
         fi
      done 
   done
   cd $iwd
}


oks-api-export()
{
  opticks-cd 
  local dir
  oks-dirs | while read dir 
  do
      local name=$(ls -1 $dir/*_API_EXPORT.hh 2>/dev/null) 
      [ ! -z "$name" ] && echo $name
  done
}

oks-api-export-vi(){ vi $(oks-api-export) ; }




oks-grep-vi(){ vi $(oks-grep -l ${1:-BLog}) ; }



oks-genrst-(){ local hdr=$1 ; cat << EOR

.. include:: $hdr
   :start-after: /**
   :end-before: **/

EOR
}
oks-genrst()
{
   local hdr=${1:-CGDMLDetector.hh}
   local stem=${hdr/.hh}
   stem=${stem/.hpp} 

   [ ! -f "$hdr" ] && echo $msg header name argument expected to exist hdr $hdr  && return 
   [ "$stem" == "$hdr" ] && echo $msg BAD argument expecting header with name ending .hh or .hpp hdr $hdr stem $stem && return 

   local rst=$stem.rst
   #echo $msg hdr $hdr stem $stem rst $rst 

   [ -f "$rst" ] && echo $msg rst $rst exists already : SKIP && return 

   echo $msg writing sphinx docs shim for hdr $hdr to rst $rst 
   oks-genrst- $hdr > $rst 

}

oks-genrst-auto()
{
   # write the shims for all headers containing "/**"
   local hh=${1:-hh}
   local hdr
   grep -Fl "/**" *.$hh | while read hdr ; do
       oks-genrst $hdr
   done
}

oks-docsrc()
{
   grep -Fl "/**" *.{hh,hpp,cc,cpp} 2>/dev/null | while read hdr ; do
      echo $hdr
   done
}

oks-docvi(){ vi $(oks-docsrc) ; }

########## building sphinx docs

oks-htmldir(){   echo $(opticks-prefix)/html ; }
oks-htmldirbb(){ echo $HOME/simoncblyth.bitbucket.org/opticks ; }
oks-docs()
{
   local iwd=$PWD
   opticks-scd
   local htmldir=$(oks-htmldir)
   local htmldirbb=$(oks-htmldirbb)

   [ -d "$htmldirbb" ] && htmldir=$htmldirbb

   sphinx-build -b html  . $htmldir
   cd $iwd

   open $htmldir/index.html
}

oks-html(){   open $(oks-htmldir)/index.html ; } 
oks-htmlbb(){ open $(oks-htmldirbb)/index.html ; } 








oks-genproj()
{
    # this is typically called from projs like ggeo- 

    local msg=" === $FUNCNAME :"
    local proj=${1}
    local tag=${2}

    [ -z "$proj" -o -z "$tag" ] && echo $msg need both proj $proj and tag $tag  && return 


    importlib-  
    importlib-exports ${proj} ${tag}_API

    plog-
    plog-genlog

    echo $msg merge the below sources into CMakeLists.txt
    oks-genproj-sources- $tag

}



oks-genlog()
{
    opticks-scd 
    local dir
    plog-
    oks-dirs | while read dir 
    do
        opticks-scd $dir

        local name=$(ls -1 *_API_EXPORT.hh 2>/dev/null) 
        [ -z "$name" ] && echo MISSING API_EXPORT in $PWD && return 
        [ ! -z "$name" ] && echo $name

        echo $PWD
        plog-genlog FORCE
    done
}


oks-genproj-sources-(){ 


   local tag=${1:-OKCORE}
   cat << EOS

set(SOURCES
     
    ${tag}_LOG.cc

)
set(HEADERS

    ${tag}_LOG.hh
    ${tag}_API_EXPORT.hh
    ${tag}_HEAD.hh
    ${tag}_TAIL.hh

)
EOS
}


oks-testname(){ echo ${cls}Test.cc ; }
oks-gentest()
{
   local msg=" === $FUNCNAME :"
   local cls=${1:-GMaterial}
   local tag=${2:-GGEO} 

   [ -z "$cls" -o -z "$tag" ] && echo $msg a classname $cls and project tag $tag must be provided && return 
   local name=$(oks-testname $cls)
   [ -f "$name" ] && echo $msg a file named $name exists already in $PWD && return
   echo $msg cls $cls generating test named $name in $PWD
   oks-gentest- $cls $tag > $name
   #cat $name

   vi $name

}
oks-gentest-(){

   local cls=${1:-GMaterial}
   local tag=${2:-GGEO}

   cat << EOT

#include <cassert>
#include "${cls}.hh"

#include "PLOG.hh"
#include "${tag}_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    ${tag}_LOG_ ;




    return 0 ;
}

EOT

}

oks-xcollect-notes(){ cat << EON

*oks-xcollect*
     copies the .bash of externals into externals folder 
     and does inplace edits to correct paths for new home.
     Also writes an externals.bash containing the precursor bash 
     functions.

EON
}
oks-xcollect()
{
   local ehome=$(opticks-home)
   local xhome=$ehome
   local iwd=$PWD 

   cd $ehome

   local xbash=$xhome/externals/externals.bash
   [ ! -d "$xhome/externals" ] && mkdir "$xhome/externals"

   echo "# $FUNCNAME " > $xbash
   
   local x
   local esrc
   local src
   local dst
   local nam
   local note

   oks-xnames | while read x 
   do
      $x-;
      esrc=$($x-source)
      src=${esrc/$ehome\/}
      nam=$(basename $src)
      dst=externals/$nam
       
      if [ -f "$dst" ]; then 
          note="previously copied to dst $dst"  
      else
          note="copying to dst $dst"  
          hg cp $src $xhome/$dst
          perl -pi -e "s,$src,$dst," $xhome/$dst 
          perl -pi -e "s,env-home,opticks-home," $xhome/$dst 
      fi
      printf "# %-15s %15s %35s %s \n" $x $nam $src "$note"
      printf "%-20s %-50s %s\n" "$x-(){" ". \$(opticks-home)/externals/$nam" "&& $x-env \$* ; }"   >> $xbash

   done 
   cd $iwd
}
oks-filemap()
{
   oks-filemap-head
   oks-filemap-body
}

oks-filemap-head(){ cat << EOH
# $FUNCNAME
# configure the spawning of opticks repo from env repo 
# see adm-opticks
#
include opticks.bash
include CMakeLists.txt
include cmake
include externals
#
EOH
}

oks-filemap-body(){
   local dir
   oks-dirs | while read dir ; do
      printf "include %s\n" $dir
   done
}



