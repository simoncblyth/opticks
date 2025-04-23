##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##

opticks-(){         source $(opticks-source) && opticks-env $* ; }

GEOM_TEMPLATE(){ cat << EOT
#!/bin/bash
notes(){ cat << EON
~/.opticks/GEOM/GEOM.sh
=========================

* GEOM.sh IS NOT UNDER SOURCE CONTROL BECAUSE IT IDENTIFIES A SPECIFIC GEOM
* associated utility functions such as scp/grab/vi
  are kept under source control in the opticks.bash GEOM bash function

EON
}

#geom=V0J008
#geom=V1J008
geom=V1J009
export GEOM=\$geom
#
EOT
}

CTX_TEMPLATE(){ cat << EOT
#!/bin/bash
notes(){ cat << EON
~/.opticks/CTX/CTX.sh
=========================

* CTX.sh IS NOT UNDER SOURCE CONTROL BECAUSE IT IS OFTEN USER+GEOMETRY SPECIFIC

EON
}

ctx=Debug_Philox
export CTX=\$ctx
#
EOT
}


TEST_TEMPLATE(){ cat << EOT
#!/bin/bash
notes(){ cat << EON
~/.opticks/TEST/TEST.sh
=========================

* TEST.sh IS NOT UNDER SOURCE CONTROL BECAUSE IT IS OFTEN USER+GEOMETRY SPECIFIC

EON
}

#test=GUN0
test=GUN1
#test=GUN2
export TEST=\$test
#
EOT
}

MOI_TEMPLATE(){ cat << EOT
#!/bin/bash -l
notes(){ cat << EON
~/.opticks/GEOM/MOI.sh : exports MOI midx:mord:gord configuring a frame in a geometry
======================================================================================

* MOI.sh IS NOT UNDER SOURCE CONTROL BECAUSE IT IS GEOMETRY SPECIFIC

The MOI envvar is used to pick a frame within a geometry.
MOI enables targeted 3D ray trace rendering and 2D simtrace slicing
to be oriented with respect to the chosen frame.

The closely related OPTICKS_INPUT_PHOTON_FRAME envvar uses the same
implementation to orient input photons within the chosen frame.

The colon delimited fields of the MOI variable are:

midx
   mesh index (aka lvid) that can be specified by solid name
mord
   mesh ordinal (usually 0) to pick between multiple occurrences
gord
   GAS ordinal to pick between instances of instanced geometry,
   and with global volumes to configure how to conjure the frame
   transforms from the global center-extent CE of the solid
   (as global volumes do not have individual transforms, unlike
    instanced volumes)

+-----------+------------------------------------------------------------------------+
| gord      |   notes, see CSGTarget::getFrameComponents for impl                    |
+===========+========================================================================+
| 0,1,2,..  | for instanced volumes this picks the instance and uses its frame       |
+-----------+------------------------------------------------------------------------+
|  -1       |                                                                        |
+-----------+------------------------------------------------------------------------+
|  -2       |  XYZ frame constructed from CE using SCenterExtentFrame.h              |
+-----------+------------------------------------------------------------------------+
|  -3       |  RTP tangential frame constructed from CE using SCenterExtentFrame.h   |
|           |  (this is useful for orienting wrt volumes placed on a sphere)         |
+-----------+------------------------------------------------------------------------+

* see CSGTarget::getFrameComponents to follow the gord:-2:XYZ gord:-3:RTP branching

EON
}

moi=ALL
#moi=sWorld:0:0
#moi=NNVT:0:0
#moi=NNVT:0:50
#moi=NNVT:0:1000
#moi=PMT_20inch_veto:0:1000  # gord:1000 [0...] GAS-ordinal selection of the instance transform to use
#moi=sChimneyAcrylic:0:-2   # gord:-2/-3 are appropriate for global (non-instanced) geometry

#export MOI=\${MOI:-\$moi}
export MOI=\$moi

EOT
}






GEOM_help(){ cat << EOH
GEOM_help : bash function help
================================

NB the GEOM.sh script is distinct from the bash function.
The GEOM.sh script is not keep in a repository as the
GEOM envvar is private to each user.


GEOM vi
    edit GEOM.sh script that sets GEOM envvar

GEOM ex
    source the GEOM.sh script which exports GEOM envvar

GEOM grab
    rsync remote GEOM directory to local

GEOM get
    synonym for grab

GEOM tmpget
    when TMP is undefined : rsync remote /tmp/$USER/opticks/GEOM/$GEOM dir to local
    when TMP is defined : rsync remote $TMP/GEOM/$GEOM dir to local

GEOM tmp
    change directory to /tmp/$USER/opticks/GEOM/$GEOM
    or $TMP/GEOM/$GEOM when TMP is defined

GEOM scp
    scp GEOM.sh script to remote

GEOM base
     change directory to GEOM base directory above all the \$GEOM dir

GEOM top
     change directory to : \$GEOM dir

GEOM lock
     make \$GEOM dir tree readonly

GEOM unlock
     make \$GEOM dir tree writable

GEOM cf
     change directory to : \$GEOM/CSGFoundry

GEOM ss
     change directory to : \$GEOM/CSGFoundry/SSim

GEOM st
     change directory to : \$GEOM/CSGFoundry/SSim/stree

GEOM std
     change directory to : \$GEOM/CSGFoundry/SSim/stree/standard


GEOM origin
     vim -R origin.gdml on top dir

GEOM source
     source the $HOME/.opticks/GEOM/GEOM.sh
     its kinda dirty to do this but handy for some debugging,
     remember to exit sessions after doing this

GEOM info
     dump variables of the GEOM bash function

GEOM help
     show this usage message

EOH
}

TMP(){
   cd ${TMP:-/tmp/$USER/opticks}
   pwd
}


MOI(){
  : opticks/opticks.bash

  local script=.opticks/GEOM/MOI.sh
  if [ ! -f "$HOME/$script" ]; then
     echo $BASH_SOURCE $FUNCNAME : GENERATE $HOME/$script
     MOI_TEMPLATE > $HOME/$script
  fi
  local defarg="vi"
  local arg=${1:-$defarg}
  if [ "$arg" == "vi" ]; then
     cmd="vi $HOME/$script"
  fi
  echo $cmd
  eval $cmd
}




CTX(){
  : opticks/opticks.bash

  local script=$HOME/.opticks/CTX/CTX.sh

  if [ ! -f "$script" ]; then
     echo $BASH_SOURCE $FUNCNAME : GENERATE $script
     mkdir -p $(dirname $script)
     CTX_TEMPLATE > $script
  fi

  source $script

  local defarg="vi"
  local arg=${1:-$defarg}
  local vars="FUNCNAME defarg arg args script CTX"

  if [ "$arg" == "vi" ]; then
     cmd="vi $script"
  elif [ "$arg" == "info" ]; then
     for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
     cmd="echo -n"
  else
     cmd="echo expecting arg to be one of $args not $arg"
  fi
  echo $cmd
  eval $cmd
}



TEST(){
  : opticks/opticks.bash

  local script=$HOME/.opticks/TEST/TEST.sh

  if [ ! -f "$script" ]; then
     echo $BASH_SOURCE $FUNCNAME : GENERATE $script
     mkdir -p $(dirname $script)
     TEST_TEMPLATE > $script
  fi

  source $script

  local defarg="vi"
  local arg=${1:-$defarg}
  local vars="FUNCNAME defarg arg args script TEST"

  if [ "$arg" == "vi" ]; then
     cmd="vi $script"
  elif [ "$arg" == "info" ]; then
     for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
     cmd="echo -n"
  else
     cmd="echo expecting arg to be one of $args not $arg"
  fi
  echo $cmd
  eval $cmd

}


GEOM(){
  : opticks/opticks.bash GEOM vi/grab/scp

  local script=.opticks/GEOM/GEOM.sh

  if [ ! -f "$HOME/$script" ]; then
     echo $BASH_SOURCE $FUNCNAME : GENERATE $HOME/$script
     GEOM_TEMPLATE > $HOME/$script
  fi

  source $HOME/$script
  #base=.opticks/GEOM/$GEOM


  local args="vi/grab/get/tmpget/tmp/scp/top/lock/unlock/cf/ss/st/std/info"
  local geombase=$HOME/.opticks/GEOM
  local GEOMBASE=${GEOMBASE:-$geombase}
  local GEOMDIR=$GEOMBASE/$GEOM
  local CFDIR=$GEOMDIR/CSGFoundry
  local ggeo=/tmp/$USER/opticks/GGeo
  local tmpbase=${TMP:-/tmp/$USER/opticks}
  local tmp=$tmpbase/GEOM/$GEOM
  local defarg="vi"
  local arg=${1:-$defarg}
  local vars="FUNCNAME defarg arg args script geombase GEOMBASE GEOM GEOMDIR CFDIR tmpbase tmp"

  if [ "$arg" == "vi" ]; then
     cmd="vi $HOME/$script"
  elif [ "$arg" == "ex" ]; then
     cmd="source $HOME/$script"
  elif [ "$arg" == "grab" -o "$arg" == "get" ]; then
     cmd="source $OPTICKS_HOME/bin/rsync.sh $GEOMDIR"
  elif [ "$arg" == "tmpget" ]; then
     cmd="source $OPTICKS_HOME/bin/rsync.sh $tmp"
  elif [ "$arg" == "tmp" ]; then
     cmd="cd $tmp"
  elif [ "$arg" == "scp" ]; then
     cmd="scp $HOME/$script P:$script"
  elif [ "$arg" == "base" ]; then
     cmd="cd $GEOMBASE"
  elif [ "$arg" == "top" ]; then
     cmd="cd $GEOMDIR"
  elif [ "$arg" == "lock" ]; then
     cmd="chmod -R ugo-w $GEOMDIR"
  elif [ "$arg" == "unlock" ]; then
     cmd="chmod -R u+w $GEOMDIR"
  elif [ "$arg" == "cf" ]; then
     cmd="cd $GEOMDIR/CSGFoundry"
  elif [ "$arg" == "ss" ]; then
     cmd="cd $GEOMDIR/CSGFoundry/SSim"
  elif [ "$arg" == "st" ]; then
     cmd="cd $GEOMDIR/CSGFoundry/SSim/stree"
  elif [ "$arg" == "std" ]; then
     cmd="cd $GEOMDIR/CSGFoundry/SSim/stree/standard"
  elif [ "$arg" == "origin" ]; then
     cmd="cd $GEOMDIR ; vim -R origin.gdml"
  elif [ "$arg" == "source" ]; then
     cmd="source $HOME/.opticks/GEOM/GEOM.sh"
  elif [ "$arg" == "help" ]; then
     cmd="GEOM_help"
  elif [ "$arg" == "info" ]; then
     for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
     cmd="echo -n"
  else
     cmd="echo expecting arg to be one of $args not $arg"
  fi
  echo $cmd
  eval $cmd

}

o(){ opticks- ; cd $(opticks-home) ; git status  ; : opticks.bash ;  }
oo(){ opticks- ; cd $(opticks-home) ; om- ; om-- ; : opticks.bash ;  }
os(){ opticks- ; cd $(opticks-home) ; ls -alst *.sh ; : opticks.bash ;  }

on(){ cd $(opticks-home)/notes  ; ls -lt | head -10 ; echo https://bitbucket.org/simoncblyth/opticks/src/master/notes/$(ls -t | head -1) ;  }
onn(){ cd $(opticks-home)/notes ; ls -lt | head -100 ;  }
on1(){   : opticks.bash ; opticks-;opticks-notes-cd ; ls -lt *.rst | head -30 ; vi $(ls -1t *.rst | head -1) ;  }
on2(){   : opticks.bash ; opticks-;opticks-notes-cd ; ls -lt *.rst | head -30 ; vi $(ls -1t *.rst | head -2 | tail -1) ;  }
on100(){ : opticks.bash ; opticks-;opticks-notes-cd ; ls -lt *.rst | head -100  ;  }

oi(){  cd $(opticks-home)/notes/issues ; ls -lt | head -10 ; echo https://bitbucket.org/simoncblyth/opticks/src/master/notes/issues/$(ls -t | head -1) ;  }
oii(){ cd $(opticks-home)/notes/issues ; ls -lt | head -100 ;  }
oi1(){   : opticks.bash ; opticks-;opticks-issues-cd ; ls -lt *.rst | head -30 ; vi $(ls -1t *.rst | head -1) ;  }
oi2(){   : opticks.bash ; opticks-;opticks-issues-cd ; ls -lt *.rst | head -30 ; vi $(ls -1t *.rst | head -2 | tail -1) ;  }
oi100(){ : opticks.bash ; opticks-;opticks-issues-cd ; ls -lt *.rst | head -100  ;  }

cu(){  local cu ; date ; for cu in *.cu ; do echo touch $cu && touch $cu ; done ; ls -l *.cu ;  }

bst(){ cd $(opticks-home)/examples/Geant4/BoundaryStandalone ;  }
sgs(){ cd $(opticks-home)/examples/Geant4/ScintGenStandalone ;  }
yst(){ cd $(opticks-home)/examples/Geant4/RayleighStandalone ;  }


oot(){ oo ; opticks-t ; : opticks.bash ; }
t(){ typeset -f $*    ; : opticks.bash ; }
rc(){ local RC=$?; echo RC $RC; return $RC ; : opticks.bash ;  }
eo(){ env | grep =INFO ; }

geom(){
   : opticks/opticks.bash
   : NB this is distinct from geom_ bash function
   local path=$HOME/.opticks/GEOM.txt
   local script=$HOME/opticks/bin/GEOM.sh
   if [ "$1" == "scp" ]; then
       scp $path P:.opticks/
   else
       vi $path $script ;
   fi
}


com_(){
   : opticks/opticks.bash

   local path0=$(opticks-home)/bin/COMMON.sh
   local path1=$(opticks-home)/bin/GEOM_.sh
   local path2=$(opticks-home)/bin/OPTICKS_INPUT_PHOTON.sh
   local path3=$(opticks-home)/bin/OPTICKS_INPUT_PHOTON_.sh
   vi $path0 $path1 $path2 $path3
}

geom_(){
   : opticks/opticks.bash
   local path=$(opticks-home)/bin/GEOM_.sh
   if [ "$1" == "scp" ]; then
       scp $path P:opticks/bin/
   elif [ "$1" == "source" ]; then
       source $path
   else
       vi $path ;
   fi
}

geomlist_(){
   : opticks/opticks.bash
   local path=$(opticks-home)/bin/geomlist.sh
   if [ "$1" == "scp" ]; then
       scp $path P:opticks/bin/
   elif [ "$1" == "source" ]; then
       source $path
   else
       vi $path ;
   fi
}

oip(){
   : opticks/opticks.bash
   local path=$(opticks-home)/bin/OPTICKS_INPUT_PHOTON.sh
   local path_=$(opticks-home)/bin/OPTICKS_INPUT_PHOTON_.sh
   if [ "$1" == "scp" ]; then
       scp ${path}  P:opticks/bin/
       scp ${path_} P:opticks/bin/
   elif [ "$1" == "source" ]; then
       source $path
   else
       vi ${path_} ${path} ;
   fi
}




fold(){
   : opticks/opticks.bash
   local path=$(opticks-home)/bin/AB_FOLD.sh
   vi $path
}


opticks-hash(){  git -C $(dirname $BASH_SOURCE) rev-parse --short HEAD ; }
opticks-source(){   echo $BASH_SOURCE ; }
opticks-ldir(){     echo $(dirname $BASH_SOURCE) ; }
opticks-vi(){       vi $(opticks-source) ; }
opticks-help(){ opticks-usage ; }
opticks-usage(){   cat << \EOU

OPTICKS BASH FUNCTIONS
========================

*opticks-docs-remote*
     open browser on the remote html documentation
     https://simoncblyth.bitbucket.io/opticks/index.html

*opticks-docs*
     open browser on the local html documentation

*opticks-docs-make*
     sphinx-build the docs

*opticks-notes*
     open browser on the local html development notes

*opticks-notes-make*
     sphinx-build the notes


Most Opticks development notes are kept in the env repository,
use start from

   env-;opticksdev-;opticksdev-vi


EOU
}

opticks-env(){
    : opticks.bash
   # dont pollute : otherwise will get infinite loops : as opticks is used in many other -env
   . $(opticks-ldir)/externals/externals.bash       ## just precursors
   . $(opticks-ldir)/integration/integration.bash   ## just precursors
}

opticks-env-info(){

   cat << EOI

       uname   : $(uname -a)
       HOME    : $HOME
       VERBOSE : $VERBOSE
       USER    : $USER

EOI
   env | grep OPTICKS

}


opticks-metadata-export()
{
    onvidia-
    onvidia-export
}





olocal-()
{
   echo -n # transitional standin for olocal-
}

opticks-home-default(){ echo $(dirname $(opticks-source)) ; }
opticks-home(){   echo ${OPTICKS_HOME:-$(opticks-home-default)} ; }  ## input from profile
opticks-name(){   basename $(opticks-home) ; }
opticks-fold(){   echo $(dirname $(opticks-home)) ; }



opticks-tboolean-shortcuts(){


   : **simulate** : aligned bi-simulation creating OK+G4 events
   ts(){  LV=$1 tboolean.sh ${@:2} ; }

   : **visualize** : load events and visualize the propagation
   tv(){  LV=$1 tboolean.sh --load ${@:2} ; }

   : **visualize** the geant4 propagation
   tv4(){  LV=$1 tboolean.sh --load --vizg4 ${@:2} ; }

   : **analyse interactively** : load events and analyse the propagation in ipython
   ta(){  LV=$1 tboolean.sh --ip ${@:2} ; }

   : **analyse** : load events and run python analysis script the propagation
   tp(){  LV=$1 tboolean.sh --py ${@:2} ; }

}


opticks-user-home(){ echo ${OPTICKS_USER_HOME:-$HOME} ; }
opticks-sharedcache-prefix-default(){ echo $(opticks-user-home)/.opticks ; }
opticks-usercache-prefix-default(){   echo $(opticks-user-home)/.opticks ; }


opticks-geocache-prefix(){    echo ${OPTICKS_GEOCACHE_PREFIX:-$(opticks-sharedcache-prefix-default)} ; }
opticks-rngcache-prefix(){    echo ${OPTICKS_RNGCACHE_PREFIX:-$(opticks-sharedcache-prefix-default)} ; }
opticks-usercache-prefix(){   echo ${OPTICKS_USERCACHE_PREFIX:-$(opticks-usercache-prefix-default)} ; }



opticks-geocachedir(){ echo $(opticks-geocache-prefix)/geocache ; }
opticks-rngcachedir(){ echo $(opticks-rngcache-prefix)/rngcache ; }
opticks-rngdir(){      echo $(opticks-rngcachedir)/RNG ; }
opticks-rngdir-cd(){   cd $(opticks-rngdir) ; }


opticks-cache-info(){ cat << EON
$FUNCNAME
=====================

    opticks-prefix                      : $(opticks-prefix)
    opticks-installcachedir             : $(opticks-installcachedir)

    OPTICKS_GEOCACHE_PREFIX             : $OPTICKS_GEOCACHE_PREFIX
    OPTICKS_RNGCACHE_PREFIX             : $OPTICKS_RNGCACHE_PREFIX
    OPTICKS_USERCACHE_PREFIX            : $OPTICKS_USERCACHE_PREFIX

    opticks-sharedcache-prefix-default  : $(opticks-sharedcache-prefix-default)
    opticks-geocache-prefix             : $(opticks-geocache-prefix)
    opticks-rngcache-prefix             : $(opticks-rngcache-prefix)

    opticks-geocachedir                 : $(opticks-geocachedir)
    opticks-rngcachedir                 : $(opticks-rngcachedir)

    opticks-rngdir                      : $(opticks-rngdir)

    opticks-usercache-prefix-default    : $(opticks-usercache-prefix-default)
    opticks-usercache-prefix            : $(opticks-usercache-prefix)


sharedcache-prefix
   geocache, rngcache

usercache-prefix
   runcache


Changes to the cache prefix layouts or envvars names
need to be done in triplicate in bash/py/C++ in::

    opticks.bash
    ana/geocache.bash
    ana/key.py
    boostrap/BOpticksResource.cc

EON
}




opticks-id(){ cat << EOI

  opticks-home   : $(opticks-home)
  opticks-prefix : $(opticks-prefix)
  opticks-name   : $(opticks-name)

EOI
}

opticks-cmakecache(){ echo $(opticks-bdir)/CMakeCache.txt ; }

opticks-pretty(){  cat ${1:-some.json} | python -m json.tool ; }


opticks-key(){     echo ${OPTICKS_KEY} ; }  # below two functions depend on the OPTICKS_KEY input envvar
opticks-keydir(){  geocache- ; geocache-keydir ; }  # referred to from docs/opticks_testing.rst
opticks-kcd(){     geocache- ; geocache-kcd ; }





opticks-tbool-info(){ cat << EOI

$FUNCNAME
======================


   opticks-tbool-path 0 : $(opticks-tbool-path 0)
   opticks-nnt-path 0   : $(opticks-nnt-path 0)


EOI
}



opticks-tbool-path(){
   local lvid=${1:-0}
   local extras=$(opticks-srcextras)
   local path=$extras/${lvid}/tbool${lvid}.bash
   echo $path
}
opticks-nnt-path(){
   local lvid=${1:-0}
   local extras=$(opticks-srcextras)
   local path=$extras/${lvid}/NNodeTest_${lvid}.cc
   echo $path
}
opticks-nnt-paths(){
    local arg
    for arg in $*
    do
        echo $(opticks-nnt-path $arg)
    done
}

opticks-nnt-vi(){
   [[ $# -eq 0 ]] && echo expecting one or more lvidx integer arguments && return
   local paths=$(opticks-nnt-paths $*)
   vi $paths
}

opticks-nnt-(){

   local path=${1:-/tmp/some/path/to/NNodeTest_LVID.cc}
   local name=$(basename $path)
   local stem=${name/.cc}
   local bindir=$(opticks-bindir)
   local bin=$bindir/$stem

   npy-
   sysrap-
   glm-
   plog-

   cat << EOL

   clang -std=c++11  $path \
            -I$(npy-sdir) \
            -I$(sysrap-sdir) \
            -I$(glm-dir) \
            -I$(plog-dir)/include \
             -lNPY \
             -lSysRap \
             -lc++ \
             -L$(opticks-bindir) \
             -Wl,-rpath,$(opticks-bindir) \
             -o $bin

EOL
}


opticks-nnt()
{
   local msg="$FUNCNAME :"
   local lvid=${1:-0}
   local path=$(opticks-nnt-path $lvid)
   local name=$(basename $path)
   local stem=${name/.cc}

   [ ! -f $path ] && echo $msg no such path && return

   echo $msg compiling $path

   eval $($FUNCNAME- $path)  && $stem

   which $stem
}




opticks-tbool-vi(){
   local path=$(opticks-tbool-path ${1:-0})
   vi $path
}

opticks-tbool(){
   local msg="$FUNCNAME :"
   local lvid=${1:-0}
   local path=$(opticks-tbool-path $lvid)
   echo $msg sourcing $path
   [ ! -f $path ] && echo $msg no such path && return
   . $path
   tbool${lvid}
}

opticks-tbool-(){
   local msg="$FUNCNAME :"
   local lvid=${1:-0}
   local path=$(opticks-tbool-path $lvid)
   echo $msg sourcing $path
   [ ! -f $path ] && echo $msg no such path && return
   . $path
   tbool${lvid}-
}




# reliance on envvars is appropriate for debugging only, not "production"
opticks-debugging-idpath(){ echo $IDPATH ; }
opticks-debugging-idfold(){ echo $(dirname $(opticks-debugging-idpath)) ; }

#opticks-tscan-dir(){ echo  $TMP/tgltf/extras/${1:-0} ; }
opticks-tscan-dir(){ echo  $(opticks-debugging-idfold)/extras/${1:-0} ; }

opticks-tscan-all(){ opticks-tscan / ; }
opticks-tscan(){
   local msg="$FUNCNAME :"
   local lvid=${1:-0}
   local dir=$(opticks-tscan-dir $lvid)
   [ ! -d $dir ] && echo $msg no such dir $dir && return
   echo $msg scanning $dir
   NScanTest $dir
}




opticks-suffix(){
   case $(uname) in
      MING*) echo .exe ;;
          *) echo -n  ;;
   esac
}

opticks-day(){ date +"%Y%m%d" ; }


opticks-sdir(){   echo $(opticks-home) ; }
opticks-scd(){  cd $(opticks-sdir)/$1 ; }
opticks-ncd(){  opticks-scd notes/issues ;  }

opticks-adir(){   echo $(dirname $(opticks-home))/opticks_ancient ; }
opticks-acd(){  cd $(opticks-adir)/$1 ; }

opticks-cmake-generator(){ echo ${OPTICKS_CMAKE_GENERATOR:-Unix Makefiles} ; }
opticks-buildtype(){       echo ${OPTICKS_BUILDTYPE:-Debug}  ; }

opticks-prefix(){ echo ${OPTICKS_PREFIX:-/usr/local/opticks}  ; }
opticks-dir(){    echo $(opticks-prefix) ; }
opticks-idir(){   echo $(opticks-prefix) ; }
opticks-bdir(){   echo $(opticks-prefix)/build ; }
opticks-bindir(){ echo $(opticks-prefix)/lib ; }   ## use lib for executables for simplicity on windows
opticks-xdir(){   echo $(opticks-prefix)/externals ; }  ## try putting externals above the build identity
opticks-installcachedir(){ echo $(opticks-prefix)/installcache ; }
opticks-setup-path(){   echo $(opticks-prefix)/bin/opticks-setup.sh ; }
opticks-bashrc-path(){  echo $(opticks-prefix)/bashrc ; }
opticks-utils-path(){   echo $(opticks-prefix)/bin/opticks-utils.sh ; }

opticks-setup(){
   local msg="=== $FUNCNAME :"
   local setup=$(opticks-setup-path)
   [ ! -f $setup ] && echo "$msg MISSING setup script $setup : incomplete opticks installation  " && return 1
   source $setup
   return 0
}

opticks-setup-find-geant4-prefix(){ opticks-setup-find-config-prefix Geant4 ; }
opticks-setup-find-boost-prefix(){  opticks-setup-find-config-prefix Boost ; }
opticks-setup-find-config-prefix(){
   : mimick CMake "find_package name CONFIG" identifing the first prefix in the path
   local name=${1:-Geant4}
   local prefix=""
   local rc=0
   local ifs=$IFS
   IFS=:
   for pfx in $CMAKE_PREFIX_PATH ; do

      : protect cmds that can give non-zero rc from "set -e" via pipeline but catch the rc
      rc=1
      ls -1 $pfx/lib*/$name-*/${name}Config.cmake 2>/dev/null 1>&2 && rc=$?
      [ $rc -eq 0 ] && prefix=$pfx && break

      ls -1 $pfx/lib*/cmake/$name-*/${name}Config.cmake 2>/dev/null 1>&2 && rc=$?
      [ $rc -eq 0 ] && prefix=$pfx && break

      ls -1 $pfx/lib*/cmake/$name/${name}Config.cmake 2>/dev/null 1>&2 && rc=$?
      [ $rc -eq 0 ] && prefix=$pfx && break

      # NB not general, doesnt find the lowercased form : but works for Geant4 and Boost
   done
   IFS=$ifs
   echo $prefix
}

opticks-setup-find-geant4-prefix-notes(){ cat << EON
Hans reports that the path to Geant4Config.cmake changed with v11.1.1::

  /data3/wenzel/geant4-v11.0.3-installst/lib/Geant4-11.0.3/Geant4Config.cmake
  /data3/wenzel/geant4-v11.1.1-install/lib/cmake/Geant4/Geant4Config.cmake

So add another pattern in opticks-setup-find-config-prefix to try to find that.

EON
}



opticks-c(){    cd $(opticks-dir)/$1 ; }
opticks-cd(){   cd $(opticks-dir)/$1 ; }
opticks-icd(){  cd $(opticks-idir); }
opticks-bcd(){  cd $(opticks-bdir); }
opticks-xcd(){  cd $(opticks-xdir); }



opticks-optix-prefix(){
   : opticks-optix-prefix is used by om-cmake-okconf to locate OptiX
   echo ${OPTICKS_OPTIX_PREFIX:-$(opticks-prefix)/externals/OptiX}
}

opticks-cuda-prefix(){ echo ${OPTICKS_CUDA_PREFIX:-/usr/local/cuda} ; }


opticks-compute-notes(){ cat << EON
opticks-compute-notes
-----------------------

Both the compute envvars are read by correspondingly named 
bash functions which are used by om-cmake-okconf which 
result in generation of the okconf CMake target which includes
the okconf-config.cmake::

    $OPTICKS_PREFIX/lib64/cmake/okconf/okconf-config.cmake

This config sets CMake COMPUTE variables available from all the active packages, eg::

    set(COMPUTE_CAPABILITY 70)
    set(COMPUTE_ARCHITECTURES 70,89)

In this way the user specified envvar values are fed into all packages. 


OPTICKS_COMPUTE_CAPABILITY

    integer string eg "70" or "89" (representing "compute_70" or "compute_89")
    identifying the virtual CUDA machine that nvcc will target when generating ptx 
    for OptiX consumption via OptixModuleCreate in PIP.cc 

    An appropriate value is the lowest capability of GPU you want to support.

OPTICKS_COMPUTE_ARCHITECTURES
    NB if this envvar is not defined it defaults to the value of 
    OPTICKS_COMPUTE_CAPABILITY

    comma delimited list (or single integer) of compute_XY or sm_XY 
    integer specifications which is modified to a semicolon delimited list and then 
    passed to the CMake CUDA_ARCHITECTURES target property
    
    An appropriate value is the GPUs you want to support. 

    For usage see CUDA targets::

        #opticks-fl CUDA_ARCHITECTURES
        sysrap/CMakeLists.txt
        qudarap/CMakeLists.txt
        CSG/CMakeLists.txt
        CSGOptiX/CMakeLists.txt  

    * https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html

    NB the CUDA_ARCHITECTURES are used by general CUDA compilation
    whereas generation of PTX for OptiX consumption uses the single integer COMPUTE_CAPABILITY



EON
}

opticks-compute-capability(){ echo ${OPTICKS_COMPUTE_CAPABILITY:-0} ; }
opticks-compute-architectures(){ echo ${OPTICKS_COMPUTE_ARCHITECTURES:-$(opticks-compute-capability)} ; }

opticks-externals(){
: emits to stdout the names of the bash precursors that download and install the externals
  cat << EOL  | grep -v ^#
bcm
glm
glfw
glew
gleq
imgui
plog
nljson
EOL
}
# need imgui even though not currently using it just for its font file

opticks-externals-retired(){ cat << EOL
#opticksaux
#assimp
#openmesh
#oimplicitmesher
#odcs
#oyoctogl
#ocsgbsp
EOL
}



opticks-preqs(){
: emits to stdout the names of the bash precursors that configure and check pre-requisite packages
   cat <<  EOP
cuda
optix
EOP
}

opticks-foreign(){
   cat << EOL
clhep
xercesc
g4
EOL
}

opticks-possibles(){ cat << EOL
oof
EOL
}


opticks-externals-notes(){ cat << EON

oimplicitmesher
    requires glm, finds it using opticks/cmake/Modules/FindGLM.cmake
    this means use common prefix with opticks
odcs
    requires glm, finds it using opticks/cmake/Modules/FindGLM.cmake
    this means use common prefix with opticks



EON
}




opticks-cmake-version(){  cmake --version | perl -ne 'm/(\d*\.\d*\.\d*)/ && print "$1" ' - ; }
opticks-externals-info(){  cat << EOI
$FUNCNAME
============================

    opticks-cmake-version  : $(opticks-cmake-version)

EOI
}

opticks-externals-install(){ opticks-installer- $(opticks-externals) ; }
opticks-foreign-install(){ opticks-installer- $(opticks-foreign) ; }
opticks-possibles-install(){ opticks-installer- $(opticks-possibles) ; }

opticks-installer-(){
    echo $FUNCNAME
    local msg="=== $FUNCNAME :"
    local pkgs=$*
    local pkg
    for pkg in $pkgs ; do

        printf "\n\n\n############## %s ###############\n\n\n" $pkg

        $pkg-
        $pkg--
        rc=$?
        [ $rc -ne 0 ] && echo $msg RC $rc from pkg $pkg : ABORTING && return $rc
    done
    return 0
}





opticks-externals-url(){     echo $FUNCNAME ; opticks-externals | opticks-ext-url ; }
opticks-externals-dist(){    echo $FUNCNAME ; opticks-externals | opticks-ext-dist ; }
opticks-externals-dir(){     echo $FUNCNAME ; opticks-externals | opticks-ext-dir ; }
opticks-externals-status(){  echo $FUNCNAME ; opticks-externals | opticks-ext-status ; }


opticks-externals-dist-scp(){
   : emit scp commands to copy distribution tarballs/zips to a remote opticks_download_cache, a GFW workaround
   local ext
   local dist
   local cmd
   for ext in $(opticks-externals) ; do
       $ext-
       dist=$($ext-dist 2>/dev/null)
       if [ -n "$dist" ]; then
           cmd="scp $dist P:/data/opticks_download_cache/"
           echo $cmd
       fi
   done
}

opticks-externals-git-scp(){
   : emit git clone and scp commands to populate the remote opticks_download_cache with bare git repos
   local ext
   local url
   local cmd

   local dir=/tmp/$FUNCNAME
   mkdir -p $dir
   cd $dir

   for ext in $(opticks-externals) ; do
       $ext-
       url=$($ext-url 2>/dev/null)
       if [ -n "$url" ]; then
           if [ "${url:(-4)}" == ".git" ]; then
               local repo=$(basename $url)
               echo git clone --bare $url
               echo scp -r $repo P:/data/opticks_download_cache/
           fi
       fi
   done
}



opticks-preqs-pc(){      opticks-pc- $(opticks-preqs) ; }
opticks-foreign-pc(){  opticks-pc- $(opticks-foreign) ; }

opticks-pc-(){
    echo $FUNCNAME
    local msg="=== $FUNCNAME :"
    local funcs=$*
    local func
    for func in $funcs ; do

        printf "\n\n\n############## %s ###############\n\n\n" $func

        $func-
        $func-pc

        rc=$?
        [ $rc -ne 0 ] && echo $msg RC $rc from func $func : ABORTING && return $rc
    done
    return 0
}



# these -setup are invoked by opticks-setup-generate
opticks-externals-setup(){   echo === $FUNCNAME ; opticks-externals | opticks-ext-setup ; }
opticks-preqs-setup(){       echo === $FUNCNAME ; opticks-preqs     | opticks-ext-setup ; }


opticks-foreign-url(){     echo $FUNCNAME ; opticks-foreign | opticks-ext-url ; }
opticks-foreign-dist(){    echo $FUNCNAME ; opticks-foreign | opticks-ext-dist ; }

opticks-possibles-url(){     echo $FUNCNAME ; opticks-possibles | opticks-ext-url ; }
opticks-possibles-dist(){    echo $FUNCNAME ; opticks-possibles | opticks-ext-dist ; }




opticks-pc-rename-kludge(){

   : used by xercesc-pc to pkg-config rename things
   : TODO : try to avoid this nasty workaround : is the renaming really needed

   local msg="=== $FUNCNAME :"

   # trust the PKG_CONFIG_PATH to yield the XercesC that Geant4 is using
   local name=${1:-xerces-c}
   local name2=${2:-OpticksXercesC}

   local pcfiledir=$(pkg-config --variable=pcfiledir $name)
   local path=$pcfiledir/$name.pc
   local path2=$pcfiledir/$name2.pc
   local path3=$(opticks-prefix)/externals/lib/pkgconfig/$name2.pc

   cat << EOI

$FUNCNAME
---------------------------

   name      : $name
   name2     : $name2
   pcfiledir : $pcfiledir
   path      : $path
   path2     : $path2
   path3     : $path3

EOI

   if [ -w "$(dirname $path2)" ]; then

       echo $msg have write permission to path2 $path2
       ln -svf $path $path2

   elif [ -w "$(dirname $path3)" ]; then

       echo $msg NO write permission for path2 $path2 resort to path3 $path3
       ln -svf $path $path3

   else
       echo $msg NO write permission to path3 $path3 either
   fi

}









opticks-setup-notes(){ cat << EON

Opticks Setup Script
-----------------------

Every full build and install of Opticks generates
the opticks setup script.::

   opticks-setup-path : $(opticks-setup-path)

The setup script is intended to encapsulate all setup
required for the use of Opticks, it is used with::

   source /usr/local/opticks/opticks-setup.sh

NB the setup script is not intended to be edited.

The setup script demarks the line between building+installing
Opticks and using it. Everything that changing would entail
a rebuild is hardcoded into the setup.

The Opticks build is sensitive to certain crucial input envvars.
As changing these input envvars would potentially invalidate
the Opticks installation the setup script makes consistency
checks that the current envvars if present match those of the installation
when the setup was generated.


Distinction between build and user environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Make a clear distinction in your mind between:

build environment
    controlled by developer and to some extent propagated into the setup script.
    It is appropriate to make demands on certain input envvars being defined
    in the opticks-setup bash functions.

user environment
    controlled by the setup script, do not demand input envvars are present instead
    hardcode them.  For integration purposes the crucial envvars CMAKE_PREFIX_PATH,
    PKG_CONFIG_PATH, LD_LIBRARY_PATH, PATH etc..  cannot be hardcoded



Optional input envvars in build environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These optional envvars determine which external libraries are linked into
the Opticks build.

CMAKE_PREFIX_PATH
   determines which external libraries CMake find_package will select

PKG_CONFIG_PATH
   determines which external libraries pkg-config and hence opticks-config
   will select, used by Non-CMake integrations


Due to the crucial effect of these envvars a consistency check of their
values at build time recorded in the setup script as::

   BUILD_CMAKE_PREFIX_PATH
   BUILD_PKG_CONFIG_PATH

Typically CMAKE_PREFIX_PATH and PKG_CONFIG_PATH will not be defined in the
input environment, so the BUILD_ prefixed envvars will be blank.
The envvars will be defined by the setup.

As running setup twice can cause such inconsistencies it is
inconvenient to raise an error for this. Instead a warning is
given and they are changed to the build time versions.


Mandatory input envvars in build environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OPTICKS_PREFIX
   determines where Opticks is installed, eg /usr/local/opticks

OPTICKS_OPTIX_PREFIX
   determines the OptiX installation to build against, eg /usr/local/optix-650

OPTICKS_CUDA_PREFIX
   determines the CUDA installation to build against, eg /usr/local/cuda-10.1

OPTICKS_COMPUTE_CAPABILITY
   influences CUDA compilation flags, optixrap fails to build without this

Note that reinstalling a different version of these externals
into the same directory eg /usr/local/cuda or /usr/local/optix
will break the installation.  To avoid this it is
recommended to use input prefixes that incorporate the version number.


Which envvars can be changed post setup ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A few envvars defined in the setup script are
purely runtime in nature and can be overriden.

TMP
   temp dir

Dev
---

opticks- && opticks-setup-generate && opticks-setup-cat && opticks-setup--


EON
}



opticks-paths()
{
   local vars="CMAKE_PREFIX_PATH PKG_CONFIG_PATH PATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH PYTHONPATH CPATH MANPATH CMTEXTRATAGS"
   local var
   for var in $vars ; do
       echo $var
       echo ${!var} | tr ":" "\n"
       echo
   done
   for var in $vars ; do
       local num=$(echo ${!var} | tr ":" "\n" | wc -l )
       printf "%-30s : %d \n" $var $num
   done
   env | grep OPTICKS | grep PREFIX
}


opticks-setup-cat(){ cat $(opticks-setup-path) ; }
opticks-setup-vi(){  vi $(opticks-setup-path)  ; }
opticks-setup--(){   source $(opticks-setup-path) ; }
#opticks-release-- MAKES NO SENSE AS THE release script relies on sourced path to give prefix

opticks-setup-generate-notes(){ cat << EON

opticks-setup-generate-notes
==============================

opticks-setup-generate is invoked via the below stack of bash functions::

    opticks-full
    opticks-full-make
    opticks-setup-generate

The setup files written by opticks-setup-generate are crucial to the Opticks
build as they set environment vars such as CMAKE_PREFIX_PATH that
configure where CMake will look for externals including NVIDIA OptiX and CUDA.

EON
}
opticks-setup-generate(){

    : opticks-full/opticks-full-make/opticks-setup-generate

    local msg="=== $FUNCNAME :"
    local rc

    : check OPTICKS_PREFIX envvar and dir exists and is distinct from opticks-home
    opticks-check-prefix
    rc=$?
    [ ! $rc -eq 0 ] && return $rc

    : check OPTICKS_COMPUTE_CAPABILITY envvar
    opticks-check-compute-capability
    rc=$?
    [ ! $rc -eq 0 ] && return $rc

    : check Geant4 is on CMAKE_PREFIX_PATH
    opticks-check-geant4
    rc=$?
    [ ! $rc -eq 0 ] && return $rc

    : check build tools on PATH
    opticks-check-tools
    rc=$?
    [ ! $rc -eq 0 ] && return $rc

    : check envvars OPTICKS_PREFIX, OPTICKS_OPTIX_PREFIX, OPTICKS_CUDA_PREFIX, OPTICKS_COMPUTE_CAPABILITY
    opticks-setup-check-mandatory-buildenv
    rc=$?
    [ ! $rc -eq 0 ] && return $rc

    : check PREFIX envvars have corresponding directories
    opticks-setup-check-mandatory-dir
    rc=$?
    [ ! $rc -eq 0 ] && return $rc

    local setup=$(opticks-setup-path)
    mkdir -p $(dirname $setup)
    echo $msg writing $setup

    local bashrc=$(opticks-bashrc-path)
    mkdir -p $(dirname $bashrc)
    echo $msg writing $bashrc

    local utils=$(opticks-utils-path)
    mkdir -p $(dirname $utils)
    echo $msg writing $utils


    local csh=${setup/.sh}.csh
    echo "# $FUNCNAME you gotta be kidding : use bash  "  > $csh

    opticks-setup-generate- > $setup
    rc=$?
    echo $msg post opticks-setup-generate- rc $rc
    [ ! $rc -eq 0 ] && echo $msg ABORT && return $rc

    opticks-bashrc-generate- > $bashrc
    rc=$?
    echo $msg post opticks-bashrc-generate- rc $rc
    [ ! $rc -eq 0 ] && echo $msg ABORT && return $rc

    opticks-utils-generate- > $utils
    rc=$?
    echo $msg post opticks-utils-generate- rc $rc
    [ ! $rc -eq 0 ] && echo $msg ABORT && return $rc


    opticks-externals-setup
    rc=$?
    echo $msg post opticks-externals-setup rc $rc

    opticks-preqs-setup
    rc=$?
    echo $msg post opticks-preqs-setup rc $rc

    return $rc
}

opticks-setup-generate-(){

    opticks-setup-hdr- $FUNCNAME
    rc=$?
    [ ! $rc -eq 0 ] && return $rc

    opticks-setup-prefix-
    #opticks-setup-consistency-check-  CMAKE_PREFIX_PATH
    #opticks-setup-consistency-check-  PKG_CONFIG_PATH
    opticks-setup-misc-
    opticks-setup-funcs-

    opticks-setup-paths-
    opticks-setup-libpaths-
    opticks-setup-geant4-
    rc=$?
    return $rc
}

opticks-bashrc-generate-(){

    opticks-setup-hdr- $FUNCNAME
    rc=$?
    [ ! $rc -eq 0 ] && return $rc

    opticks-bashrc-prefix-

    opticks-setup-identifier-
    opticks-setup-misc-
    opticks-setup-funcs-

    opticks-setup-paths-
    opticks-setup-libpaths-
    opticks-setup-geant4-
    rc=$?
    return $rc
}

opticks-utils-generate-(){ opticks-utils-generate-- |  perl -pe 's,cd_func,cd,g' -  ; }
opticks-utils-generate--(){
   echo "# $FUNCNAME"
   echo "# some funcs are needed before sourcing release : so collect them here "
   declare -f opticks-prepend-prefix
}


opticks-setup-identifier-(){ cat << EOI

# [ $FUNCNAME
export OPTICKS_IDENTIFIER=$($(opticks-home)/git_identifier.sh)
export OPTICKS_IDENTIFIER_NOTE="$($(opticks-home)/git_identifier.sh NOTE)"
export OPTICKS_IDENTIFIER_TIME="$($(opticks-home)/git_identifier.sh TIME)"
okid()
{
    : $FUNCNAME
    local vv="OPTICKS_PREFIX OPTICKS_IDENTIFIER OPTICKS_IDENTIFIER_NOTE OPTICKS_IDENTIFIER_TIME"
    for v in \$vv ; do printf "%30s : %s \n"  "\$v" "\${!v}" ; done
}
# ] $FUNCNAME

EOI

}


opticks-setup-check-mandatory-buildenv()
{
    local msg="=== $FUNCNAME MISSING mandatory envvar in buildenv :"

    if [ -z "$OPTICKS_PREFIX" ]; then
        echo $msg OPTICKS_PREFIX
        return 1
    fi
    if [ -z "$OPTICKS_OPTIX_PREFIX" ]; then
        echo $msg OPTICKS_OPTIX_PREFIX
        return 1
    fi
    if [ -z "$OPTICKS_CUDA_PREFIX" ]; then
        echo $msg OPTICKS_CUDA_PREFIX
        return 1
    fi
    if [ -z "$OPTICKS_COMPUTE_CAPABILITY" ]; then
        echo $msg OPTICKS_COMPUTE_CAPABILITY
        return 1
    fi

    return 0
}
opticks-setup-check-mandatory-dir()
{
    local msg="=== $FUNCNAME MISSING mandatory directory  :"

    if [ ! -d "$OPTICKS_PREFIX" ]; then
        echo $msg OPTICKS_PREFIX
        return 1
    fi
    if [ ! -d "$OPTICKS_OPTIX_PREFIX" ]; then
        echo $msg OPTICKS_OPTIX_PREFIX
        return 1
    fi
    if [ ! -d "$OPTICKS_CUDA_PREFIX" ]; then
        echo $msg OPTICKS_CUDA_PREFIX
        return 1
    fi

    if [ ! -f "$OPTICKS_CUDA_PREFIX/bin/nvcc" ]; then
        echo $FUNCNAME MISSING nvcc
        return 2
    fi
    return 0
}

opticks-setup-hdr-(){ cat << EOH
#!/bin/bash
#
#    D O   N O T   E D I T
#
# generated by $1
#
# $FUNCNAME $(date)

NAME=\$(basename \$BASH_SOURCE)
MSG="=== \$NAME :"

if [ "\$BASH_SOURCE" == "\$0" ]; then
   echo \$MSG ERROR the \$BASH_SOURCE file needs to be sourced not executed
   exit 1
   # normally would return from sourced script but here have detected are being executed so exit
fi

EOH
}

opticks-setup-consistency-check-(){
   local var=${1:-CMAKE_PREFIX_PATH}

   : note how the buildtime values of the vars are captured into the
   : generated script then checked to be consistent later
   : although note that a warning is provided but no action

   cat << EOC

# $FUNCNAME
# record buildenv optional envvars
BUILD_$var=${!var}

if [ -n "\$$var" ]; then
   if [ "\$$var" != "\$BUILD_$var" ]; then
       echo \$MSG WARNING inconsistent $var between build time and setup time
       printf "%s %-25s \n"  "\$MSG" $var
       echo \$$var | tr ":" "\n"
       printf "%s %-25s \n"  "\$MSG" BUILD_$var
       echo \$BUILD_$var | tr ":" "\n"
       echo

       #echo \$MSG WARNING resetting $var to the build time input value
       #export $var=\$BUILD_$var
   else
       echo \$MSG consistent $var between build time and usage
       printf "%s %25s \n"  "\$MSG" $var
       echo \$$var | tr ":" "\n"
    fi
fi
EOC
}


opticks-gob()
{
    : cmake configures, builds, installs and runs example code  : see examples/*/gob.sh

    # many of the CMake examples do not follow standard naming so the problem is knowing
    # if an executable is created and what is its name

    local msg="=== $FUNCNAME :"
    local sdir=$(pwd)
    local name=$(basename $sdir)
    local bdir=/tmp/$USER/opticks/$name/build

    [ -z "$OPTICKS_PREFIX" ] && echo $msg OPTICKS_PREFIX is required envvar && return 1
    [ ! -d "$OPTICKS_PREFIX" ] && echo $msg OPTICKS_PREFIX $OPTICKS_PREFIX directory does not exist && return 2
    [ ! -f "$name.cc" ] && echo $msg MISSING $name.cc in PWD $PWD && return 3
    [ ! -f "CMakeLists.txt" ] && echo $msg MISSING CMakeLists.txt in PWD $PWD && return 4

    local iwd=$PWD
    rm -rf $bdir
    mkdir -p $bdir && cd $bdir && pwd

    cmake $sdir \
         -G "Unix Makefiles" \
         -DCMAKE_BUILD_TYPE=Debug \
         -DOPTICKS_PREFIX=$OPTICKS_PREFIX \
         -DCMAKE_INSTALL_PREFIX=$OPTICKS_PREFIX \
         -DCMAKE_MODULE_PATH=$OPTICKS_PREFIX/cmake/Modules

    echo $msg make
    make
    [ "$(uname)" == "Darwin" ] && echo "Kludge sleep 2s" && sleep 2

    echo $msg make install
    make install

    bin=$(which $name)
    ls -l $bin

    echo $msg $bin
    $bin

    cd $iwd
}


opticks-goc()
{
    # this was developed in examples/UseSysRapNoCMake/go.sh

    : opticks-config configures, builds, installs and runs example code  : see examples/*/goc.sh

    local msg="=== $FUNCNAME :"
    [ "$(which oc)" == "" ] && echo $msg error oc opticks-config script is not in PATH && return 1

    local sdir=$(pwd)
    local snam=$(basename $sdir)
    local bdir=/tmp/$USER/opticks/$snam/build


    local iwd=$PWD
    rm -rf $bdir
    mkdir -p $bdir && cd $bdir && pwd


    local pkg=${snam/NoCMake}
    pkg=${pkg/Use}

    echo $msg snam $snam pkg $pkg
    local ccs=$sdir/*.cc
    local num_main=0
    local num_unit=0

    : compile cc without mains
    for cc in $ccs
    do

        if [[ "$(grep ^int\ main $cc)" == "int main"* ]]; then
            num_main=$(( ${num_main} + 1 ))
        else
            num_unit=$(( ${num_unit} + 1 ))
            echo gcc -c $cc $(oc -cflags $pkg) -fpic
                 gcc -c $cc $(oc -cflags $pkg) -fpic
        fi
    done


    local libflag=""
    if [ "$num_unit" != "0" ]; then

        local sfx=""
        case $(uname) in
          Darwin) sfx=dylib ;;
           Linux) sfx=so ;;
        esac
        local libname=Use$pkg
        local lib=lib$libname.$sfx

        : create a library of the non-mains
        echo gcc -shared -o $lib $(ls *.o) $(oc -libs $pkg)
             gcc -shared -o $lib $(ls *.o) $(oc -libs $pkg)

        libflag="-L$(pwd) -l$libname"
    fi


    : compile and link the mains and run them
    for cc in $ccs
    do
        if [[ "$(grep ^int\ main $cc)" == "int main"* ]]; then
            main=$cc
            name=$(basename $cc)
            name=${name/.cc}
            echo main $main name $name libflag $libflag

            echo gcc -c $cc -o $name.o $(oc -cflags $pkg)
                 gcc -c $cc -o $name.o $(oc -cflags $pkg)

            echo gcc -o $name $name.o $libflag $(oc -libs $pkg)
                 gcc -o $name $name.o $libflag $(oc -libs $pkg)

            echo ./$name
                 ./$name
        fi
    done
    cd $iwd
}


opticks-setup-funcs-(){ opticks-setup-funcs-- | perl -pe 's,cd_func,cd,g' - | perl -pe 's,--color=auto,,g' -  ; }
opticks-setup-funcs--(){
   echo "# $FUNCNAME"
   declare -f opticks-setup-
   declare -f opticks-setup-info-
   declare -f opticks-setup-info
   #declare -f opticks-gob
   #declare -f opticks-goc
   declare -f opticks-setup-find-config-prefix
   declare -f opticks-setup-find-geant4-prefix

   # type emits "name is a function" in some versions of bash
   # requiring : perl -pe 's,^(\S* is a function),#$1,' -
   # but it seems declare -f is more uniform
}

opticks-setup-prefix-OLD-(){ cat << EHEAD
# $FUNCNAME

# mandatory envvars from buildenv propagated into userenv via this setup
export OPTICKS_PREFIX=$OPTICKS_PREFIX
export OPTICKS_CUDA_PREFIX=$OPTICKS_CUDA_PREFIX
export OPTICKS_OPTIX_PREFIX=$OPTICKS_OPTIX_PREFIX

HERE_OPTICKS_PREFIX=\$(dirname \$(dirname \$BASH_SOURCE))

if [ "\$OPTICKS_PREFIX" != "\$HERE_OPTICKS_PREFIX" ]; then
   echo \$MSG build time OPTICKS_PREFIX \$OPTICKS_PREFIX is not consistent with HERE_OPTICKS_PREFIX \$HERE_OPTICKS_PREFIX
   echo \$MSG opticks setup scripts cannot be moved
   return 1
else
   echo \$MSG build time OPTICKS_PREFIX \$OPTICKS_PREFIX is consistent with HERE_OPTICKS_PREFIX \$HERE_OPTICKS_PREFIX
fi

#echo \$MSG $FUNCNAME

EHEAD
}

opticks-setup-prefix-(){ cat << EHEAD
# $FUNCNAME

# getting OPTICKS_PREFIX from HERE location assuming depth 1
HERE_OPTICKS_PREFIX=\$(dirname \$(dirname \$BASH_SOURCE))
export OPTICKS_PREFIX=\$HERE_OPTICKS_PREFIX
export OPTICKS_CUDA_PREFIX=\${OPTICKS_CUDA_PREFIX:-$OPTICKS_CUDA_PREFIX}
export OPTICKS_OPTIX_PREFIX=\${OPTICKS_OPTIX_PREFIX:-$OPTICKS_OPTIX_PREFIX}
#

EHEAD
}

opticks-bashrc-prefix-(){ cat << EHEAD
# $FUNCNAME

# getting OPTICKS_PREFIX from HERE location assuming depth 0
HERE_OPTICKS_PREFIX=\$(dirname \$BASH_SOURCE)
export OPTICKS_PREFIX=\$HERE_OPTICKS_PREFIX
export OPTICKS_CUDA_PREFIX=\${OPTICKS_CUDA_PREFIX:-$OPTICKS_CUDA_PREFIX}
export OPTICKS_OPTIX_PREFIX=\${OPTICKS_OPTIX_PREFIX:-$OPTICKS_OPTIX_PREFIX}
#

EHEAD
}






opticks-setup-()
{
   local mode=${1:-prepend}
   local var=${2:-PATH}
   local dir=${3:-/tmp}

   local st=""
   : dir exists and is not in the path variable already
   if [ -d "$dir" ]; then

       if [[ ":${!var}:" != *":${dir}:"* ]]; then
           if [ -z "${!var}" ]; then
               export $var=$dir
               st="new"
           else
               st="add"
               case $mode in
                  prepend) export $var=$dir:${!var}  ;;
                   append) export $var=${!var}:$dir  ;;
               esac
           fi
        else
           st="skip"
        fi
    else
        st="nodir"
    fi
    if [ -n "$OPTICKS_SETUP_VERBOSE" ];  then printf "=== %s %10s %10s %20s %s\n" $FUNCNAME $st $mode $var $dir ; fi
}

opticks-setup-info-()
{
   for var in $* ; do
      echo $var
      echo ${!var} | tr ":" "\n"
      echo
   done
}

opticks-setup-info()
{
   opticks-setup-info- PATH CMAKE_PREFIX_PATH PKG_CONFIG_PATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH

   echo "env | grep OPTICKS"
   env | grep OPTICKS
}


opticks-setup-paths-(){ cat << EOS
# $FUNCNAME
# THIS IS THE PRIMARY PURPOSE OF THE GENERATED SCRIPT
# SETTING PATH VARIABLES TO ALLOW USE OF OPTICKS EXECUTABLES
# AND EXTERNAL LIBS THAT THEY DEPEND UPON

opticks-setup- append PATH \$OPTICKS_CUDA_PREFIX/bin   ## nvcc
opticks-setup- append PATH \$OPTICKS_PREFIX/bin
opticks-setup- append PATH \$OPTICKS_PREFIX/lib

opticks-setup- append PYTHONPATH \$OPTICKS_PREFIX/py

opticks-setup- append CMAKE_PREFIX_PATH \$OPTICKS_PREFIX
opticks-setup- append CMAKE_PREFIX_PATH \$OPTICKS_PREFIX/externals
opticks-setup- append CMAKE_PREFIX_PATH \$OPTICKS_OPTIX_PREFIX

opticks-setup- append PKG_CONFIG_PATH \$OPTICKS_PREFIX/lib/pkgconfig
opticks-setup- append PKG_CONFIG_PATH \$OPTICKS_PREFIX/lib64/pkgconfig
opticks-setup- append PKG_CONFIG_PATH \$OPTICKS_PREFIX/externals/lib/pkgconfig
opticks-setup- append PKG_CONFIG_PATH \$OPTICKS_PREFIX/externals/lib64/pkgconfig


EOS
}

opticks-setup-libpaths-(){
    local LIBRARY_PATH
    case $(uname) in
       Darwin) LIBRARY_PATH="DYLD_LIBRARY_PATH" ;;
        Linux) LIBRARY_PATH="LD_LIBRARY_PATH" ;;
    esac

cat << EOS
# $FUNCNAME
opticks-setup- append $LIBRARY_PATH \$OPTICKS_PREFIX/lib
opticks-setup- append $LIBRARY_PATH \$OPTICKS_PREFIX/lib64
opticks-setup- append $LIBRARY_PATH \$OPTICKS_PREFIX/externals/lib
opticks-setup- append $LIBRARY_PATH \$OPTICKS_PREFIX/externals/lib64

opticks-setup- append $LIBRARY_PATH \$OPTICKS_CUDA_PREFIX/lib
opticks-setup- append $LIBRARY_PATH \$OPTICKS_CUDA_PREFIX/lib64

EOS
}

opticks-setup-misc-(){ cat << EOM
# $FUNCNAME

export TMP=\${TMP:-/tmp/\$USER/opticks}
## mkdir -p \${TMP}
## see sysrap/STTF.h sysrap/sconfig.h should now default to this, so not needed for source build
## HMM it might still be needed for binary release, and does no harm
export OPTICKS_STTF_PATH=\$OPTICKS_PREFIX/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf

EOM
}

opticks-setup-misc-removed-(){ cat << EOR
export OPTICKS_EVENT_BASE=\${OPTICKS_EVENT_BASE:-/tmp/\$USER/opticks}
mkdir -p \${OPTICKS_EVENT_BASE}

EOR
}

opticks-geant4-prefix-notes(){ cat << EON

Getting the prefix relies on find_package.py which
depends on PATH and CMAKE_PREFIX_PATH
it is far too complicated to do in the setup script.

It must be run during opticks-setup by the administrator/developer
that is installing opticks with the result hardcoded into the setup.

Chicken-and-egg dependency probelm :

the find_package needs the setup environment, but this is
still trying to generate that

Instead treat OPTICKS_GEANT4_PREFIX as a user input

Just use opticks-setup-find-geant4-prefix



EON
}

#opticks-geant4-prefix(){ echo ${OPTICKS_GEANT4_PREFIX:-$(opticks-prefix)/externals} ; }

opticks-check-geant4(){

    local msg="=== $FUNCNAME :"
    local g4_prefix=$(opticks-setup-find-geant4-prefix)  # search CMAKE_PREFIX_PATH for Geant4Config.cmake
    local g4_script=${g4_prefix}/bin/geant4.sh

    if [ -z "$g4_prefix" ]; then
        echo $msg ERROR no g4_prefix : failed to find Geant4Config.cmake along CMAKE_PREFIX_PATH
        return 1
    fi

    if [ ! -f "$g4_script" ]; then
        echo $msg ERROR g4_script $g4_script does not exist : Geant4 installation must be incomplete
        return 2
    fi
   # $(opticks-home)/bin/find_package.py G4 --index 0 --nocache
    return 0
}


opticks-check-prefix(){
    local msg="=== $FUNCNAME :"
    if [ -z "$OPTICKS_PREFIX" ]; then
        echo $msg ERROR no OPTICKS_PREFIX envvar
        return 1
    fi
    if [ ! -d "$OPTICKS_PREFIX" ]; then
        echo $msg ERROR no OPTICKS_PREFIX $OPTICKS_PREFIX directory
        return 2
    fi
    if [ "$OPTICKS_PREFIX" == "$(opticks-home)" ]; then
        echo $msg ERROR OPTICKS_PREFIX $OPTICKS_PREFIX directory MUST NOT be the same as source directory opticks-home:$(opticks-home)
        return 3
    fi
    return 0
}


opticks-check-compute-capability-msg(){ cat << EON
$FUNCNAME
=======================================

Envvar OPTICKS_COMPUTE_CAPABILITY : $OPTICKS_COMPUTE_CAPABILITY

The OPTICKS_COMPUTE_CAPABILITY must be set appropriately
for your GPU as it controls the CUDA compilation flags.
See cmake/Modules/OpticksCUDAFlags.cmake

To obtain the compute capability of your GPU run the
deviceQuery program from the CUDA samples, it
reports a line like the below::

    CUDA Capability Major/Minor version number:    7.0

This corresponds setting the below envvar in your ~/.opticks_config::

    export OPTICKS_COMPUTE_CAPABILITY=70

Opticks/OptiX supports a compute capability greater than or equal to 30
although modern GPUs are typically 70 or 75

EON
}

opticks-check-compute-capability(){
    local msg="=== $FUNCNAME :"
    if [ -z "$OPTICKS_COMPUTE_CAPABILITY" ]; then
        echo $msg ERROR no OPTICKS_COMPUTE_CAPABILITY envvar
        opticks-check-compute-capability-msg
        return 1
    fi

   local occ=$OPTICKS_COMPUTE_CAPABILITY
    if [ "${occ/.}" != "${occ}"  ]; then
       echo $msg OPTICKS_COMPUTE_CAPABILITY $occ : ERROR envvar must contain an integer expression such as 70 or 75 , not 7.0 or 7.5
       opticks-check-compute-capability-msg
       return 2
    fi

    if [ $occ -lt 30 ]; then
        echo $msg OPTICKS_COMPUTE_CAPABILITY $occ : ERROR it must must be 30 or more
        opticks-check-compute-capability-msg
        return 3
    else
        echo $msg OPTICKS_COMPUTE_CAPABILITY $occ : looking good it is an integer expression of  30 or more
    fi
    return 0
}




opticks-tools(){ cat << EOT
cmake
git
make
python
EOT
}

opticks-check-tools(){
   local msg="=== $FUNCNAME :"
   local tool
   for tool in $(opticks-tools) ; do
       [ ! -x "$(which $tool 2>/dev/null)" ] && echo $msg missing $tool && return 1
   done
   return 0
}



opticks-prepend-prefix-notes(){ cat << EON

opticks-prepend-prefix
    intended for build environment path setup, ie configuring access to "foreign" externals,
    ie externals that are not built by opticks-externals-install

1. check the prefix by seeing if it exists and contains bin lib64 lib
2. if CMAKE_PREFIX_PATH is defined prepend the prefix to it otherwise set CMAKE_PREFIX_PATH to the prefix
3. ditto for PKG_CONFIG_PATH but using libdir/pkgconfig where libdir is lib64 or lib
4. if the bindir exists then then do the same for PATH
5. same for the library path

EON
}


opticks-prepend-prefix(){
    local msg="=== $FUNCNAME :"
    local prefix=$1
    [ ! -d "$prefix" ] && echo $msg prefix $prefix does not exist && return 1

    local bindir=$prefix/bin
    local libdir=""
    if [ -d "$prefix/lib64" ]; then
        libdir=$prefix/lib64
    elif [ -d "$prefix/lib" ]; then
        libdir=$prefix/lib
    fi

    [ -z "$libdir" ] && echo $msg FAILED to find libdir under prefix $prefix && return 2

    if [ -z "$CMAKE_PREFIX_PATH" ]; then
        export CMAKE_PREFIX_PATH=$prefix
    else
        export CMAKE_PREFIX_PATH=$prefix:$CMAKE_PREFIX_PATH
    fi

    if [ -z "$PKG_CONFIG_PATH" ]; then
        export PKG_CONFIG_PATH=$libdir/pkgconfig
    else
        export PKG_CONFIG_PATH=$libdir/pkgconfig:$PKG_CONFIG_PATH
    fi

    if [ -d "$bindir" ]; then
        if [ -z "$PATH" ]; then
            export PATH=$bindir
        else
            export PATH=$bindir:$PATH
        fi
    fi

    case $(uname) in
       Darwin) libpathvar=DYLD_LIBRARY_PATH ;;
        Linux) libpathvar=LD_LIBRARY_PATH ;;
    esac

    if [ -z "${!libpathvar}" ]; then
        export ${libpathvar}=$libdir
    else
        export ${libpathvar}=$libdir:${!libpathvar}
    fi
}







opticks-setup-geant4-(){ cat << EOS
# $FUNCNAME

export OPTICKS_GEANT4_PREFIX=\$(opticks-setup-find-geant4-prefix)

if [ -n "\$OPTICKS_GEANT4_PREFIX" ]; then
    if [ -f "\$OPTICKS_GEANT4_PREFIX/bin/geant4.sh" ]; then
        [ -n "\$OPTICKS_SETUP_VERBOSE" ] && echo === $FUNCNAME : sourcing \$OPTICKS_GEANT4_PREFIX/bin/geant4.sh
        source \$OPTICKS_GEANT4_PREFIX/bin/geant4.sh
    else
        echo ERROR no \$OPTICKS_GEANT4_PREFIX/bin/geant4.sh at OPTICKS_GEANT4_PREFIX : \$OPTICKS_GEANT4_PREFIX
        return 1
    fi
else
    echo === $FUNCNAME : WARNING no OPTICKS_GEANT4_PREFIX : Geant4 will need be setup by other means
fi

EOS
}


# for finding system boost
opticks-boost-includedir(){ echo ${OPTICKS_BOOST_INCLUDEDIR:-/tmp} ; }
opticks-boost-libdir(){     echo ${OPTICKS_BOOST_LIBDIR:-/tmp} ; }
opticks-boost-info(){ cat << EOI
$FUNCNAME
===================

NB have moved to a more flexible approach to control the version of
Boost to use via CMAKE_PREFIX_PATH/PKG_CONFIG_PATH see oc.bash om.bash

   opticks-boost-includedir : $(opticks-boost-includedir)
   opticks-boost-libdir     : $(opticks-boost-libdir)

EOI
}

opticks-cuda-capable()
{
   : rc 0 when there is a CUDA capable GPU
   case $(uname) in
      Linux) nvidia-smi 1>/dev/null ;;
     Darwin) system_profiler SPDisplaysDataType | grep NVIDIA > /dev/null  ;;
   esac
}


opticks-full-notes(){ cat << EON
opticks-full-notes
====================

opticks-full has traditionally been used for full source builds,
getting externals and building them generating and
invoking setup to configure use of those externals.
Now that are moving more to release builds could consider
getting externals from former releases.

HMM: but thats kinda complicated : and probably not worthy of the
effort, the opticks externals do not take long to build.


opticks-info
    dump config of the build

opticks-full-externals
    for each of the externals : bcm glm imgui plog nljson
    invokes a pair of bash functions eg "bcm-;bcm--"
    that get and install the externals into $OPTICKS_PREFIX/externals

opticks-full-make
    invokes opticks-setup-generate writing $OPTICKS_PREFIX/bin/opticks-setup.sh
    then sources that with opticks-setup and calls om- om-install
    installing Opticks into $OPTICKS_PREFIX/lib etc..

opticks-install-extras
    opticks-install-cmake-modules enables building against the installed opticks with CMake,
    opticks-install-tests enables testing the installed opticks with ctest

opticks-cuda-capable
    checks for a GPU using nvidia-smi

opticks-full-prepare
    when a GPU is detected invokes opticks-prepare-installation to create
    QCurandState files in default dir ~/.opticks/rngcache/RNG

EON
}

opticks-full()
{
    local msg="=== $FUNCNAME :"
    local rc

    opticks-info
    [ $? -ne 0 ] && echo $msg ERR from opticks-info && return 1

    opticks-full-externals
    [ $? -ne 0 ] && echo $msg ERR from opticks-full-externals && return 2

    opticks-full-make
    [ $? -ne 0 ] && echo $msg ERR from opticks-full-make && return 3

    opticks-install-extras
    [ $? -ne 0 ] && echo $msg ERR from opticks-install-extras && return 4

    opticks-cuda-capable
    rc=$?
    if [ $rc -eq 0 ]; then
        echo $msg detected GPU proceed with opticks-full-prepare
        opticks-full-prepare
        rc=$?
        [ $rc -ne 0 ] && echo $msg ERR from opticks-full-prepare && return 5
    else
        echo $msg detected no CUDA cabable GPU - skipping opticks-full-prepare
        rc=0
    fi
    return 0
}

opticks-full-externals()
{
    local msg="=== $FUNCNAME :"
    echo $msg START $(date)
    local rc

    echo $msg installing the below externals into $(opticks-prefix)/externals : eg bcm glm glfw glew gleq imgui plog nljson
    opticks-externals
    opticks-externals-install
    rc=$?
    [ $rc -ne 0 ] && return $rc

    echo $msg config-ing the preqs : eg cuda optix
    opticks-preqs
    opticks-preqs-pc
    rc=$?
    [ $rc -ne 0 ] && return $rc

    echo $msg config-ing the foreign : eg boost clhep xercesc g4
    opticks-foreign
    #opticks-foreign-pc
    rc=$?
    [ $rc -ne 0 ] && return $rc

    echo $msg DONE $(date)
    return 0
}

opticks-full-make()
{
    local msg="=== $FUNCNAME :"
    echo $msg START $(date)
    local rc

    echo $msg generating setup script
    opticks-setup-generate
    rc=$?
    [ $rc -ne 0 ] && return $rc

    local setup=$(opticks-setup-path)
    [ ! -f "$setup" ] && echo $msg ABORT missing opticks setup script $setup && return 1

    echo $msg running setup : as opticks-full-prepare needs to find and run executable
    opticks-setup

    om-
    cd $(om-home)
    om-cleaninstall
    rc=$?
    [ $rc -ne 0 ] && return $rc

    echo $msg DONE $(date)
    return 0
}

opticks-install-extras()
{
   local msg="=== $FUNCNAME :"
   local iwd=$PWD

   opticks-cd  ## install directory

   echo $msg install cmake/Modules
   opticks-install-cmake-modules
   [ $? -ne 0 ] && echo $msg ERROR after opticks-install-cmake-modules && return 1

   echo $msg install ctest
   opticks-install-tests
   [ $? -ne 0 ] && echo $msg ERROR after opticks-install-tests && return 1

   cd $iwd
   return 0
}



opticks-install-tests()
{
   : formerly this was okdist-install-tests

   local msg="=== $FUNCNAME :"
   local bdir=$(opticks-bdir)
   local dest=$(opticks-dir)/tests
   local script=$(opticks-dir)/bin/CTestTestfile.py
   local fold=$(opticks-fold)

   local vars="FUNCNAME bdir dest script fold"
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done

   : using the source tree for PYTHONPATH as this is needed
   : as part of the installation

   PYTHONPATH=$fold $script $bdir --dest $dest
   local testscript=$dest/ctest.sh
   opticks-install-tests-script- > $testscript
   chmod ugo+x $testscript
}

opticks-install-tests-script-(){ cat << EOS
#!/bin/bash

cd \$(dirname \$(realpath \$BASH_SOURCE))

# As ctest doesnt currently work from a readonly dir
# this copies the released ctest metadata folders to TTMP
# directory and runs the ctest from there
#
# ctest -N
# ctest --output-on-failure

ttmp=/tmp/\$USER/opticks/tests
TTMP=\${TTMP:-\$ttmp}
mkdir -p \$TTMP
cp -r . \${TTMP}/

cd \$TTMP
pwd

ctest -N
ctest --output-on-failure

EOS
}


opticks-install-cmake-modules()
{
   : formerly this was okdist-install-cmake-modules
   local msg="=== $FUNCNAME :"
   local home=$(opticks-home)
   local dest=$(opticks-prefix)
   local script=$(opticks-prefix)/bin/CMakeModules.py

   local vars="FUNCNAME home dest script"
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done

   $script --home $home --dest $dest
}






opticks-full-prepare()
{
    : this needs a CUDA capable GPU
    local msg="=== $FUNCNAME :"
    echo $msg START $(date)
    local rc
    opticks-prepare-installation
    rc=$?
    [ $rc -ne 0 ] && return $rc
    echo $msg DONE $(date)
    return 0
}




opticks-ext-setup()
{
   : append outputs from existing $ext-setup funcs into opticks-setup-path

   local msg="=== $FUNCNAME :"
   local rc
   while read ext
   do
        echo $msg $ext
        $ext-

        type $ext-setup 1> /dev/null 2> /dev/null
        rc=$?
        if [ "$rc" == "0" ]; then
            $ext-setup >> $(opticks-setup-path)
            #$ext-setup >> $(opticks-release-path)
            rc=$?
            [ $rc -ne 0 ] && echo $msg RC $rc from ext $ext : ABORTING && return $rc
        else
            echo -n
            #echo $msg missing function $ext-setup
        fi
   done
   return 0
}


opticks-ext-url(){
   local ext
   while read ext
   do
        $ext-
        printf "%30s :  %s \n" $ext $($ext-url)
   done
}

opticks-ext-dist(){
   local ext
   local dist
   while read ext
   do
        $ext-
        dist=$($ext-dist 2>/dev/null)
        printf "%30s :  %s \n" $ext $dist
   done
}



opticks-ext-dir(){
   local ext
   local dir
   while read ext
   do
        $ext-
        dir=$($ext-dir 2>/dev/null)
        printf "%30s :  %s \n" $ext $dir
   done
}
opticks-ext-status(){
   local ext
   local dir
   local iwd=$PWD
   while read ext
   do
        $ext-
        dir=$($ext-dir 2>/dev/null)
        printf "\n\n%30s :  %s \n" $ext $dir
        if [ -d "$dir" ]; then
            cd $dir
            [ -d ".hg" ]  && hg paths -v && hg status .
            [ -d ".git" ] && git remote -v && git status .
        else
            echo no such dir : maybe system repo install
        fi
   done
   cd $iwd
}



opticks-git-clone-notes(){  cat << EOC

git clone wrapper similar to opticks-curl that will clone
from local bare repos in $OPTICKS_DOWNLOAD_CACHE in order to avoid the sometimes
very slow or failing cloning thru the GFW

Subseqently decided to favor git repo with working directory
over bare git repo as its quicker to check the version in use.

Github repo url : https://github.com/simoncblyth/plog.git



EOC
}

opticks-git-clone(){
   local msg="=== $FUNCNAME :"
   local dir=$PWD
   local url=$1
   local repo=$(basename $url)
   local stem=${repo/.git}

   if [ -z "$url" -o -z "$repo" ]; then
       cmd="echo $msg BAD url $url repo $repo"
   elif [ -n "$OPTICKS_DOWNLOAD_CACHE" -a -d "$OPTICKS_DOWNLOAD_CACHE/$stem" ]; then
       cmd="git clone $OPTICKS_DOWNLOAD_CACHE/$stem"
   elif [ -n "$OPTICKS_DOWNLOAD_CACHE" -a -d "$OPTICKS_DOWNLOAD_CACHE/$repo" ]; then
       cmd="git clone $OPTICKS_DOWNLOAD_CACHE/$repo"
   else
       cmd="git clone $url"
   fi
   echo $msg dir $dir url $url stem $stem OPTICKS_DOWNLOAD_CACHE $OPTICKS_DOWNLOAD_CACHE cmd $cmd
   eval $cmd
}


opticks-curl-notes(){ cat << EON

*opticks-curl url*
    when OPTICKS_DOWNLOAD_CACHE envvar is defined and OPTICKS_DOWNLOAD_CACHE/dist
    exists where dist is the basename obtained from the url then the dist is
    copied to the pwd instead of being curled there

    Precision account O "blyth" defines in .bashrc the OPTICKS_DOWNLOAD_CACHE as /data/opticks_download_cache

EON
}


opticks-curl(){
   local msg="=== $FUNCNAME :"
   local dir=$PWD
   local url=$1
   local dist=$(basename $url)
   local cmd
   if [ -z "$url" -o -z "$dist" ]; then
       cmd="echo $msg BAD url $url dist $dir"
   elif [ -n "$OPTICKS_DOWNLOAD_CACHE" -a -f "$OPTICKS_DOWNLOAD_CACHE/$dist" ]; then
       cmd="cp $OPTICKS_DOWNLOAD_CACHE/$dist $dist"
   else
       cmd="curl -L -O $url"
   fi
   echo $msg dir $dir url $url dist $dist OPTICKS_DOWNLOAD_CACHE $OPTICKS_DOWNLOAD_CACHE cmd $cmd
   eval $cmd
}

opticks-tag(){
   local iwd=$PWD
   cd $(opticks-home)
   git tag | sed -n '$p'
   cd $iwd
}


opticks-tar()
{
    : create and extract binary release tarball

    local msg=" === $FUNCNAME :"
    okdist-
    okdist--

    [ $? -ne 0 ] && echo $msg ERROR after okdist-- && return 1
    return 0
}




opticks-g4-clean-build()
{
    local arg="extg4:"

    om-
    om-subs $arg
    om-clean $arg

    type $FUNCNAME
    read -p "$FUNCNAME : enter YES to proceed to to clean and build : " ans

    [ "$ans" != "YES" ] && echo skip && return

    om-clean $arg | sh
    om-conf $arg
    om-make $arg
}



opticks-locations(){ cat << EOL

$FUNCNAME
==================

      OPTICKS_PREFIX  :    $OPTICKS_PREFIX
      opticks-prefix  :    $(opticks-prefix)
      #opticks-optix-install-dir :
      opticks-optix-prefix :  $(opticks-optix-prefix)
      opticks-cuda-prefix :  $(opticks-cuda-prefix)



      opticks-source   :   $(opticks-source)
      opticks-home     :   $(opticks-home)
      opticks-name     :   $(opticks-name)


      opticks-sdir     :   $(opticks-sdir)
      opticks-idir     :   $(opticks-idir)
      opticks-bdir     :   $(opticks-bdir)
      opticks-xdir     :   $(opticks-xdir)
      ## cd to these with opticks-scd/icd/bcd/xcd

      opticks-installcachedir   :  $(opticks-installcachedir)
      opticks-bindir            :  $(opticks-bindir)

EOL
}


opticks-info(){
   opticks-externals-info
   opticks-locations
   opticks-env-info
   opticks-externals-url
   opticks-externals-dist
   opticks-foreign-url
   opticks-foreign-dist
}




opticks-wipe(){
  local msg="=== $FUNCNAME : "
   local bdir=$(opticks-bdir)
   echo $msg wiping build dir $bdir
   rm -rf $bdir
}


opticks--(){
   local bdir=$1
   if [ "$bdir" == "" ]; then
      bdir=$(opticks-home)
   fi
   shift
   local iwd=$(pwd)
   cd $bdir
   om-
   om-make $1

   cd $iwd
}



opticks-prepare-installation-notes(){ cat << EON

$FUNCNAME
===================================


RNG
    ~/.opticks/rngcache/RNG

    HMM: maybe better to share this, eg for use on a GPU cluster,
    could use an envvar OPTICKS_CACHE_PREFIX
    Which defaults to $HOME/.opticks to facilitate sharing

EON
}



opticks-prepare-InputPhotons-notes(){ cat << EON
opticks-prepare-InputPhotons-notes
------------------------------------

opticks-prepare-InputPhotons is run during installation by opticks-full-prepare
so it is done just after the build and before opticks environment is setup.
So it sets PYTHONPATH to opticks-fold which is the folder above opticks-home
in order to be able to import python modules from the source tree which
are needed to generate the input photons.

EON
}


opticks-prepare-InputPhotons()
{
    local msg="=== $FUNCNAME :"
    mkdir -p $HOME/.opticks/InputPhotons
    echo $msg used by some tests

    PYTHONPATH=$(opticks-fold) $(opticks-home)/ana/input_photons.sh
}


opticks-prepare-installation()
{
    local msg="=== $FUNCNAME :"
    echo $msg

    #qudarap-
    #qudarap-prepare-installation
    : following move to Philox no more need to prepare curandState files

    opticks-prepare-InputPhotons
}

opticks-check-installation()
{
    local msg="=== $FUNCNAME :"
    local rc=0
    local iwd=$PWD

    qudarap-
    qudarap-check-installation
    rc=$?

    cd $iwd
    echo $msg rc $rc
    return $rc
}

opticks-former-check-installation()
{
    # see notes/issues/opticks-t-how-difficult-to-revive-with-new-geometry-workflow.rst

    local msg="=== $FUNCNAME :"
    local rc=0
    local iwd=$PWD

    local dir=$(opticks-installcachedir)

    if [ ! -d "$dir" ]; then
        echo $msg missing dir $dir
        rc=100
    else
        if [ ! -d "$dir/PTX" ]; then
            echo $msg $dir/PTX : missing PTX : compiled OptiX programs created when building oxrap-
            rc=101
        else
            qudarap-
            qudarap-check-installation
            rc=$?
        fi
    fi

    cd $iwd
    echo $msg rc $rc
    return $rc
}


opticks-t(){  opticks-t- $* ; }   ## see om-test-one for details of ctest arguments
opticks-t0(){ CUDA_VISIBLE_DEVICES=0 opticks-t $* ; }
opticks-t1(){ CUDA_VISIBLE_DEVICES=1 opticks-t $* ; }

opticks-i(){
   local iwd=$PWD
   om-
   local bdir=$(om-bdir integration)
   cd $bdir
   om-test -V
   cd $iwd
}


opticks-former-check-key()
{
   local msg="=== $FUNCNAME :"
   if [ -z "$OPTICKS_KEY" ]; then
       echo $msg OPTICKS_KEY envvar is not defined : read the docs https://simoncblyth.bitbucket.io/opticks/docs/testing.html
       return 1
   fi
   return 0
}

opticks-check-no-key()
{
   local msg="=== $FUNCNAME :"
   if [ -n "$OPTICKS_KEY" ]; then
       echo $msg ERROR : OPTICKS_KEY envvar is defined : that is very old workflow
       return 1
   fi
   return 0
}






opticks-t-notes(){ cat << EON

Basic environment (PATH and envvars to find data)
should happen at profile level (or at least be invoked from there)
not here (and there) for clarity of a single location
where smth is done.

Powershell presents a challenge to this principal,
TODO:find a cross platform way of doing envvar setup

EON
}


opticks-t-()
{
   local msg="=== $FUNCNAME : "
   local iwd=$PWD

   local rc=0
   opticks-check-installation
   rc=$?
   [ $rc -ne 0 ] && echo $msg ABORT : missing installcache components : create with opticks-prepare-installation && return $rc

   opticks-check-no-key
   rc=$?
   [ $rc -ne 0 ] && echo $msg ABORT : opticks-check-no-key failed && return $rc


   ## if 1st arg is a directory, cd there to run ctest
   ## otherwise run from the top level bdir

   local arg=$1
   if [ "${arg:0:1}" == "/" -a -d "$arg" ]; then
       bdir=$arg
       shift
   else
       bdir=$(opticks-bdir)
   fi
   cd $bdir

   om-
   om-test
   om-test-help

   cd $iwd
}

opticks-t-notes(){ cat << EON
$FUNCNAME
=====================

*opticks-t-* is invoked by the subproj test functions such as *oxrap-t*
with the corresponding build dir as the first argument.


EON
}


opticks-tl(){ om- ; om-testlog $* ; om-test-help ;  }
opticks-tld(){ opticks-tl --level debug ; }


opticks-ts()
{
   ## list tests taking longer than 1 second
   local arg=$1
   if [ "${arg:0:1}" == "/" -a -d "$arg" ]; then
       bdir=$arg
       shift
   else
       bdir=$(opticks-bdir)
   fi
   #perl -n -e 'm,[123456789]\.\d{2} sec, && print  ' $bdir/ctest.log   ## this missins 10.00 20.00
   grep " sec" $bdir/ctest.log | grep -v " 0.* sec" -

}


opticks-t--()
{
   [ "$(which ctest 2>/dev/null)" == "" ] && ctest3 $* || ctest $*
}


opticks-ifind(){ opticks-find "$1" -Hi ; }
opticks-findl(){ opticks-find "$1" -l ; }
opticks-find(){
   local str="${1:-ENV_HOME}"
   local opt=${2:--H}

   local iwd=$PWD
   opticks-scd

   find . -name '*.sh' -exec grep $opt "$str" {} \;
   find . -name '*.bash' -exec grep $opt "$str" {} \;
   find . -name '*.cu' -exec grep $opt "$str" {} \;
   find . -name '*.cc' -exec grep $opt "$str" {} \;
   find . -name '*.hh' -exec grep $opt "$str" {} \;
   find . -name '*.cpp' -exec grep $opt "$str" {} \;
   find . -name '*.hpp' -exec grep $opt "$str" {} \;
   find . -name '*.h' -exec grep $opt "$str" {} \;
   find . -name '*.txt' -exec grep $opt "$str" {} \;
   find . -name '*.py' -exec grep $opt "$str" {} \;
   find . -name '*.cmake' -exec grep $opt "$str" {} \;

   #cd $iwd
}



opticks-if(){ opticks-f "$1" -Hi ; }
opticks-fl(){ opticks-f "$1" -l ; }
opticks-fh(){ HERE=1 opticks-f $* ; }
opticks-fa(){ ANCIENT=1 opticks-f $* ; : search a sibling opticks_ancient tree ;  }
opticks-f(){
   : search C/C++ code/headers, txt, cmake python scripts etc.. BUT NOT .rst
   : .rst due to too many hits within issues
   : to seach that use opticks-r

   local str="${1:-ENV_HOME}"
   local opt=${2:--H}

   local iwd=$PWD

   if [ -n "$ANCIENT" ]; then
       opricks-acd
   else
       [ -z "$HERE" ] && opticks-scd
       : when HERE is defined use the invoking directory : otherwise start from opticks-sdir
   fi

   find . \
        \( \
       -name '*.sh' -or \
       -name '*.bash' -or \
       -name '*.cu' -or \
       -name '*.cc' -or \
       -name '*.hh' -or \
       -name '*.cpp' -or \
       -name '*.hpp' -or \
       -name '*.h' -or \
       -name '*.txt' -or \
       -name '*.cmake' -or \
       -name '*.py' \
        \) \
       -exec grep $opt "$str" {} \;

}

opticks-py(){ XX=py opticks-xx $* ; }
opticks-sh(){ XX=sh opticks-xx $* ; }
opticks-xx()
{
   : search opticks python scripts

   local str="${1:-ENV_HOME}"
   local opt=${2:--H}
   [ -z "$HERE" ] && opticks-scd

   opticks-scd
   find . \
        \( \
       -name "*.$XX" \
        \) \
       -exec grep $opt "$str" {} \;

}


opticks-rl(){ opticks-r "$1" -l ; }
opticks-r(){
   : search rst, bash, txt, cmake py  BUT not code or headers
   local str="${1:-ENV_HOME}"
   local opt=${2:--H}

   local iwd=$PWD
   opticks-scd

   find . \
        \( \
       -name '*.rst' -or \
       -name '*.bash' -or \
       -name '*.txt' -or \
       -name '*.cmake' -or \
       -name '*.py' \
        \) \
       -exec grep $opt "$str" {} \;

}





opticks-c(){
   : search C/C++ code/headers only, exclude txt cmake python scripts etc..
   local str="${1:-ENV_HOME}"
   local opt=${2:--H}

   local iwd=$PWD
   opticks-scd

   find . \
        \( \
       -name '*.cc' -or \
       -name '*.hh' -or \
       -name '*.cpp' -or \
       -name '*.hpp' -or \
       -name '*.h' \
        \) \
       -exec grep $opt "$str" {} \;
}





opticks-unset--()
{
   local pfx=${1:-OPTICKS_}
   local kv
   local k
   local v
   env | grep $pfx | while read kv ; do

       k=${kv/=*}
       v=${kv/*=}

       #printf "%50s %s \n" $k $v
       echo unset $k
   done
}

opticks-unset-()
{
   opticks-unset-- OPTICKS_
   opticks-unset-- DAE_
   opticks-unset-- IDPATH
}
opticks-unset()
{
   local tmp=/tmp/unset.sh
   opticks-unset- >  $tmp

   echo unset with : . $tmp
}



# opticks-all-projs- nearly duplicates om-subs--all so have removed to .old




opticks-log(){ find $(opticks-home) -name '*.log' ; }
opticks-rmlog(){ find $(opticks-home) -name '*.log' -exec rm -f {} \; ; }
opticks-distclean(){ opticks-rmdirs- bin build gl include lib ptx  ; }
opticks-fullclean(){ opticks-rmdirs- bin build gl include lib ptx externals  ; }
opticks-rmdirs-(){
   local base=$(opticks-dir)
   local msg="# $FUNCNAME : "
   echo $msg pipe to sh to do the deletion
   local name
   for name in $*
   do
      local dir=$base/$name
      [ -d "$dir" ] && echo rm -rf $dir ;
   done
}

opticks-cleanbuild()
{
   opticks-distclean
   opticks-distclean | sh
   opticks-full
}


opticks-make-()
{
    local iwd=$PWD
    local bdir=$1
    shift
    cd $bdir
    make $*
    cd $iwd
}


########## runtime setup ########################

opticks-path(){ echo $PATH | tr ":" "\n" ; }
opticks-path-add(){
  local dir=$1
  : only prepend the dir when not already there
  [ "${PATH/$dir}" == "${PATH}" ] && export PATH=$dir:$PATH
}


opticks-llp-(){
    local prefix=${1:-$(opticks-prefix)}
    cat << EOL
$prefix/lib
$prefix/lib64
$prefix/externals/lib
$prefix/externals/lib64
$prefix/externals/optix/lib64
EOL
}
opticks-join(){ local ifs=$IFS ; IFS="$1"; shift; echo "$*" ; IFS=$ifs ;  }
opticks-llp(){  opticks-join : $($FUNCNAME- $*) ; }


opticks-dump(){ cat << EOD

Standardize all paths to be "physical" as given by "pwd -P"
for om-cd toggling between source and build trees to work

This also seems to make building Opticks significantly faster.

   G                          : $G
   PYTHONPATH                 : $PYTHONPATH
   ENV_HOME                   : $ENV_HOME
   OPTICKS_HOME               : $OPTICKS_HOME
   TMP                        : $TMP
   LOCAL_BASE                 : $LOCAL_BASE
   OPTICKS_RESULTS_PREFIX     : $OPTICKS_RESULTS_PREFIX


   OPTICKS_PREFIX             : $OPTICKS_PREFIX
   OPTICKS_OPTIX_PREFIX       : $OPTICKS_OPTIX_PREFIX
   OPTICKS_CUDA_PREFIX        : $OPTICKS_CUDA_PREFIX

   OPTICKS_COMPUTE_CAPABILITY : $OPTICKS_COMPUTE_CAPABILITY
   OPTICKS_KEY                : $OPTICKS_KEY

EOD

   echo $PATH | tr ":" "\n" ;
}



opticks-export()
{
   opticks-path-add $(opticks-prefix)/lib
   opticks-path-add $(opticks-prefix)/bin
   opticks-path-add $(opticks-home)/bin
   opticks-path-add $(opticks-home)/ana

   opticksdata-
   opticksdata-export
}


opticks-okdist-dirlabel-notes(){ cat << EON
opticks-okdist-dirlabel-notes
-------------------------------

    opticks-okdist-dirlabel : $(opticks-okdist-dirlabel)

Examples::

   x86_64-centos7-gcc48-geant4_10_04_p02-dbg

The label is used by okdist- for naming directories that contain
Opticks binary distributions.

Note that the below versions are not included in this directory label as
they are encompassed by the Opticks version.

* OptiX version
* CUDA Version
* NVIDIA Driver Version


CUDA is treated separately and lib access is from LD_LIBRARY_PATH
so perhaps it belongs in the name ?

junosw releases::

    /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.2.3/


HMM better to start from junosw pattern, but with cuda version::

    /cvmfs/opticks.ihep.ac.cn/el9_amd64_gcc11_cuda12.4/Release/v0.0.1/


EON
}

opticks-okdist-mode(){ echo dbg ; }
opticks-okdist-dirlabel(){
    #g4-
    #local label=$(arch)-$(opticks-os-release)-$(opticks-compiler-version)-$(g4-nom)-$(opticks-okdist-mode)
    local label=$(opticks-os-release)_$(opticks-os-arch)_$(opticks-compiler-version)
    local ulabel=${label//\//}
    : remove all slashes from the label
    echo $ulabel
}

opticks-compiler-version(){  echo gcc$(opticks-gcc-version) ; }
opticks-gcc-version(){ gcc -dumpversion | perl -pe 's/\.//g' - ; } # 4.2.1 clang compatibility with gcc ?

opticks-os-arch(){
   case $(uname -m) in
     x86_64) echo amd64 ;;
   esac
}

opticks-os-release(){
   local v=$(opticks-redhat-release) ;
   case $v in
     7.?) echo el7 ;;
     9.?) echo el9 ;;
   esac
}
opticks-redhat-release(){  cat /etc/redhat-release | sed  's/.*\s\([0-9]*\.[0-9]*\).*/\1/' - ; }
opticks-redhat-release-alt(){ cat /etc/redhat-release|tr -cd "[0-9][.]" ; } # -cd:complement set and delete not translate



########### bitbucket commits

opticks-co(){      opticks-open  https://bitbucket.org/simoncblyth/opticks/commits/all ; }



########## building opticks docs

opticks-bb(){      opticks-open  http://simoncblyth.bitbucket.io/opticks/index.html ; }

opticks-docs-page(){     echo ${P:-index.html} ; }
#opticks-docs-page(){     echo ${P:-docs/orientation.html} ; }
#opticks-docs-page(){     echo ${P:-docs/debug.html} ; }

opticks-docs-vi(){       local page=$(opticks-docs-page) ; vi $(opticks-home)/${page/.html/.rst} ; }
opticks-docs-remote(){   opticks-open  http://simoncblyth.bitbucket.io/opticks/$(opticks-docs-page) ; }
opticks-docs-local(){    opticks-open  http://localhost/opticks/$(opticks-docs-page) ; }

opticks-notes-remote(){  opticks-open  http://simoncblyth.bitbucket.io/opticks_notes/index.html ; }
opticks-docs(){    opticks-open  $(opticks-docs-htmldir)/index.html ; }
opticks-docs-htmldir(){
   local htmldirbb=$HOME/simoncblyth.bitbucket.io/opticks
   [ -d "$htmldirbb" ] && echo $htmldirbb || echo $(opticks-prefix)/html
}




opticks-docs-make-info(){ cat << EOI

$FUNCNAME
========================

opticks-docs-htmldir : $(opticks-docs-htmldir)

EOI
}


opticks-docs-make()
{
   local iwd=$PWD
   opticks-scd
   sphinx-build -b html  . $(opticks-docs-htmldir)
   cd $iwd

   #opticks-docs
   opticks-docs-local

   echo $msg publish the docs via bitbucket commit/push from $(opticks-docs-htmldir)

}

########## building opticks dev notes




opticks-issues-cd(){  cd $(opticks-home)/notes/issues/$1 ; }
opticks-notes-cd(){ cd $(opticks-home)/notes/$1 ; }
opticks-refs-cd(){  cd $HOME/opticks_refs ; }
opticks-refs-ls(){  opticks-refs-cd ; ls -lt *.pdf | head -30 ; }
or(){
   : ~/opticks/opticks.bash
   opticks-refs-ls ;
}
or1(){
   : ~/opticks/opticks.bash
   opticks-refs-ls
   open $(ls -1t *.pdf | head -1) ;
}



opticks-notes-image(){
    find $(opticks-home)/notes -name '*.rst' -exec grep -H image {} \;
}


opticks-notes-notes(){ cat << EON

Planted a top level link::

   simon:/ blyth$ sudo ln -s /Users/blyth/simoncblyth.bitbucket.org/env

To allow default file links like the below to resolve from
local html::

   file:///env/graphics/ggeoview/issues/offset_bottom/dpib.png


EON
}

opticks-notes(){ opticks-open  $(opticks-notes-htmldir)/index.html ; }
opticks-notes-htmldir(){
   local htmldirbb=$HOME/simoncblyth.bitbucket.org/opticks_notes
   [ -d "$htmldirbb" ] && echo $htmldirbb || echo $(opticks-prefix)/notes/html
}
opticks-notes-make()
{
   local iwd=$PWD
   opticks-scd "notes"
   sphinx-build -b html  . $(opticks-notes-htmldir)
   cd $iwd
   opticks-notes
}

##############



opticks-open()
{
  local url=$1
  case $(uname) in
      Linux) firefox $url ;;
     Darwin) open $url ;;
      MING*) chrome $url ;;
  esac
}




## [WIP] modern CMake proj-by-proj style building

om-(){       . $(opticks-home)/om.bash      && om-env $* ; }
om(){  om- ; om-- $* ; }
omt(){ om- ; om-test $* ; }

oe-(){       . $(opticks-home)/oe.bash      && oe-env $* ; }
oe(){        oe- 2>/dev/null ; }
oc-(){       . $(opticks-home)/bin/oc.bash  && oc-env $* ; }
opnovice-(){ . $(opticks-home)/notes/geant4/opnovice.bash      && opnovice-env $* ; }



### opticks CMake projects all residing in top level folders ##

okconf-(){          . $(opticks-home)/okconf/okconf.bash && okconf-env $* ; }
sysrap-(){          . $(opticks-home)/sysrap/sysrap.bash && sysrap-env $* ; }

#brap-(){            . $(opticks-home)/boostrap/brap.bash && brap-env $* ; }
#npy-(){             . $(opticks-home)/npy/npy.bash && npy-env $* ; }
#okc-(){             . $(opticks-home)/optickscore/okc.bash && okc-env $* ; }

#ggeo-(){            . $(opticks-home)/ggeo/ggeo.bash && ggeo-env $* ; }
#asirap-(){          . $(opticks-home)/assimprap/asirap.bash && asirap-env $* ; }
#openmeshrap-(){     . $(opticks-home)/openmeshrap/openmeshrap.bash && openmeshrap-env $* ; }
#okg-(){             . $(opticks-home)/opticksgeo/okg.bash && okg-env $* ; }

#oglrap-(){          . $(opticks-home)/oglrap/oglrap.bash && oglrap-env $* ; }
#cudarap-(){         . $(opticks-home)/cudarap/cudarap.bash && cudarap-env $* ; }

qudarap-(){         . $(opticks-home)/qudarap/qudarap.bash && qudarap-env $* ; }

#thrap-(){           . $(opticks-home)/thrustrap/thrap.bash && thrap-env $* ; }
#oxrap-(){           . $(opticks-home)/optixrap/oxrap.bash && oxrap-env $* ; }

#okop-(){            . $(opticks-home)/okop/okop.bash && okop-env $* ; }
#okgl-(){            . $(opticks-home)/opticksgl/okgl.bash && okgl-env $* ; }
#ok-(){              . $(opticks-home)/ok/ok.bash && ok-env $* ; }
#cfg4-(){            . $(opticks-home)/cfg4/cfg4.bash && cfg4-env $* ; }
#okg4-(){            . $(opticks-home)/okg4/okg4.bash && okg4-env $* ; }

#g4ok-(){            . $(opticks-home)/g4ok/g4ok.bash && g4ok-env $* ; }
#x4-(){              . $(opticks-home)/extg4/x4.bash  && x4-env $* ; }
#c4-(){              . $(opticks-home)/c4/c4.bash  && c4-env $* ; }
#x4gen-(){           . $(opticks-home)/extg4/x4gen.bash  && x4gen-env $* ; }
#yog-(){             . $(opticks-home)/yoctoglrap/yog.bash && yog-env $* ; }

bin-(){             . $(opticks-home)/bin/bin.bash && bin-env $* ; }
integration-(){     . $(opticks-home)/integration/integration.bash && integration-env $* ; }



okconf(){ okconf- ; okconf-cd $* ; }
sysrap(){ sysrap- ; sysrap-cd $* ; }
brap(){   brap-;    brap-cd $* ; }
npy(){    npy- ;    npy-cd $* ; }
okc(){    okc-;     okc-cd $* ; }

ggeo(){        ggeo-;        ggeo-cd $* ; }
asirap(){      asirap-;      asirap-cd $* ; }
openmeshrap(){ openmeshrap-; openmeshrap-cd $* ; }
okg(){         okg-;         okg-cd $* ; }

oglrap(){   oglrap-  ; oglrap-cd $* ; }
cudarap(){  cudarap- ; cudarap-cd $* ; }
qudarap(){  qudarap- ; qudarap-cd $* ; }

pwd_(){   [ -z "$QUIET" ] && pwd  ; }
pwdt_(){  [ -z "$QUIET" ] && pwd && ls -l *.sh ; }


# optix7 expts
c(){  cd $(opticks-home)/CSG ; pwd_ ; }
ct(){ cd $(opticks-home)/CSG/tests ; pwdt_ ; }
cg(){ cd $(opticks-home)/CSG_GGeo ; pwd_ ; }
gc(){ cd $(opticks-home)/GeoChain ; pwd_ ; }
cx(){ cd $(opticks-home)/CSGOptiX ; pwd_ ; }
cxt(){ cd $(opticks-home)/CSGOptiX/tests ; pwdt_ ; }
u4(){ cd $(opticks-home)/u4 ; pwd_ ; }
u4t(){ cd $(opticks-home)/u4/tests ; pwdt_ ; }
#c4(){ cd $(opticks-home)/c4 ; pwd_ ; }
c4(){ cd ~/customgeant4 ; git status ; }

gx(){ cd $(opticks-home)/g4cx ; pwd_ ; }
gxt(){ cd $(opticks-home)/g4cx/tests ; pwdt_ ;  }
gd(){ cd $(opticks-home)/gdxml ; pwd_ ; }
gdt(){ cd $(opticks-home)/gdxml/tests ; pwdt_ ; }

qu(){ qudarap $* ; pwd_ ; }
qt(){ qudarap tests ; pwdt_ ; }
sy(){ sysrap $* ; pwd_ ; }
y(){  sysrap $* ; pwd_ ; }
st(){ sysrap tests ; pwdt_ ; }



thrap(){    thrap-   ; thrap-cd $* ; }
oxrap(){    oxrap-   ; oxrap-cd $* ; }

okop(){     okop- ; okop-cd $* ; }
okgl(){     okgl- ; okgl-cd $* ; }
ok(){       ok- ; ok-cd $* ; }
cfg4(){     cfg4- ; cfg4-cd $* ; }
okg4(){     okg4- ; okg4-cd $* ; }

g4ok(){     g4ok- ; g4ok-cd $* ; }
x4(){       x4- ; x4-cd $* ; }
x4t(){      x4- ; x4-cd tests ; }
x4gen(){    x4gen- ; x4gen-cd $* ; }
yog(){      yog- ; yog-cd $* ; }

bin(){          bin- ; bin-cd $* ; }
integration(){  integration- ; integration-cd $* ; }




## opticks misc including python analysis/debugging ##

ana-(){             . $(opticks-home)/ana/ana.bash && ana-env $*  ; }
a(){     cd $(opticks-home)/ana ; pwd ; }
cfh-(){             . $(opticks-home)/ana/cfh.bash && cfh-env $*  ; }
tests-(){           . $(opticks-home)/tests/tests.bash && tests-env $*  ; }
tools-(){           . $(opticks-home)/tools/tools.bash && tools-env $*  ; }
notes-(){           . $(opticks-home)/notes/notes.bash && notes-env $*  ; }
pmt-(){             . $(opticks-home)/ana/pmt/pmt.bash && pmt-env $* ; }
ab-(){              . $(opticks-home)/bin/ab.bash      && ab-env $* ; }
abe-(){             . $(opticks-home)/bin/abe.bash     && abe-env $* ; }
ev-(){              . $(opticks-home)/bin/ev.bash      && ev-env $* ; }
scan-(){            . $(opticks-home)/bin/scan.bash    && scan-env $* ; }
hh-(){              . $(opticks-home)/bin/hh.bash      && hh-env $* ; }
vbx-(){             . $(opticks-home)/bin/vbx.bash     && vbx-env $* ; }
ptx-(){             . $(opticks-home)/bin/ptx.bash     && ptx-env $* ; }
ezgdml-(){          . $(opticks-home)/bin/ezgdml.bash  && ezgdml-env $* ; }
odocker-(){         . $(opticks-home)/bin/odocker.bash && odocker-env $* ; }
olxd-(){            . $(opticks-home)/bin/olxd.bash    && olxd-env $* ; }
onvidia-(){         . $(opticks-home)/bin/onvidia.bash && onvidia-env $* ; }
nsight-(){          . $(opticks-home)/bin/nsight.bash  && nsight-env $* ; }

# override old original from env, $(env-home)/nuwa/detdesc/pmt/pmt.bash

### opticks g4 examples ########
g4x-(){             . $(opticks-home)/examples/g4x.bash && g4x-env $* ; }

### opticks launchers ########

okr-(){             . $(opticks-home)/bin/okr.bash && okr-env $* ; }
okdist-(){          . $(opticks-home)/bin/okdist.bash && okdist-env $* ; }
scdist-(){          . $(opticks-home)/bin/scdist.bash && scdist-env $* ; }
oks-(){             . $(opticks-home)/bin/oks.bash && oks-env $* ; }
winimportlib-(){    . $(opticks-home)/bin/winimportlib.bash && winimportlib-env $* ; }
ggv-(){             . $(opticks-home)/bin/ggv.bash && ggv-env $* ; }
vids-(){            . $(opticks-home)/bin/vids.bash && vids-env $* ; }
op-(){              . $(opticks-home)/bin/op.sh ; }
fn-(){              . $(opticks-home)/bin/fn.bash && fn-env $* ; }

#### opticks top level tests ########





geocache-(){      . $(opticks-home)/ana/geocache.bash  && geocache-env $* ; }
ckm-(){           . $(opticks-home)/bin/ckm.bash  && ckm-env $* ; }
ckm(){            ckm- ; ckm-cd $* ; }
cks(){            cd $(opticks-home)/examples/Geant4/CerenkovStandalone ; }

####### below functions support analysis on machines without a full opticks install
####### by copying some parts of an opticks install to corresponding local locations

opticks-host(){ echo ${OPTICKS_HOST:-192.168.1.101} ; }
opticks-user(){ echo ${OPTICKS_USER:-$USER} ; }

opticks-scp(){
   local msg="=== $FUNCNAME : "
   local host=$(opticks-host)
   local user=$(opticks-user)
   local src=$1
   local dst=$2
   mkdir -p $(dirname $dst)

   [ -d "$dst" ] && echo $msg dst $dst exists already && return

   local cmd="scp -r $user@$host:$src $(dirname $dst)"
   echo $msg \"$cmd\"

   local ans
   read -p "$msg Proceed with the above command ? [Yy] "  ans
   if [ "$ans" == "Y" -o "$ans" == "y" ]; then
       echo $msg OK PROCEEDING
       eval $cmd
   else
       echo $msg OK SKIP
   fi
}


opticks-prefix-ref(){ echo ${OPTICKS_PREFIX_REF:-/home/simonblyth/local/opticks} ; }
opticks-prefix-loc(){ echo  $HOME/local/opticks ; }

opticks-okc-ref(){    echo $(opticks-prefix-ref)/installcache/OKC ; }
opticks-okc-loc(){    echo $(opticks-prefix-loc)/installcache/OKC ; }

opticks-geo-ref(){ echo $(opticks-prefix-ref)/opticksdata/export/$(opticks-geopath-ref) ; }
opticks-geo-loc(){ echo $(opticks-prefix-loc)/opticksdata/export/$(opticks-geopath-ref) ; }

opticks-geopath-ref(){ echo DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae ; }


opticks-geo-get(){
   local user=${1:-$USER}
   [ ! -d "$(opticks-prefix-loc)/opticksdata" ] && echo YOU NEED TO optickdata-get FIRST && return 1

   opticks-scp $(opticks-geo-ref) $(opticks-geo-loc)
   opticks-scp $(opticks-okc-ref) $(opticks-okc-loc)
}

opticks-evt-ref(){ echo ${OPTICKS_EVT_REF:-/tmp/simonblyth/opticks/evt} ; }
opticks-evt-loc(){ echo /tmp/$USER/opticks/evt ; }
opticks-evt-get()
{
    local subd=${1:-reflect}
    opticks-scp $(opticks-evt-ref)/$subd $(opticks-evt-loc)/$subd
}


opticks-analysis-only-setup()
{
   opticksdata-
   opticksdata-get

   opticks-geo-get

   opticks-evt-get rainbow
}

opticks-analysis-only-check()
{
   cat << EOC

   PATH : $PATH

       This needs to include directories where the tools are installed, eg

           hg (Mercurial)
           python
           ipython

   LOCAL_BASE : $LOCAL_BASE

       Should be \$HOME/local ie $HOME/local
       This is used by opticksdata- functions to
       identify the location for the opticksdata repository clone

   PYTHONPATH : $PYTHONPATH

       Should be same as \$HOME ie $HOME
       This allows you to :
           python -c "import opticks"

EOC

}


opticks-lib-ext()
{
   case $(uname) in
     Linux) echo so ;;
     Darwin) echo dylib ;;
   esac
}


opticks-lib-ls()
{
    local ext=$(opticks-lib-ext)

    cd $(opticks-prefix)/lib
    ls -1 *.$ext | perl -pe "s/.$ext//" - | perl -pe "s/^lib//" -



}


opticks-find-types()
{
    opticks-find Types.hpp
}

opticks-find-typ()
{
    opticks-find Typ.hpp
}

opticks-find-flags()
{
    opticks-find OpticksFlags.hh
}




opticks-cls ()
{
    opticks-cls- "." $*
}
opticks-cls-()
{
    local iwd=$PWD;
    opticks-scd;
    local base=${1:-.};
    local name=${2:-DsG4OpBoundaryProcess};
    local h=$(find $base -name "$name.h");
    local hh=$(find $base -name "$name.hh");
    local hpp=$(find $base -name "$name.hpp");
    local cc=$(find $base -name "$name.cc");
    local cpp=$(find $base -name "$name.cpp");
    local icc=$(find $base -name "$name.icc");
    local tcc=$(find $base -name "${name}Test.cc");
    local vcmd="vi  $h $hh $hpp $icc $cpp $cc $tcc";
    echo $vcmd;
    eval $vcmd;
    cd $iwd
}


opticks-cmake-projs-vi-(){    ls -1  $(opticks-home)/CMakeLists.txt $(opticks-home)/*/CMakeLists.txt ; }
opticks-cmake-examples-vi-(){ ls -1 $(opticks-home)/examples/*/CMakeLists.txt ; }
opticks-cmake-tests-vi-(){    ls -1 $(opticks-home)/*/tests/CMakeLists.txt ; }

opticks-cmake-vi-()
{
    opticks-cmake-projs-vi-
    opticks-cmake-examples-vi-
    opticks-cmake-tests-vi-
}

opticks-cmake-projs-vi(){     vi $($FUNCNAME-) ; }
opticks-cmake-examples-vi(){  vi $($FUNCNAME-) ; }
opticks-cmake-tests-vi(){     vi $($FUNCNAME-) ; }
opticks-cmake-vi(){           vi $($FUNCNAME-) ; }




opticks-h()
{
   local msg="=== $FUNCNAME $(pwd) :"

   echo $msg header includes
   grep -h ^#include *.hh *.hpp 2>/dev/null | sort | uniq

   #echo $msg implementation includes
   #grep -h ^#include *.cc *.cpp 2>/dev/null | sort | uniq
}


opticks-rpath(){ grep RPATH $(opticks-home)/examples/*/CMakeLists.txt ; }

opticks-bcm-deploy-(){  find . -name CMakeLists.txt -exec grep -l bcm_deploy {} \; ; }
opticks-bcm-deploy(){ vi $(opticks-bcm-deploy-) ; }

opticks-deps(){   $(opticks-home)/bin/CMakeLists.py $* ; }
opticks-deps-vi(){ vi $(opticks-home)/bin/CMakeLists.py ; }

opticks-executables(){  find . -type f -perm +111 -print | grep -v dylib | grep -v a.out | grep -v .bin | grep -v .cmake | grep -v opticks-config ; }




opticks-cmake-check(){
   opticks-scd
   find . -name CMakeLists.txt -exec grep -l OpticksBuildOptions {} \; | wc -l
   find . -name CMakeLists.txt | wc -l

   ## see bin/CMakeLists.py for easier way of making such consistency checks

}

opticks-pdoc(){ o ; vi okop/OKOP.rst opticksgeo/OKGEO.rst optixrap/OXRAP.rst thrustrap/THRAP.rst g4ok/G4OK.rst ; }



opticks-linecount(){

   opticks-scd
   find . -path ./.hg -prune -o -name '*.*'  | xargs wc -l

}

opticks-installed-headers(){
   find $OPTICKS_PREFIX/include \( -name '*.h' -or -name '*.hh' -or -name '*.hpp' \) -exec ${*:-echo} {} \;
}



opticks-src(){      echo https://bitbucket.org/simoncblyth/opticks/src/master ; }
opticks-src-rel(){  echo ${1:-notes/tasks/tasks.rst}  ; }
opticks-src-url(){  echo $(opticks-src)/$(opticks-src-rel $*) ; }
opticks-src-path(){ echo $(opticks-home)/$(opticks-src-rel $*) ; }
opticks-src-open(){
   echo open $(opticks-src-url $*)
   echo vi $(opticks-src-path $*)
   open $(opticks-src-url $*)
   vi $(opticks-src-path $*)
}

opticks-tasks(){    opticks-src-open notes/tasks/tasks.rst ; }
opticks-progress(){ opticks-src-open notes/progress.rst ; }
opticks-examples(){ opticks-src-open examples/README.rst ; }


opticks-u(){
   : open the bitbucket src url corresponding to the current directory or path argument within the opticks repository
   local msg="=== $FUNCNAME :"
   local arg=${1:-$PWD}
   local path
   if [ "${arg:0:1}" == "/" ]; then
       path=$arg
   else
       path=$PWD/$arg
   fi
   local rel=${path/$(opticks-home)\/}
   local url=$(opticks-src)/$rel
   echo $msg path $path rel $rel url $url
   open $url
}


opticks-set-optix-prefix()
{
    local ver=${1:-6}
    [ "$(uname)" == "Darwin" ] && ver=5

    local opticks_optix_prefix=$OPTICKS_OPTIX_PREFIX
    case $ver in
        5) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX5_PREFIX} ;;
        6) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX6_PREFIX} ;;
        7) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX7_PREFIX} ;;
        *) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX6_PREFIX} ;;
    esac

    if [ "$OPTICKS_OPTIX_PREFIX" == "$opticks_optix_prefix" ]; then
        echo $msg OPTICKS_OPTIX_PREFIX $OPTICKS_OPTIX_PREFIX : is unchanged
    else
        echo $msg OPTICKS_OPTIX_PREFIX $OPTICKS_OPTIX_PREFIX : is changed from $opticks_optix_prefix
    fi
}

opticks-chkvar()
{
    local msg="=== $FUNCNAME :"
    local var ;
    for var in $* ; do
        if [ -z "${!var}" -o ! -d "${!var}" ]; then
            echo $msg missing required envvar $var ${!var} OR non-existing directory
            return 1
        fi
        printf "%20s : %s \n" $var ${!var}
    done
    return 0
}


opticks-build7-notes(){ cat << EON
opticks-build7-notes
======================

NB opticks-build7 or b7 shortcut is currently restricted to the CSGOptiX package.

Builds CSGOptiX with environment modified to use NVIDIA OptiX 7
rather then the default of NVIDIA OptiX 6

Switching between OptiX versions is achieved by setting the
OPTICKS_OPTIX_PREFIX envvar which is used for example by
cmake/Modules/FindOpticksOptiX.cmake

Attempting to discern the OPTIX_VERSION by sourcing a
buildenv.sh script generated at config time is an
inherently flawed approach. Are now using the CSGOptiXVersion
executable that is built and installed together with the library,
so can get the version in scripts by capturing the output from that executable.

Although what opticks-build7 does looks exceedingly similar to what om would do,
there is a crucial difference in that **the om environment setup is not invoked**.

This function avoids a problem with using scripts to do similar because using
scripts forces the jre environment setup to be repeated in the script (or done in
.bash_profile/.bashrc)  but as the environment setup is installation specific
it should not be in the opticks repository and doing in login scripts feels dirty.
Doing in a function is equivalent to sourcing, hence the existing environment is reused.

EON
}

opticks-build7()
{
    export OPTICKS_HOME=$(opticks-home)  ## TODO: include OPTICKS_HOME in the opticks-setup
    opticks-set-optix-prefix 7

    opticks-chkvar OPTICKS_PREFIX OPTICKS_HOME OPTICKS_OPTIX_PREFIX
    [ $? -ne 0 ] && echo $msg checkvar FAIL && return 1

    cd $(opticks-home)/CSGOptiX
    local sdir=$(pwd)
    local name=$(basename $sdir)

    local bdir=/tmp/$USER/opticks/${name}.build
    rm -rf $bdir && mkdir -p $bdir
    [ ! -d $bdir ] && exit 1
    cd $bdir && pwd

    cmake $sdir \
         -DCMAKE_BUILD_TYPE=Debug \
         -DOPTICKS_PREFIX=${OPTICKS_PREFIX} \
         -DCMAKE_MODULE_PATH=${OPTICKS_HOME}/cmake/Modules \
         -DCMAKE_INSTALL_PREFIX=${OPTICKS_PREFIX}

    [ $? -ne 0 ] && echo $msg conf FAIL && return 1

    make
    [ $? -ne 0 ] && echo $msg make FAIL && return 2

    make install
    [ $? -ne 0 ] && echo $msg install FAIL && return 3

    opticks-set-optix-prefix 6
    cd $sdir && pwd
    return 0
}


opticks-geompath()
{
    : uses the first of dotpath or homepath
    local dotpath=$HOME/.opticks/GEOM.txt
    local homepath=$(opticks-home)/GEOM.txt
    local path=""
    if [ -f "$dotpath" ]; then
        path=$dotpath
    elif [ -f "$homepath" ]; then
        path=$homepath
    fi
    echo $path
}

opticks-geom()
{
    local geompath=$(opticks-geompath)
    local geom=""
    if [ -n "$geompath" ]; then
        local catgeom=$(cat $geompath 2>/dev/null | grep -v \#) && [ -n "$catgeom" ] && geom=$(echo ${catgeom%%_*})
    fi
    echo $geom
}

opticks-geom-(){
    local msg="=== $FUNCNAME :"
    local geompath=$(opticks-geompath)
    echo $msg geompath $geompath
    vi $geompath
}


opticks-hookup(){ source $OPTICKS_HOME/bin/geocache_hookup.sh ${1:-last} ; }


opticks-key-remote(){ echo $OPTICKS_KEY_REMOTE ; }
opticks-key-remote-dir(){
   local msg="=== $FUNCNAME "
   [ -z "$OPTICKS_KEY_REMOTE" ] && echo $msg missing required envvar OPTICKS_KEY_REMOTE && return 1
   local opticks_key_remote_dir=$(OPTICKS_KEY=$(opticks-key-remote) OPTICKS_GEOCACHE_PREFIX=.opticks SOpticksResourceTest --keydir)
   echo $opticks_key_remote_dir
}

opticks-key-remote-dir-cd(){
   local krd=$(opticks-key-remote-dir)
   cd $HOME/$krd/CSG_GGeo/CSGFoundry
   pwd
}
krcd(){ opticks-key-remote-dir-cd ; }


opticks-switch-last(){
    local opticks_geocache_prefix=$HOME/.opticks
    local geocache_sh=${OPTICKS_GEOCACHE_PREFIX:-$opticks_geocache_prefix}/geocache/geocache.sh
    source $geocache_sh
}
opticks-switch-key(){
    : hmm this is doing similar to ~/opticks/bin/geocache_hookup.sh but much more succinctly
    local arg=$1
    case $arg in
       remote) export OPTICKS_KEY=$OPTICKS_KEY_REMOTE ; export OPTICKS_GEOCACHE_PREFIX=$HOME/.opticks  ;;
          old) export OPTICKS_KEY=$OPTICKS_KEY_OLD  ;;
          new) export OPTICKS_KEY=$OPTICKS_KEY_NEW  ;;
         last) opticks-switch-last ;;
         asis)  echo -m ;;
    esac
}



opticks-cfbase-(){ SOpticksResourceTest --cfbase ; }
opticks-cfbase(){ cd $(opticks-cfbase-) ; }

opticks-cf-(){  echo $(opticks-cfbase-)/CSGFoundry ; }
opticks-cf(){   cd $(opticks-cf-) ; }




dbg__()
{
   : opticks/opticks.bash
   case $(uname) in
      Darwin) lldb__ $* ;;
      Linux) gdb__ $* ;;
   esac
}

gdb__ ()
{
    : opticks/opticks.bash prepares and invokes gdb - sets up breakpoints based on BP envvar containing space delimited symbols;
    if [ -z "$BP" ]; then
        H="";
        B="";
        T="-ex r";
    else
        H="-ex \"set breakpoint pending on\"";
        B="";
        for bp in $BP;
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" -ex r";
    fi;
    local runline="gdb $H $B $T --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}


opticks-ptx-(){ echo $(opticks-prefix)/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx ; }
opticks-ptx-head(){   head -19 $(opticks-ptx-) ; }

opticks-ptx(){

   local ptx=$(opticks-ptx-)
   if [ ! -f "$ptx" ];  then
       ptx=/tmp/CSGOptiX_generated_CSGOptiX7.cu.ptx
   fi

   local num_printf_lines=$(grep printf $ptx | wc -l )
   local num_f64_lines=$(grep \\.f64 $ptx | wc -l )

   vars="BASH_SOURCE FUNCNAME ptx num_printf_lines num_f64_lines"
   for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done

}



opticks-prefix-find-stale()
{
   local iwd=$PWD
   cd $OPTICKS_PREFIX
   pwd
   local tops="lib lib64"
   local ref=lib/OKConfTest
   for top in $tops ; do
       echo $FUNCNAME find files in $top 10 minutes or more older than ref $ref 
       find $top -type f ! -newermt "$(date -r $ref) - 10 min"
   done
   cd $iwd
}


