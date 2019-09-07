#!/usr/bin/env bash
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


t-source(){   echo $BASH_SOURCE ; }
t-vi(){       vi $(t-source) ; }
t-usage(){ cat << \EOU

t- 
======================================================

Aim to pull out the generally useful dispatch stuff
from tboolean to a neutral location between test geometry 
and full geometry 

Objective is to get something like the opticks-tboolean-shortcuts ts/tv/ta
to work in general for full geometry too.


EOU
}

t-env(){ olocal- ;  }
t-dir(){ echo $(opticks-home)/tests ; }
t-cd(){  cd $(t-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

t-ip-(){  ipython -i --pdb -- $(which $(t-script)) --det $(t-det) --pfx $(t-pfx) --tag="$(t-tag)" $* ; }
t-py-(){                              $(t-script)  --det $(t-det) --pfx $(t-pfx) --tag="$(t-tag)" $* ; }

t-script(){ echo tokg4.py ; }

t-det(){ echo g4live ; }
t-pfx(){ echo source ; }
t-tag(){ echo ${TAG:-1} ; }

t-cat(){ echo g4live ; }


t--()
{
   local msg="=== $FUNCNAME :"
   local RC
   echo $msg 

   local cmdline="$*"

   if [ "${cmdline/--ip}" != "${cmdline}" ]; then
       t-ip- $* 
   elif [ "${cmdline/--py}" != "${cmdline}" ]; then
       t-py- $* 
   elif [ "${cmdline/--noalign}" != "${cmdline}" ]; then
       t-o- --okg4  $*   
       RC=$?
   else
       t-o- --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero $*   
       RC=$?
   fi 
   echo $msg $funcname RC $RC
   return $RC
}

t-o-(){

    local msg="=== $FUNCNAME :"
    local cmdline=$*
    local stack=2180  # default

    o.sh  \
            $cmdline \
            --envkey \
            --rendermode +global,+axis \
            --geocenter \
            --stack $stack \
            --eye 1,0,0 \
            --torch \
            --torchdbg \
            --tag $(t-tag) \
            --pfx $(t-pfx) \
            --cat $(t-cat) \
            --anakey $(t-anakey) \
            --args \
            --save 


    RC=$?
    echo $FUNCNAME RC $RC

    cat << EON > /dev/null
            --dbganalytic \
            --dbgemit \
            --dumpenv \
            --strace \
            --args \
            --timemax $tmax \
            --animtimemax $tmax \
            --torchconfig "$(t-torchconfig)" \


EON

    return $RC 
}




