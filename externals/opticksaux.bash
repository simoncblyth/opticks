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

opticksaux-source(){   echo $BASH_SOURCE ; }
opticksaux-dir(){      echo $(opticks-prefix)/opticksaux ; }
opticksaux-vi(){       vi $BASH_SOURCE ; }

opticksaux-usage(){ cat << EOU
Opticks Auxiliary Data
===========================

Locations
-----------

~/opticksaux 
       opticksaux git clone used to upload geometry and other opticks data

$OPTICKS_INSTALL_PREFIX/opticksaux
       readonly opticksaux bitbucket git clone that only pulls, this is what users see



Pushing from the https clone requires password::

    [blyth@localhost opticksaux]$ git push 
    Username for 'https://bitbucket.org': simoncblyth
    Password for 'https://simoncblyth@bitbucket.org': 
    Counting objects: 7, done.
    Delta compression using up to 48 threads.
    Compressing objects: 100% (5/5), done.
    Writing objects: 100% (5/5), 221.03 KiB | 0 bytes/s, done.
    Total 5 (delta 0), reused 0 (delta 0)
    To https://bitbucket.org/simoncblyth/opticksaux.git
       995b46d..0faa056  master -> master
    [blyth@localhost opticksaux]$ 


EOU
}
opticksaux-env(){ echo -n ;  }
opticksaux-c(){   cd $(opticksaux-dir)/$1 ; }
opticksaux-cd(){  cd $(opticksaux-dir)/$1 ; }

opticksaux-url(){       echo https://bitbucket.org/simoncblyth/opticksaux.git ; }
opticksaux-url-ssh(){   echo git@bitbucket.org:simoncblyth/opticksaux.git ; }

opticksaux-jv5(){ echo $(opticksaux-dir)/$(opticksaux-xpath j1808)_v5.gdml ; }  
opticksaux-jv5-vi(){ vi $(opticksaux-jv5) ; }

opticksaux-dx-(){  echo $(opticksaux-dir)/$(opticksaux-xpath dybx) ; }
opticksaux-dx(){   echo $(opticksaux-dx-).gdml ; }
opticksaux-dx-vi(){ vi $(opticksaux-dx) ; }

opticksaux-xpath(){
   case $1 in 
       j1808) echo export/juno1808/g4_00 ;;  
       dybx)  echo export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export ;;
   esac 
}

opticksaux-info(){ cat << EOI

   opticksaux-url       : $(opticksaux-url)
   opticksaux-url-ssh   : $(opticksaux-url-ssh)
   opticksaux-dir       : $(opticksaux-dir)

   opticksaux-jv5 : $(opticksaux-jv5)     opticksaux-jv5-vi 
   opticksaux-dx  : $(opticksaux-dx)      opticksaux-dx-vi 

EOI
}

opticksaux-get(){
   local msg="$FUNCNAME :"
   local iwd=$PWD
   local dir=$(dirname $(opticksaux-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(opticksaux-url)
   local nam=$(basename $url)
   nam=${nam/.git}
   if [ ! -d "$nam/.git" ]; then
        local cmd="git clone $url "
        echo $msg proceeding with \"$cmd\" from $dir 
        eval $cmd
   else
        echo $msg ALREADY CLONED from $url to $(opticksaux-dir) 
   fi
   cd $iwd
}

opticksaux--()
{
   opticksaux-get
   opticksaux-pc
}

opticksaux-pc(){ echo $FUNCNAME placeholder ; }

opticksaux-pull()
{
   local msg="$FUNCNAME :"
   local iwd=$PWD
   opticksaux-cd
   echo $msg PWD $PWD
   git pull    # equivalent to : hg pull + hg up 
   cd $iwd  
}


opticksaux-setup(){ cat << EOS
# $FUNCNAME
EOS
}
