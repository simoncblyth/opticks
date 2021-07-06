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
opticksaux-jv5-vi(){ vim -R $(opticksaux-jv5) ; }
opticksaux-jv5-cd(){ cd $(dirname $(opticksaux-jv5)) ; }

opticksaux-j21-version(){ echo v2 ; }
opticksaux-j21(){ echo $(opticksaux-dir)/$(opticksaux-xpath j2102)_$(opticksaux-j21-version).gdml ; }  
opticksaux-j21-vi(){ vim -R $(opticksaux-j21) ; }
opticksaux-j21-cd(){ cd $(dirname $(opticksaux-j21)) ; }


opticksaux-j21-notes(){ cat << EON
see ~/jnu/issues/geocache-j2102-shakedown.rst  
EON
}
opticksaux-j21v1(){ echo $(opticksaux-dir)/$(opticksaux-xpath j2102)_v1.gdml ; }  
opticksaux-j21v1-vi(){ vim -R $(opticksaux-j21v0) ; }
opticksaux-j21v1-cd(){ cd $(dirname $(opticksaux-j21v0)) ; }



opticksaux-dx0(){  echo $(opticksaux-dir)/$(opticksaux-xpath dybx)_v0.gdml ; }
opticksaux-dx0-vi(){ vim -R $(opticksaux-dx0) ; }
opticksaux-dx0-cd(){ cd $(dirname $(opticksaux-dx0)) ; }

opticksaux-dx1(){  echo $(opticksaux-dir)/$(opticksaux-xpath dybx)_v1.gdml ; }
opticksaux-dx1-vi(){ vim -R $(opticksaux-dx1) ; }
opticksaux-dx1-cd(){ cd $(dirname $(opticksaux-dx1)) ; }

opticksaux-dx1-notes(){ cat << EON
Manually copied dx0 to dx1 and added sensor surfaces, to generate some DYB sensor 
hits in order to test hit formation within a smaller geometry than JUNO.
EON
}


opticksaux-xpath(){
   case $1 in 
       j2102) echo export/juno2102/tds_ngt_pcnk_sycg_202102 ;;  
       j1808) echo export/juno1808/g4_00 ;;  
       dybx)  echo export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export ;;
   esac 
}

opticksaux-info(){ cat << EOI

   opticksaux-url       : $(opticksaux-url)
   opticksaux-url-ssh   : $(opticksaux-url-ssh)
   opticksaux-dir       : $(opticksaux-dir)

   opticksaux-jv5 :  $(opticksaux-jv5)     opticksaux-jv5-vi 
   opticksaux-dx0  : $(opticksaux-dx0)     opticksaux-dx0-vi 
   opticksaux-dx1  : $(opticksaux-dx1)     opticksaux-dx1-vi 

EOI
}

opticksaux-ls-(){
   local geoms="jv5 dx0"
   local geom
   for geom in $geoms ; do 
       local gdml=$(opticksaux-$geom)
       echo $gdml 
   done
}
opticksaux-ls(){
  local gdmls=$(opticksaux-ls-) 
  local gdml
  for gdml in $gdmls ; do echo  $gdml ; done
  for gdml in $gdmls ; do ls -l $gdml ; done
  for gdml in $gdmls ; do du -h $gdml ; done
}


opticksaux-get(){
   local msg="$FUNCNAME :"
   local iwd=$PWD
   local dir=$(dirname $(opticksaux-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(opticksaux-url)
   local nam=$(basename $url)
   nam=${nam/.git}
   if [ ! -d "$nam/.git" ]; then
        opticks-git-clone $url 
   else
        echo $msg ALREADY CLONED from $url to $(opticksaux-dir) 
        opticksaux-update
   fi
   cd $iwd
}

opticksaux-update()
{    
   local msg="$FUNCNAME :"
   opticksaux-cd
   local status=$(git status --porcelain)
   if [ -n "$status" ]; then
       echo $msg : UNEXPECTED local changes detected : you need to manually update your clone from PWD $PWD
   else
       echo $msg : no local changes proceed to pull      
       git pull
   fi 
}



opticksaux--()
{
   local msg="=== $FUNCNAME :"
   opticksaux-get
   [ $? -ne 0 ] && echo $msg get FAIL && return 1
   opticksaux-pc
   [ $? -ne 0 ] && echo $msg pc FAIL && return 2
   return 0 
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
