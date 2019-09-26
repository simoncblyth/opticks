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

okdist-source(){ echo $BASH_SOURCE ; }
okdist-sdir(){ echo $(dirname $(okdist-source)) ; }
okdist-py(){ echo $(okdist-sdir)/okdist.py ; }
okdist-vi(){ vi $(okdist-source) $(okdist-py) $(opticks-home)/notes/issues/packaging-opticks-and-externals-for-use-on-gpu-cluster.rst ; }
okdist-env(){ echo -n ; }
okdist-usage(){  cat << \EOU

Opticks Binary Distribution : create tarball for explosion on cvmfs 
=====================================================================

Dev Notes
----------

* notes/issues/packaging-opticks-and-externals-for-use-on-gpu-cluster.rst


cvmfs layout
----------------

::

    /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks/0.0.0-alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/


workflow for Opticks binary releases
----------------------------------------









EOU
}

okdist-tmp(){     echo /tmp/$USER/opticks/okdist-test ; }
okdist-cd(){      cd $(okdist-tmp) ; }

okdist-releases-dir-default(){ echo /cvmfs/opticks.ihep.ac.cn/ok/releases ; }
okdist-releases-dir(){ echo ${OKDIST_RELEASES_DIR:-$(okdist-releases-dir-default)} ; } 
okdist-rcd(){ cd $(okdist-releases-dir) ; }


okdist-title(){   echo Opticks ; }
okdist-version(){ echo 0.0.0_alpha ; }
#okdist-ext(){     echo .tar.gz ; }   # slow to create and only half the size, .tar is better while testing
okdist-ext(){     echo .tar ; }  
okdist-prefix(){ echo $(okdist-title)-$(okdist-version)/$(opticks-okdist-dirlabel) ; }  
okdist-name(){   echo $(okdist-title)-$(okdist-version)$(okdist-ext) ; }
okdist-path(){   echo $(opticks-dir)/$(okdist-name) ; }    

okdist-release-prefix(){ echo $(okdist-releases-dir)/$(okdist-prefix) ; } 


okdist-info(){ cat << EOI
$FUNCNAME
=============

   okdist-ext    : $(okdist-ext)
   okdist-prefix : $(okdist-prefix)
   okdist-name   : $(okdist-name)
   okdist-path   : $(okdist-path)

   okdist-tmp    : $(okdist-tmp)

   opticks-dir   : $(opticks-dir)
       Opticks installation directory 

   okdist-releases-dir : $(okdist-releases-dir)
        Directory holding binary releases, from which tarballs are exploded   

   okdist-install-tests 
        Creates $(opticks-dir)/tests populated with CTestTestfile.cmake files 

   okdist-create
        Creates distribution tarball in the installation directory  

   okdist-explode
        Explode distribution tarball from the releases directory 

   okdist-release-prefix : $(okdist-release-prefix) 
        Absolute path to exploded release distribution

   okdist--
       From the installation directory, creates tarball with 
       all paths starting with the okdist-prefix  

EOI
}


okdist-install-tests()
{
   opticks-
   local bdir=$(opticks-bdir)
   local dest=$(opticks-dir)/tests
   CTestTestfile.py $bdir --dest $dest
}


okdist-create()
{
   local iwd=$PWD

   opticks-
   opticks-cd  ## install directory 

   okdist-install-tests 

   okdist.py --distprefix $(okdist-prefix) --distname $(okdist-name)  --exclude_geant4

   ls -al $(okdist-name) 
   du -h $(okdist-name) 

   cd $iwd
}


okdist-ls(){      echo $FUNCNAME ; local p=$(okdist-path) ; ls -l $p ; du -h $p ; }

okdist-explode(){ $FUNCNAME- $(okdist-path) ; }
okdist-explode-(){    
    local msg="=== $FUNCNAME :"
    local iwd=$PWD
    local path=$1 

    if [ -z "$path" ]; then 
        echo $msg expects path argument
        return 1 
    fi  
    
    if [ ! -f "$path" ]; then 
        echo $msg path $path does not exist
        return 2 
    fi 

    local releases_dir=$(okdist-releases-dir)
    if [ ! -d $releases_dir ]; then 
        echo $msg creating releases dir $releases_dir
        mkdir -p $releases_dir
    fi 

    cd $releases_dir

    local opt=""
    [ -n "$VERBOSE" ] && opt="v"  


    local prefix=$(okdist-prefix)   # eg Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg

    if [ -d "$prefix" ]; then
       echo $msg an exploded tarball is already present at prefix $prefix
       local ans
       #read -p "Enter Y to delete this directory : " ans
       ans="Y"
       [ "$ans" == "Y" ] && echo $msg proceeding to delete $prefix && rm -rf $prefix  
    fi 



    echo $msg exploding distribution $path from PWD $PWD
    case $(okdist-ext) in 
       .tar.gz) tar zx${opt}f $path ;;
          .tar) tar  x${opt}f $path ;;
    esac

    cd $iwd
}

okdist-lst(){
    local path=$(okdist-path)
    case $(okdist-ext) in 
       .tar.gz) tar ztvf $path ;;
          .tar) tar  tvf $path ;;
    esac
}

okdist--(){        

   okdist-create
   okdist-explode
   okdist-ls  
}










okdist-test-notes(){ cat << EON

* ppm snaps defaulting to triangulated, still need --xanalytic switch.
* how to be sure are using the packaged libs ?

  * recall some RPATH setup for Linux using ORIGIN which enables the
    libs to be found relative to the executables 
  * notes/issues/packaging-opticks-and-externals-for-use-on-gpu-cluster.rst  

  * where is the sensitivity to opticks-f OPTICKS_INSTALL_PREFIX

EON
}


okdist-snap-()
{
    PATH=$(okdist-tmp)/lib:$PATH which OpSnapTest   
}

okdist-test()
{
    #OPTICKS_INSTALL_PREFIX=$(okdist-tmp) $(okdist-tmp)/lib/OpticksResourceTest --envkey 
    #LD_TRACE_LOADED_OBJECTS=1 $(okdist-tmp)/lib/OpticksResourceTest

   # gdb --args \

    OPTICKS_INSTALL_PREFIX=$(okdist-tmp) \
        $(okdist-tmp)/lib/OpSnapTest --envkey --xanalytic --target 352851 --eye -1,-1,-1 --snapconfig "steps=10,eyestartz=-1,eyestopz=5" --size 2560,1440,1 --embedded

    local ppm=$(opticks-dir)/tmp/snap00000.ppm 
    ls -l $ppm
    open $ppm
}



