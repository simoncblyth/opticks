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

Recall the many trees 
------------------------

1. source tree, in the repository 
2. build tree, in which Makefiles are CMake generated and build products are made
3. install tree,  where source build products end up after an install 
4. distribution tarball, collection of products of the install  
5. release tree, the result of exploding the distribution tarball 

* a bitbucket clone gives you 1
* running *om--* yields 2 and 3
* runnning *okdist--* yields 4 and 5 

Note that this needs to be repeated on the workstation 
and the GPU cluster gateway node.


Gotchas
---------

1. development not user environment required, otherwise fails with python modules not found 
2. even scripts need to be installed before okdist-- can package them
   for consistency reasons, to do this:

   * list them as needing install in eg bin/CMakeLists.txt 
   * run the om-- for the package, eg bin
   * only then can okdist-- package the updated/added scripts into tarball


workflow for Opticks binary releases
----------------------------------------

0. workstation: get to clean revision by commit and push, then build and test::

     o
     om--
     opticks-t 

0.5 workstation: test creating and exploding tarball onto fake /cvmfs::

    okdist--

0.6 workstation: test the binary release by running tests as unpriviled simon::

     su - simon 
     opticks-release-test 

1. lxslc: update repo and build 

2. lxslc: create tarball and test exploding it into releases:: 
 
     okdist--

3. lxslc: test release from user like environment, 
   
   * switch from developer_setup to user_setup in ~/.bash_profile and ssh into another tab  
   * see scdist- for examples of the source lines for user setup

   * run tests::

     opticks-release-check  
     opticks-release-test    ## everything that needs GPU will fail 


4. lxslc: copy tarball and python script to stratum zero node::

      scp $(okdist-path) ~/opticks/bin/oktar.py O:  

5a. automated way::

   ssh O   
       ## login to stratum zero

   cvmfs_server transaction opticks.ihep.ac.cn 
       ## start transaction 

    ~/oktar.py ~/Opticks-0.0.0_alpha.tar   ## check the tarball
    ~/oktar.py ~/Opticks-0.0.0_alpha.tar --explode --base /cvmfs/opticks.ihep.ac.cn/ok/releases  ## explode the tarball

   cvmfs_server publish -m "First Release Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg" opticks.ihep.ac.cn
       ## close and publish this transaction


5b. manual way of publishing the tarball release onto cvmfs::

   ssh O   
       ## login to stratum zero

   cvmfs_server transaction opticks.ihep.ac.cn 
       ## start transaction 

   cd /cvmfs/opticks.ihep.ac.cn
   mkdir -p ok/releases                
       ## create top folders if this is first release 

   cd /cvmfs/opticks.ihep.ac.cn/ok/releases 
       ## get into releases folder

   tar tvf ~/Opticks-0.0.0_alpha.tar   
       ## list tarball contents, check relative paths are correct 

   rm -rf Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg  
       ## delete any prior attempts for this architecture/versions 

   tar xvf ~/Opticks-0.0.0_alpha.tar   
       ## explode the tarball

   cd /cvmfs
       ## get out of dodge

   cvmfs_server publish -m "First Release Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg" opticks.ihep.ac.cn
       ## close and publish this transaction





okdist testing on workstation
-----------------------------------

To test running from exploded tarball onto fake /cvmfs 
on gold workstation add the below to setup::

    unset OKDIST_RELEASES_DIR
    export OKDIST_RELEASES_DIR=/cvmfs/opticks.ihep.ac.cn/ok/releases

This will override the default okdist-releases-dir which is a releases 
folder inside the normal opticks-dir.

EOU
}

okdist-tmp(){     echo /tmp/$USER/opticks/okdist-test ; }
okdist-cd(){      cd $(okdist-tmp) ; }

okdist-revision(){   
  if [ -d "$(opticks-home)/.hg" ]; then 
     okdist-revision-hg
  elif [ -d "$(opticks-home)/.git" ]; then 
     okdist-revision-git
  else
     echo $FUNCNAME-error 
  fi 
}
okdist-revision-hg(){   hg --debug id -i $(opticks-home) ; } 
okdist-revision-git(){  ( cd $(opticks-home) && git rev-parse HEAD ) ; } 

okdist-releases-dir-default(){ echo $(opticks-dir)/releases ; }
okdist-releases-dir(){         echo ${OKDIST_RELEASES_DIR:-$(okdist-releases-dir-default)} ; } 
okdist-rcd(){                  cd $(okdist-releases-dir) ; }


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

   date             : $(date)
   epoch            : $(date +"%s")
   uname -a         : $(uname -a)
   okdist-revision  : $(okdist-revision)   

   okdist-ext    : $(okdist-ext)
   okdist-prefix : $(okdist-prefix)
   okdist-name   : $(okdist-name)
   okdist-path   : $(okdist-path)

   opticks-dir   : $(opticks-dir)
       Opticks installation directory 


   okdist-releases-dir-default : $(okdist-releases-dir-default)
   OKDIST_RELEASES_DIR         : $OKDIST_RELEASES_DIR 

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

okdist-install-cmake-modules()
{
   opticks-
   local home=$(opticks-home)
   local dest=$(opticks-dir)

   CMakeModules.py --home $home --dest $dest
}

okdist-metadata()
{
   local mdir="$(opticks-dir)/metadata"

   [ ! -d "$mdir" ] && mkdir -p "$mdir"

   okdist-info     > $mdir/okdist-info.txt
   okdist-revision > $mdir/okdist-revision.txt
}


okdist-deploy-opticks-site()
{
   local msg="=== $FUNCNAME :"
   echo $msg $PWD
   local script=bin/opticks-site.bash
   if [ -f "$script" ]; then
       source $script
       opticks-site-deploy
   else
       echo $msg missing script $script
   fi
}


okdist-create()
{
   local msg="=== $FUNCNAME :"
   local iwd=$PWD

   opticks-
   opticks-cd  ## install directory 

   echo $msg write metadata
   okdist-metadata

   echo $msg install tests
   okdist-install-tests 

   echo $msg install cmake/Modules 
   okdist-install-cmake-modules 

   echo $msg create tarball
   okdist.py --distprefix $(okdist-prefix) --distname $(okdist-name) 

   echo $msg list tarball
   ls -al $(okdist-name) 
   du -h $(okdist-name) 

   echo $msg okdist-deploy-opticks-site
   okdist-deploy-opticks-site


   cd $iwd
}


okdist-ls(){      echo $FUNCNAME ; local p=$(okdist-path) ; ls -l $p ; du -h $p ; }



okdist-explode-notes(){ cat << EON
$FUNCNAME
======================

* okdist-path argument is the absolute path of the tarball, which 
  is typically directly inside the install dir opticks-dir 

* relative paths inside tarballs are such that the tarballs 
  should always be exploded from the releases dir in order to get the intended layout,  
  okdist-explode does this

* directories with preexisting exploded tarballs are deleted, to 
  avoid mixing 


EON
}




okdist-explode(){ oktar.py $(okdist-path) --explode --base $(okdist-releases-dir) ; }


okdist-explode-old(){ $FUNCNAME- $(okdist-path) ; }
okdist-explode-old-(){    
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
    echo $msg explode tarball $path from $PWD

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



