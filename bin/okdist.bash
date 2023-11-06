okdist-env(){ echo -n ; }
okdist-source(){ echo $BASH_SOURCE ; }
okdist-vi(){ vi $BASH_SOURCE $(okdist-py) \
     $(opticks-home)/notes/issues/packaging-opticks-and-externals-for-use-on-gpu-cluster.rst ; 
}
okdist-sdir(){ echo $(dirname $BASH_SOURCE) ; }
okdist-py(){ echo $(okdist-sdir)/okdist.py ; }


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
* running *oo* yields 2 and 3
* runnning *okdist--* yields 4 and 5 

Note that this needs to be repeated on the workstation 
and the GPU cluster gateway node (lxslc7).

Gotchas
---------

1. development not user environment required, otherwise fails with python modules not found 
2. even scripts need to be installed before okdist-- can package them
   for consistency reasons, to do this:

   * list them as needing install in eg bin/CMakeLists.txt 
   * run the om-- for the package, eg bin
   * only then can okdist-- package the updated/added scripts into tarball


workflow for Opticks binary release
----------------------------------------

0. workstation: get to clean revision by commit and push, then build and test::

     oo
     opticks-t 

1. workstation: test creating and exploding tarball onto fake /cvmfs::

    okdist--

2. workstation: test the binary release by running tests as unpriviled simon::

     su - simon 
     opticks-release-test 

3. lxslc: update repo and build 

4. lxslc: create tarball and test exploding it into release dir:: 
 
     okdist--

5. lxslc: test release from user like environment, 
   
   * switch from developer_setup to user_setup in ~/.bash_profile and ssh into another tab  
   * see scdist- for examples of the source lines for user setup

   * run tests::

     opticks-release-check  
     opticks-release-test    ## everything that needs GPU will fail 


6. lxslc: copy tarball and python script to stratum zero node::

      okdist- ; scp $(okdist-path) $(which oktar.py) O:  

7. automated way::

   ssh O   
       ## login to stratum zero

   cvmfs_server transaction opticks.ihep.ac.cn 
       ## start transaction 

    ~/oktar.py ~/Opticks-0.0.0_alpha.tar   ## check the tarball
    ~/oktar.py ~/Opticks-0.0.0_alpha.tar --explode --base /cvmfs/opticks.ihep.ac.cn/ok/release  ## explode the tarball

   cvmfs_server publish -m "First technical release Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg" opticks.ihep.ac.cn
       ## close and publish this transaction


8. manual way of publishing the tarball release onto cvmfs::

   ssh O   
       ## login to stratum zero

   cvmfs_server transaction opticks.ihep.ac.cn 
       ## start transaction 

   cd /cvmfs/opticks.ihep.ac.cn
   mkdir -p ok/release 
       ## create top folders if this is first release 

   cd /cvmfs/opticks.ihep.ac.cn/ok/release
       ## get into release folder

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


run installed ctests
-------------------------------

After okdist-- okdist-create-tests 
tests are installed into the "tests" subfolder
under the installation prefix::

    opticks-
    opticks-cd tests

    ctest -N                   ## list tests
    ctest --output-on-failure  ## run them all 


okdist testing on workstation
-----------------------------------

To test running from exploded tarball onto fake /cvmfs 
on gold workstation add the below to setup::

    unset OKDIST_RELEASE_DIR
    export OKDIST_RELEASE_DIR=/cvmfs/opticks.ihep.ac.cn/ok/release

This will override the default okdist-release-dir which is a release
folder inside the normal opticks-dir.

EOU
}

okdist-tmp(){     echo /tmp/$USER/opticks/okdist-test ; }
okdist-cd(){      cd $(okdist-tmp) ; }
okdist-revision(){ git -C $(opticks-home) rev-parse HEAD ; } 
okdist-release-dir-default(){ echo $(opticks-dir)_release ; }
okdist-release-dir(){         echo ${OKDIST_RELEASE_DIR:-$(okdist-release-dir-default)} ; } 
okdist-rcd(){                  cd $(okdist-release-dir) ; }

okdist-title(){   echo Opticks ; }
okdist-version(){ echo 0.0.1_alpha ; } # HMM:eventually this should be tied to opticks tags  
okdist-ext(){     echo .tar ; }  # .tar.gz is slow to create and only half the size : .tar better while testing 
okdist-prefix(){ echo $(okdist-title)-$(okdist-version)/$(opticks-okdist-dirlabel) ; }  
okdist-name(){   echo $(okdist-title)-$(okdist-version)$(okdist-ext) ; }
okdist-path(){   echo $(opticks-dir)/$(okdist-name) ; }    

okdist-release-prefix(){ echo $(okdist-release-dir)/$(okdist-prefix) ; } 


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


   okdist-release-dir-default : $(okdist-release-dir-default)
   OKDIST_RELEASE_DIR         : $OKDIST_RELEASE_DIR 

   okdist-release-dir : $(okdist-release-dir)
        Directory holding binary release, from which tarballs are exploded   

   okdist-install-tests 
        Creates $(opticks-dir)/tests populated with CTestTestfile.cmake files 

   okdist-create
        Creates distribution tarball in the installation directory  

   okdist-explode
        Explode distribution tarball from the release directory 

   okdist-release-prefix : $(okdist-release-prefix) 
        Absolute path to exploded release distribution

   okdist--
       From the installation directory, creates tarball with 
       all paths starting with the okdist-prefix  

EOI
}


okdist-install-tests()
{
   local msg="=== $FUNCNAME :"
   opticks-
   local bdir=$(opticks-bdir)
   local dest=$(opticks-dir)/tests
   echo $msg bdir $bdir dest $dest
   CTestTestfile.py $bdir --dest $dest
   local script=$dest/ctest.sh 

   cat << EOT > $script
#!/bin/bash -l 
#ctest -N 
ctest --output-on-failure
EOT

   chmod ugo+x $script 

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
    
       source $(opticks-site-path)   # replace all opticks-site functions with the deployed ones
       opticks-site-deploy-html 
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

   #echo $msg okdist-deploy-opticks-site
   #okdist-deploy-opticks-site

   cd $iwd
}


# list the tarball 
okdist-ls(){      echo $FUNCNAME ; local p=$(okdist-path) ; ls -l $p ; du -h $p ; }



okdist-explode-notes(){ cat << EON
$FUNCNAME
======================

* okdist-path argument is the absolute path of the tarball, which 
  is typically directly inside the install dir opticks-dir 

* relative paths inside tarballs are such that the tarballs 
  should always be exploded from the release dir in order to get the intended layout,  
  okdist-explode does this

* directories with preexisting exploded tarballs are deleted, to 
  avoid mixing 

For example the exploded prefix directory is::

    /usr/local/opticks_release/Opticks-0.0.1_alpha/i386-10.13.6-gcc4.2.1-geant4_10_04_p02-dbg

EON
}

okdist-explode(){ 
   : bin/okdist.bash explode tarball into release dir 
   oktar.py $(okdist-path) --explode --base $(okdist-release-dir) ; 
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

