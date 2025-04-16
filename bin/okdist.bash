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

* **test installation has been moved to opticks-install-tests**



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

okdist-tmp(){     echo /tmp/$USER/opticks/okdist ; }
okdist-cd(){      cd $(okdist-tmp) ; }

# git -C not available in older git
#okdist-revision(){ git -C $(opticks-home) rev-parse HEAD ; }
okdist-revision(){  echo $(cd $(opticks-home) && git rev-parse HEAD) ; }

#okdist-release-dir-default(){ echo $(opticks-dir)_release ; }
okdist-release-dir-default(){ echo $(opticks-dir) ; }
okdist-release-dir(){         echo ${OKDIST_RELEASE_DIR:-$(okdist-release-dir-default)} ; }

okdist-title(){   echo Opticks ; }
okdist-version(){ opticks-tag ; }
okdist-ext(){     echo .tar ; }  # .tar.gz is slow to create and only half the size : .tar better while testing
okdist-prefix-old(){ echo $(okdist-title)-$(okdist-version)/$(opticks-okdist-dirlabel) ; }
okdist-prefix(){ echo $(opticks-okdist-dirlabel)/$(okdist-stem) ; }
okdist-stem(){   echo $(okdist-title)-$(okdist-version) ; }
okdist-name(){   echo $(okdist-stem)$(okdist-ext) ; }
okdist-path(){   echo $(okdist-release-dir)/$(okdist-name) ; }


okdist-release-prefix(){ echo $(okdist-release-dir)/$(okdist-prefix) ; }
okdist-rcd(){            cd $(okdist-release-prefix) ; }

okdist-info(){ cat << EOI
$FUNCNAME
=============

   date             : $(date)
   epoch            : $(date +"%s")
   uname -a         : $(uname -a)
   okdist-revision  : $(okdist-revision)

   okdist-ext    : $(okdist-ext)
   okdist-prefix : $(okdist-prefix)
   opticks-tag   : $(opticks-tag)
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




okdist-install-metadata()
{
   local mdir="$(opticks-dir)/metadata"

   [ ! -d "$mdir" ] && mkdir -p "$mdir"

   okdist-info     > $mdir/okdist-info.txt
   okdist-revision > $mdir/okdist-revision.txt
}



okdist-install-extras()
{
   local msg="=== $FUNCNAME :"
   local iwd=$PWD

   opticks-
   opticks-cd  ## install directory

   echo $msg write metadata
   okdist-install-metadata

   cd $iwd
}


okdist-tarball-notes(){ cat << EON
$FUNCNAME
======================

* okdist-path argument is the absolute path of the tarball, which
  is typically directly inside the install dir opticks-dir

* directories with preexisting extracted tarballs are deleted, to
  avoid mixing

* okdist-prefix is the prefix used within the tarball
  eg Opticks-v0.3.3/i386-10.13.6-gcc421-geant4_10_04_p02-dbg
  that relative prefix becomes the holder directories on extracting

For example the extracted prefix directory is::

    /usr/local/opticks_release/Opticks-0.0.1_alpha/i386-10.13.6-gcc4.2.1-geant4_10_04_p02-dbg

HMM: recall that not everything is installed by CMake other
files such as the below are installed with opticks-setup-generate
that is run by opticks-full::

    bin/opticks-setup.sh
    bin/opticks-setup.csh

EON
}

okdist-tarball-create(){
   : bin/okdist.bash create tarball

   echo === $FUNCNAME

   opticks-cd
   $(opticks-home)/bin/oktar.py $(okdist-path) create --prefix $(okdist-prefix)

   echo $msg list tarball : $(okdist-path)
   ls -al $(okdist-path)
   du -h $(okdist-path)
}

okdist-tarball-extract(){
   : bin/okdist.bash extract tarball into release dir

   echo === $FUNCNAME
   $(opticks-home)/bin/oktar.py $(okdist-path) extract --base $(okdist-release-dir) ;
}

okdist-tarball-extract-plant-latest-link()
{
    local pfx=$(okdist-release-prefix) ## eg /data/blyth/opticks_Debug/el7_amd64_gcc1120/Opticks-v0.3.5
    local nam=$(basename $pfx)    ## eg Opticks-v0.3.5
    local dir=$(dirname $pfx)     ## eg /data/blyth/opticks_Debug/el7_amd64_gcc1120
    local LNK=Opticks-vLatest
    local iwd=$PWD
    cd $dir && ln -sfn $nam $LNK
    [ $? -ne 0 ] && echo $FUNCNAME - ERROR PLANTING LINK && return 1

    pwd
    ls -alst .

    cd $iwd
    return 0
}


okdist-tarball-dump(){
   : bin/okdist.bash extract tarball into release dir

   echo === $FUNCNAME
   oktar.py $(okdist-path) dump
}

# list the tarball
okdist-ls(){      echo $FUNCNAME ; local p=$(okdist-path) ; ls -l $p ; du -h $p ; }

okdist-lst(){
    local path=$(okdist-path)
    case $(okdist-ext) in
       .tar.gz) tar ztvf $path ;;
          .tar) tar  tvf $path ;;
    esac
}

okdist--(){
   okdist-install-extras
   okdist-tarball-create
   okdist-tarball-extract
   okdist-tarball-extract-plant-latest-link
   okdist-ls

   #echo $msg okdist-deploy-opticks-site
   #okdist-deploy-opticks-site
}

okdist-deploy-notes(){ cat << EON
Maybe reuse installed bin/opticks-setup.sh instead of the site machinery ?
To setup use of the binary release.


    src=/usr/local/opticks
    dst=/some/new/prefix
    f=check.txt

    sed -i -e "s,$src,$dst,g" $f

OR could pass in a prefix to opticks-setup-generate

EON
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


okdist-deploy-to-cvmfs()
{
   local dist=$(okdist-path)
   local name=$(basename $dist)

   local cmd0="scp $dist O:"
   local cmd1="ssh O \"./ok_deploy_to_cvmfs.sh $name\""
   local ii="0 1"

   for i in $ii
   do
      _cmd="cmd$i"
      echo ${_cmd}
      echo ${!_cmd}
      eval ${!_cmd}
      [ $? -ne 0 ] && echo $FUNCNAME - FAILED
   done

}


