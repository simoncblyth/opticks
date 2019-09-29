
scdist-source(){ echo $BASH_SOURCE ; }
scdist-sdir(){   echo $(dirname $BASH_SOURCE) ; }
scdist-py(){     echo $(scdist-sdir)/scdist.py ; }
scdist-vi(){     vi $BASH_SOURCE $(scdist-py)  ; }
scdist-env(){ echo -n ; }
scdist-usage(){  cat << \EOU

Opticks Shared Cache : geocache and rngcache  
==============================================

The Opticks shared cache enables all users on all nodes with access 
to cache to share the same geometry cache files (geocache) 
and curandState files (rngcache). 

This avoids wasteful duplication of disk storage and processing. 


Workflow to publish a shared cache
-----------------------------------

1. workstation: collect geocache and rngcache into /opticks/sharedcache

2. workstation: delete files that are not needed from /opticks/sharedcache

3. workstation: create tarball, explode it as a test::

   scdist--

4. workstation: test usage from other user, example environment setup in next section:: 

   su - simon 

   release-test

5. workstation: copy tarball to remote node for publishing::

   scdist-publish 

6: from GPU cluster gateway node (lxslc): check tarball relative paths and explode::

   scdist-publish-cd

   ls -l  
   tar tvf ...

   scdist-publish-explode



TODO : slim the cache
--------------------------

::

    7.5G    /opticks/sharedcache/OpticksSharedCache-0.0.0_alpha.tar


The rngmax storage approach for curandStates is simple, 
but it wastes probably 3-4 GB with duplicated curandStates.



Environment setup for use of Opticks binary distribution and shared cache from GPU cluster/gateway node
-----------------------------------------------------------------------------------------------------------

::

    source /hpcfs/juno/junogpu/blyth/local/opticks/externals/envg4.bash

    #source /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/bin/release.bash  
    # real /cvmfs

    source /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/bin/release.bash
    # testing release on /hpcfs before push it to /cvmfs

    source /hpcfs/juno/junogpu/blyth/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/bin/sharedcache.bash


TODO
------

Need a single script location so this triplet of paths doesnt 
have to be duplicated by each user : perhaps:: 

    source /hpcfs/opticks.ihep.ac.cn/Opticks-0.0.0_alpha/oki.bash



Workstation : Test Environment setup for use of Opticks binary distribution and shared cache (from ~simon/.bashrc)
-----------------------------------------------------------------------------------------------------------------------
::

    source /home/blyth/local/opticks/externals/envg4.bash

    source /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/bin/opticks-release.bash  # fake /cvmfs

    source /opticks/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/bin/sharedcache.bash



EOU
}


scdist-base(){  echo /opticks/sharedcache ; }  ## analogous to opticks-dir for binaries and libs
scdist-cd(){    cd $(scdist-base) ; }

scdist-releases-dir-default(){ echo /opticks/opticks.ihep.ac.cn/sc/releases ; }
scdist-releases-dir(){         echo ${SCDIST_RELEASES_DIR:-$(scdist-releases-dir-default)} ; } 
scdist-rcd(){                  cd $(scdist-releases-dir) ; }


scdist-title(){   echo OpticksSharedCache ; }
scdist-version(){ echo 0.0.0_alpha ; }
scdist-ext(){     echo .tar ; }  
scdist-prefix(){  echo $(scdist-title)-$(scdist-version) ; }  
scdist-name(){    echo $(scdist-title)-$(scdist-version)$(scdist-ext) ; }
scdist-path(){    echo $(scdist-base)/$(scdist-name) ; }    

scdist-release-prefix(){ echo $(scdist-releases-dir)/$(scdist-prefix) ; } 


scdist-info(){ cat << EOI
$FUNCNAME
=============

   date          : $(date)
   epoch         : $(date +"%s")
   uname -a      : $(uname -a)

   scdist-ext    : $(scdist-ext)
   scdist-prefix : $(scdist-prefix)
   scdist-name   : $(scdist-name)
   scdist-path   : $(scdist-path)

   scdist-base   : $(scdist-base)
       Shared Cache source directory 


   scdist-releases-dir-default : $(scdist-releases-dir-default)
   SCDIST_RELEASES_DIR         : $SCDIST_RELEASES_DIR 

   scdist-releases-dir : $(scdist-releases-dir)
        Directory holding releases, from which tarballs are exploded   

   scdist-create
        Creates distribution tarball 

   scdist-explode
        Explode distribution tarball from the releases directory 

   scdist-release-prefix : $(scdist-release-prefix) 
        Absolute path to exploded release distribution

   scdist--
       From the installation directory, creates tarball with 
       all paths starting with the scdist-prefix  

EOI
}


scdist-metadata()
{
   local mdir="$(scdist-base)/metadata"
   [ ! -d "$mdir" ] && mkdir -p "$mdir"
   scdist-info     > $mdir/scdist-info.txt
}



scdist-create()
{
   local msg="=== $FUNCNAME :"
   local iwd=$PWD

   scdist-cd      ## analogous to install directory 

   echo $msg install setup script
   mkdir -p bin
   cp $(opticks-home)/bin/opticks-sharedcache.bash bin/

   echo $msg write metadata
   scdist-metadata

   echo $msg create tarball
   scdist.py --distprefix $(scdist-prefix) --distname $(scdist-name) 

   echo $msg list tarball
   ls -al $(scdist-name) 
   du -h $(scdist-name) 

   cd $iwd
}


scdist-ls(){      echo $FUNCNAME ; local p=$(scdist-path) ; ls -l $p ; du -h $p ; }


scdist-explode-notes(){ cat << EON
$FUNCNAME
======================

* scdist-path argument is the absolute path of the tarball, which 
  is typically directly inside scdist-dir 

* relative paths inside tarballs are such that the tarballs 
  should always be exploded from the releases dir in order to get the intended layout,  
  scdist-explode does this

* directories with preexisting exploded tarballs are deleted, to 
  avoid mixing 

EON
}



scdist-test-explode(){ 
    local iwd=$PWD

    local releases_dir=$(scdist-releases-dir)

    scdist-prepare-releases-dir $releases_dir
    rc=$?
    [ $rc -ne 0 ] && echo $msg ERROR rc $rc && return $rc

    cd $releases_dir
    rc=$?
    [ $rc -ne 0 ] && echo $msg ERROR rc $rc && return 4

    scdist-explode-here- $(scdist-path) ;

    cd $iwd
}


scdist-prepare-releases-dir()
{
    local msg="=== $FUNCNAME :"
    local rc=0
    local releases_dir=$1
    if [ ! -d $releases_dir ]; then 
        echo $msg creating releases dir $releases_dir
        mkdir -p $releases_dir
        rc=$?
        [ $rc -ne 0 ] && echo $msg ERROR rc $rc && return $rc
    fi 
    return $rc
}


scdist-explode-here-(){    
    local msg="=== $FUNCNAME :"
    local path=$1 
    local rc=0

    if [ -z "$path" ]; then 
        echo $msg expects path argument to tarball to be exploded
        return 1 
    fi  
    
    if [ ! -f "$path" ]; then 
        echo $msg path $path does not exist
        return 2 
    fi 
 
    echo $msg explode tarball $path from $PWD
    local opt=""
    [ -n "$VERBOSE" ] && opt="v"  

    local prefix=$(scdist-prefix) 

    if [ -d "$prefix" ]; then
       echo $msg an exploded tarball is already present at prefix $prefix
       local ans
       #read -p "Enter Y to delete this directory : " ans
       ans="Y"
       [ "$ans" == "Y" ] && echo $msg proceeding to delete $prefix && rm -rf $prefix  
    fi 

    echo $msg exploding distribution $path from PWD $PWD
    case $(scdist-ext) in 
       .tar.gz) tar zx${opt}f $path ;;
          .tar) tar  x${opt}f $path ;;
    esac

}

scdist-lst(){
    local path=$(scdist-path)
    case $(scdist-ext) in 
       .tar.gz) tar ztvf $path ;;
          .tar) tar  tvf $path ;;
    esac
}

scdist--(){        

    scdist-create
    scdist-test-explode
    scdist-ls  
}


scdist-publish-notes(){ cat << EON

scdist-publish
    done from workstation, copies tarball from workstation to GPU cluster OR gateway node

scdist-publish-explode
    done from GPU cluster or gateway node (lxslc), explodes tarball with pre-deletion to avoid mixing 
    NB this is only significant scdist- function done from gateway or GPU cluster node

EON
}

scdist-publish-node(){ echo ${SCDIST_PUBLISH_NODE:-L7} ; }
scdist-publish-base(){ echo ${SCDIST_PUBLISH_BASE:-/hpcfs/juno/junogpu/blyth/opticks.ihep.ac.cn/sc/releases} ; }
scdist-publish()
{
    local pnode=$(scdist-publish-node)
    local pbase=$(scdist-publish-base)
    local cmd

    cmd="ssh $pnode mkdir -p $pbase"
    echo $cmd
    eval $cmd

    cmd="scp $(scdist-path) $pnode:$pbase/"
    echo $cmd
    eval $cmd
}

scdist-publish-cd(){  cd $(scdist-publish-base) ; }

scdist-publish-explode()
{
    local msg="=== $FUNCNAME :"
    local pbase=$(scdist-publish-base)
    [ ! -d "$pbase" ] && echo $msg ERROR NON EXISTING pbase $pbase && return 1

    cd $pbase

    local pname=$(scdist-name)   
    scdist-explode-here- $pname 
}


