
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

2. workstation: delete files that are not needed from  /opticks/sharedcache

3. workstation: create the tarball, explode it as a test::

   scdist--

4. workstation: test usage from other user, example environment setup next section:: 

   su - simon 

   release-test


4. copy tarball to remote node for publishing 



Example environment setup for use of Opticks binary distribuation and shared cache (from ~simon/.bashrc)
----------------------------------------------------------------------------------------------------------

::

    export LD_LIBRARY_PATH=/home/blyth/local/opticks/externals/lib64:$LD_LIBRARY_PATH    
    source ~blyth/g4-envg4.bash

       ## setup access to Geant4 libs and data 

    source /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/bin/release.bash

       ## setup access to Opticks executables, scripts and libs including externals other than Geant4

    source /opticks/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/bin/sharedcache.bash

       ## setup access to the shared cache 

    export OPTICKS_DEFAULT_INTEROP_CVD=1  

       ## fixup for interop running on machines with multiple GPUs, 
       ## selects the GPU cvd (CUDA_VISIBLE_DEVICES) ordinal that is connected to monitor


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
   cp $(opticks-home)/bin/sharedcache.bash bin/

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

scdist-explode(){ $FUNCNAME- $(scdist-path) ; }
scdist-explode-(){    
    local msg="=== $FUNCNAME :"
    local iwd=$PWD
    local path=$1 
    local rc=0

    if [ -z "$path" ]; then 
        echo $msg expects path argument
        return 1 
    fi  
    
    if [ ! -f "$path" ]; then 
        echo $msg path $path does not exist
        return 2 
    fi 

    local releases_dir=$(scdist-releases-dir)
    if [ ! -d $releases_dir ]; then 
        echo $msg creating releases dir $releases_dir
        mkdir -p $releases_dir
        rc=$?
        [ $rc -ne 0 ] && echo $msg ERROR rc $rc && return 3
    fi 

    cd $releases_dir
    rc=$?
    [ $rc -ne 0 ] && echo $msg ERROR rc $rc && return 4
 
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

    cd $iwd
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
    scdist-explode
    scdist-ls  
}


scdist-publish-notes(){ cat << EON

* copies tarball to remote publish node 

EON
}

scdist-publish-node(){ echo ${SCDIST_PUBLISH_NODE:-L7} ; }
scdist-publish-base(){ echo ${SCDIST_PUBLISH_BASE:-/hpcfs/juno/junogpu/blyth/opticks.ihep.ac.cn/ok/releases} ; }
scdist-publish()
{
    local rnode=$(scdist-publish-node)
    local rbase=$(scdist-publish-base)
    ssh $rnode mkdir -p $rbase
    scp $(scdist-path) $rnode:$rbase/
}

scdist-publish-cd(){  cd $(scdist-publish-base) ; }
