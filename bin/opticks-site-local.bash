opticks-site-local-notes(){ cat << EON

opticks-site-local.bash
=========================

This script is source "included" from the parent script opticks-site.bash, 
it acts to collect local customizations for a site allowing the main script 
to stay generic.

After updates here use::

    opticks-site-      # redefine the optick-site bash functions picking up the changes
    opticks-site-info
    opticks-site-check

OPTICKS_SITE_RELEASE_DIR
    envvar override allowing to switch release directory, usage examples::

    export OPTICKS_SITE_RELEASE_DIR=/hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg

         check a release candidate before publishing it to /cvmfs

    export OPTICKS_SITE_RELEASE_DIR=$(opticks-dir)

        check the source "release" for fast debugging, without needed to make releases, 
        while using mostly the same machinery as actual releases  


EON
}

opticks-site-user-prefix(){      echo /tmp/hpcfs/juno/junogpu ; }
opticks-site-user-home(){        echo $(opticks-site-user-prefix)/$USER ; } 
opticks-site-user-tmp(){         echo $(opticks-site-user-prefix)/$USER/oktmp ; } 
opticks-site-user-out(){         echo $(opticks-site-user-prefix)/$USER/okout ; } 
opticks-site-envg4(){            echo $(opticks-site-user-prefix)/blyth/local/opticks/externals/opticks-envg4.bash ; }
opticks-site-sharedcache(){      echo $(opticks-site-user-prefix)/blyth/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/bin/opticks-sharedcache.bash ; }
opticks-site-path(){             echo $(opticks-site-user-prefix)/blyth/opticks.ihep.ac.cn/opticks-site.bash ; }
opticks-site-path-local(){       local path=$(opticks-site-path) ; echo ${path/.bash/-local.bash} ;  }

opticks-site-release-base(){    echo /cvmfs/opticks.ihep.ac.cn/ok/releases ; }
opticks-site-release-version(){  echo Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg  ; }

opticks-site-release-dir(){      echo ${OPTICKS_SITE_RELEASE_DIR:-$(opticks-site-release-base)/$(opticks-site-release-version)} ; }

opticks-site-release(){          echo $(opticks-site-release-dir)/bin/opticks-release.bash ; }



