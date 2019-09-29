opticks-sharedcache-source(){ echo $BASH_SOURCE ; }
opticks-sharedcache-vi(){     vi $BASH_SOURCE ; } 
opticks-sharedcache-(){       source $BASH_SOURCE ; }

opticks-sharedcache-dir(){    echo $(dirname $BASH_SOURCE) ; }
opticks-sharedcache-prefix(){ echo $(dirname $(dirname $BASH_SOURCE)) ; }
opticks-sharedcache-cd(){     cd $(opticks-sharedcache-prefix) ; }


opticks-sharedcache-usage(){ cat << EOU
$FUNCNAME
============================

This sharedcache.bash script sets up access to the geocache and rngcache, 
which are needed by the Opticks libraries, executables and scripts.

Configure access to the shared cache by including a line in your ~/.bashrc similar to the below:: 

   source /hpcfs/juno/junogpu/blyth/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/bin/opticks-sharedcache.bash 
   source                  /opticks/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/bin/opticks-sharedcache.bash 


FUNCTIONS
-----------

opticks-sharedcache-main
    sets two envvars configuring geocache, run on sourcing this script

opticks-sharedcache-unset
    unsets the two envvars 

opticks-sharedcache-info
    dump function outputs and envvars

EOU
}

opticks-sharedcache-info(){ cat << EOI
$FUNCNAME
==========================

  opticks-sharedcache-source  : $(opticks-sharedcache-source)
  opticks-sharedcache-dir     : $(opticks-sharedcache-dir)
  opticks-sharedcache-prefix  : $(opticks-sharedcache-prefix)

  OPTICKS_SHARED_CACHE_PREFIX : $OPTICKS_SHARED_CACHE_PREFIX
  OPTICKS_KEY                 : $OPTICKS_KEY

EOI
}

opticks-sharedcache-unset(){
   unset OPTICKS_SHARED_CACHE_PREFIX
   unset OPTICKS_KEY
}

opticks-sharedcache-main(){

   unset OPTICKS_SHARED_CACHE_PREFIX
   export OPTICKS_SHARED_CACHE_PREFIX=$(opticks-sharedcache-prefix)

   unset OPTICKS_KEY
   #export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce   ## geocache-j1808-v5-key
   export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.5aa828335373870398bf4f738781da6c # geocache-dx-v0-key

   ## hmm need different default key while testing and in real usage ? 
}


opticks-sharedcache-main

