sharedcache-source(){ echo $BASH_SOURCE ; }
sharedcache-vi(){     vi $BASH_SOURCE ; } 
sharedcache-(){       source $BASH_SOURCE ; }

sharedcache-dir(){    echo $(dirname $BASH_SOURCE) ; }
sharedcache-prefix(){ echo $(dirname $(dirname $BASH_SOURCE)) ; }
sharedcache-cd(){     cd $(sharedcache-prefix) ; }


sharedcache-usage(){ cat << EOU
$FUNCNAME
============

This sharedcache.bash script sets up access to the geocache and rngcache, 
which are needed by the Opticks libraries, executables and scripts.

Configure access to the shared cache by including a line in your ~/.bashrc similar to the below:: 

   source /hpcfs/juno/junogpu/blyth/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/bin/sharedcache.bash 
   source                  /opticks/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/bin/sharedcache.bash 

EOU
}

sharedcache-info(){ cat << EOI
$FUNCNAME
============

  sharedcache-source : $(sharedcache-source)
  sharedcache-dir    : $(sharedcache-dir)
  sharedcache-prefix : $(sharedcache-prefix)

EOI
}


sharedcache-main(){

   export OPTICKS_SHARED_CACHE_PREFIX=$(sharedcache-prefix)

   unset OPTICKS_KEY
   #export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce   ## geocache-j1808-v5-key
   export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.5aa828335373870398bf4f738781da6c # geocache-dx-v0-key

   ## hmm need different default key while testing and in real usage ? 
}


sharedcache-main

