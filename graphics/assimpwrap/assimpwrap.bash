# === func-gen- : graphics/assimpwrap/assimpwrap fgp graphics/assimpwrap/assimpwrap.bash fgn assimpwrap fgh graphics/assimpwrap
assimpwrap-src(){      echo graphics/assimpwrap/assimpwrap.bash ; }
assimpwrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(assimpwrap-src)} ; }
assimpwrap-vi(){       vi $(assimpwrap-source) ; }
assimpwrap-env(){      elocal- ; }
assimpwrap-usage(){ cat << EOU

AssimpWrap
============

Wrapping the Assimp 3D Asset Importer Library

* Used by raytrace-


Workflow
---------

::

   assimpwrap-extra



FUNCTIONS
----------

Border Surfaces Looking Reasonable with expected AD symmetry
---------------------------------------------------------------

::

    AssimpGGeo::convertMaterials materialIndex 1
        bspv1 __dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458
        bspv2 __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08 
    AssimpGGeo::convertMaterials materialIndex 2
        bspv1 __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468
        bspv2 __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0 
    AssimpGGeo::convertMaterials materialIndex 4
        bspv1 __dd__Geometry__AD__lvSST--pvOIL0xc241510
        bspv2 __dd__Geometry__AD__lvADE--pvSST0xc128d90 
    AssimpGGeo::convertMaterials materialIndex 5
        bspv1 __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528
        bspv2 __dd__Geometry__AD__lvADE--pvSST0xc128d90 
    AssimpGGeo::convertMaterials materialIndex 6
        bspv1 __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE20xc0479c8
        bspv2 __dd__Geometry__AD__lvADE--pvSST0xc128d90 
    AssimpGGeo::convertMaterials materialIndex 7
        bspv1 __dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xc13c018
        bspv2 __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270 
    AssimpGGeo::convertMaterials materialIndex 8
        bspv1 __dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xc15a498
        bspv2 __dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc5c5f20 
    AssimpGGeo::convertMaterials materialIndex 11
        bspv1 __dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbf55b10
        bspv2 __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270 

    AssimpGGeo::convertStructureVisit border surface

    obs# 0 nodeIndex 3149 obs 0x11c2b2750 idx  7
        pv_p __dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xc13c018
        pv   __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270
    ibs# 0 nodeIndex 3150 ibs 0x11c2b3370 idx 11
        pv   __dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbf55b10
        pv_p __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270
    ibs# 1 nodeIndex 3152 ibs 0x11c2b2b00 idx  8
        pv   __dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xc15a498
        pv_p __dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc5c5f20

    obs# 1 nodeIndex 3154 obs 0x11c2b23d0 idx  5
        pv_p __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528
        pv   __dd__Geometry__AD__lvADE--pvSST0xc128d90
    ibs# 2 nodeIndex 3155 ibs 0x11c2b2220 idx  4
        pv   __dd__Geometry__AD__lvSST--pvOIL0xc241510
        pv_p __dd__Geometry__AD__lvADE--pvSST0xc128d90
    obs# 2 nodeIndex 4427 obs 0x11c2b1bc0 idx  2
        pv_p __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468
        pv   __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0
    obs# 3 nodeIndex 4430 obs 0x1061cc450 idx  1
        pv_p __dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458
        pv   __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08

    obs# 4 nodeIndex 4814 obs 0x11c2b2530 idx  6
        pv_p __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE20xc0479c8
        pv   __dd__Geometry__AD__lvADE--pvSST0xc128d90
    ibs# 3 nodeIndex 4815 ibs 0x11c2b2220 idx  4
        pv   __dd__Geometry__AD__lvSST--pvOIL0xc241510
        pv_p __dd__Geometry__AD__lvADE--pvSST0xc128d90
    obs# 5 nodeIndex 6087 obs 0x11c2b1bc0 idx  2
        pv_p __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468
        pv   __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0
    obs# 6 nodeIndex 6090 obs 0x1061cc450 idx  1
        pv_p __dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458
        pv   __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08



EOU
}
assimpwrap-idir(){ echo $(local-base)/env/graphics ; }  # prefix
assimpwrap-bdir(){ echo $(local-base)/env/graphics/assimpwrap.build ; }
assimpwrap-sdir(){ echo $(env-home)/graphics/assimpwrap ; }

assimpwrap-icd(){  cd $(assimpwrap-idir); }
assimpwrap-bcd(){  cd $(assimpwrap-bdir); }
assimpwrap-scd(){  cd $(assimpwrap-sdir); }

assimpwrap-cd(){  cd $(assimpwrap-sdir); }

assimpwrap-mate(){ mate $(assimpwrap-dir) ; }

assimpwrap-wipe(){
    local bdir=$(assimpwrap-bdir)
    rm -rf $bdir
}


assimpwrap-cmake(){
   local bdir=$(assimpwrap-bdir)
   mkdir -p $bdir
   assimpwrap-bcd
   cmake $(assimpwrap-sdir) -DCMAKE_INSTALL_PREFIX=$(assimpwrap-idir) -DCMAKE_BUILD_TYPE=Debug 
}

assimpwrap-make(){
    local iwd=$PWD
    assimpwrap-bcd
    local rc
    make $*
    rc=$?
    cd $iwd
    return $rc 
}

assimpwrap-install(){
   assimpwrap-make install
}


assimpwrap-bbin(){ echo $(assimpwrap-bdir)/AssimpWrapTest ; }
assimpwrap-bin(){ echo $(assimpwrap-idir)/bin/AssimpWrapTest ; }

assimpwrap-geokey(){
    case $1 in
      extra) echo DAE_NAME_DYB ;; 
          *) echo DAE_NAME_DYB_NOEXTRA  ;;
    esac  
}

assimpwrap-material(){
    echo __dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot 
    #echo __dd__Materials__GdDopedLS0xc2a8ed0 
}

assimpwrap-ggctrl(){
    echo __dd__
}


assimpwrap-export(){
    export ASSIMPWRAP_GEOKEY="$(assimpwrap-geokey $1)"
    export ASSIMPWRAP_QUERY="index:1,depth:2" 
    export ASSIMPWRAP_MATERIAL="$(assimpwrap-material)" 
    export ASSIMPWRAP_GGCTRL="$(assimpwrap-ggctrl)" 
    export-
    export-export
    env | grep ASSIMPWRAP
}


assimpwrap-run(){
    assimpwrap-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

    assimpwrap-export 
    $DEBUG $(assimpwrap-bin) $*  
}

assimpwrap--(){

    assimpwrap-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

    assimpwrap-install $*
}

assimpwrap-lldb(){
    DEBUG=lldb assimpwrap-run
}

assimpwrap-brun(){
   echo running from bdir not idir : no install needed, but must set library path
   local bdir=$(assimpwrap-bdir)
   DYLD_LIBRARY_PATH=$bdir $DEBUG $bdir/AssimpWrapTest 
}

assimpwrap-test(){
    local arg=$1
    assimpwrap-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

    assimpwrap-export $arg
    DEBUG=lldb assimpwrap-brun
}

assimpwrap-extra(){
   assimpwrap-test extra
}

assimpwrap-otool(){
   otool -L $(assimpwrap-bin)
}


