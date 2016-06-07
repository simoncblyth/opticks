opticksdata-src(){      echo opticksdata.bash ; }
opticksdata-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opticksdata-src)} ; }
opticksdata-vi(){       vi $(opticksdata-source) ; }
opticksdata-usage(){ cat << EOU

Opticks Data
=============

From point of view of opticks regard this as another external ?

See Also
--------

* bitbucket- for notes on repo creation
* export- for the initial population

::

    simon:~ blyth$ export-;export-copy-
    mkdir -p /Users/blyth/opticksdata/export/DayaBay_VGDX_20140414-1300
    cp /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.dae /Users/blyth/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    mkdir -p /Users/blyth/opticksdata/export/Far_VGDX_20140414-1256
    cp /usr/local/env/geant4/geometry/export/Far_VGDX_20140414-1256/g4_00.dae /Users/blyth/opticksdata/export/Far_VGDX_20140414-1256/g4_00.dae
    mkdir -p /Users/blyth/opticksdata/export/Lingao_VGDX_20140414-1247
    cp /usr/local/env/geant4/geometry/export/Lingao_VGDX_20140414-1247/g4_00.dae /Users/blyth/opticksdata/export/Lingao_VGDX_20140414-1247/g4_00.dae
    mkdir -p /Users/blyth/opticksdata/export/LXe
    cp /usr/local/env/geant4/geometry/export/LXe/g4_00.dae /Users/blyth/opticksdata/export/LXe/g4_00.dae
    simon:~ blyth$ 
    simon:~ blyth$ export-;export-copy- | sh 



EOU
}
opticksdata-env(){      elocal- ; opticks- ;  }
opticksdata-dir(){ echo $(opticks-prefix)/opticksdata ; }
opticksdata-cd(){  cd $(opticksdata-dir); }

opticksdata-url(){ echo http://bitbucket.org/simoncblyth/opticksdata ; }

opticksdata-get(){
   local dir=$(dirname $(opticksdata-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(opticksdata-url)
   local nam=$(basename $url)
   [ ! -d "$nam" ] && hg clone $url 

}
