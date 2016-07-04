opticksdata-src(){      echo opticksdata.bash ; }
opticksdata-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(opticksdata-src)} ; }
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
opticksdata-env(){      olocal- ; opticks- ;  }
opticksdata-dir(){ echo $(opticks-prefix)/opticksdata ; }
opticksdata-cd(){  cd $(opticksdata-dir); }

opticksdata-url(){ echo http://bitbucket.org/simoncblyth/opticksdata ; }

opticksdata-get(){
   local msg="$FUNCNAME :"
   local dir=$(dirname $(opticksdata-dir)) &&  mkdir -p $dir && cd $dir

   local url=$(opticksdata-url)
   local nam=$(basename $url)
   if [ ! -d "$nam" ]; then
        hg clone $url 
   else
        echo $msg cloned from $url already  
   fi

}

opticksdata--()
{
   opticksdata-get
}


opticksdata-name(){ echo $(opticksdata-xname $1).dae ; }
opticksdata-xname(){
  local base=$(opticksdata-dir)/export
  case $1 in 
       dyb) echo $base/DayaBay_VGDX_20140414-1300/g4_00 ;;
       dybf) echo $base/DayaBay_VGDX_20140414-1300/g4_00 ;;
       dpib) echo $base/dpib/cfg4 ;; 
       far) echo $base/Far_VGDX_20140414-1256/g4_00 ;;
    lingao) echo $base/Lingao_VGDX_20140414-1247/g4_00 ;;
       lxe) echo $base/LXe/g4_00 ;;
       jpmt) echo $base/juno/test3 ;;
       juno) echo $base/juno/nopmt ;;
       jtst) echo $base/juno/test ;;
  esac
}



opticksdata-export(){
   export DAE_NAME=$(opticksdata-name dyb)
   export DAE_NAME_DYB=$(opticksdata-name dyb)
   export DAE_NAME_DPIB=$(opticksdata-name dpib)
}


opticksdata-export-ps(){  

   vs- 
   cat << EOP

# paste into powershell profile


\$env:DAE_NAME = "$(vs-wp $(opticksdata-name dyb))"
\$env:DAE_NAME_DYB = "$(vs-wp $(opticksdata-name dyb))"
\$env:DAE_NAME_DPIB = "$(vs-wp $(opticksdata-name dpib))"


EOP
}




