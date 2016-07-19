opticksdata-src(){      echo opticksdata.bash ; }
opticksdata-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(opticksdata-src)} ; }
opticksdata-vi(){       vi $(opticksdata-source) ; }
opticksdata-usage(){ cat << EOU

Opticks Data
=============

From point of view of opticks regard this as another external ?


Locations
-----------

~/opticksdata 
       opticksdata bitbucket clone used to upload geometry and other opticks data

/usr/local/opticks/opticksdata
       readonly opticksdata bitbucket clone that only pulls, this is what users see


FUNCTIONS
-----------

::

    === opticksdata-export-ini : writing OPTICKS_DAEPATH_ environment to /usr/local/opticks/opticksdata/config/opticksdata.ini
    OPTICKSDATA_DAEPATH_DPIB=/usr/local/opticks/opticksdata/export/dpib/cfg4.dae
    OPTICKSDATA_DAEPATH_DYB=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    OPTICKSDATA_DAEPATH_FAR=/usr/local/opticks/opticksdata/export/Far_VGDX_20140414-1256/g4_00.dae
    OPTICKSDATA_DAEPATH_JPMT=/usr/local/opticks/opticksdata/export/juno/test3.dae
    OPTICKSDATA_DAEPATH_LIN=/usr/local/opticks/opticksdata/export/Lingao_VGDX_20140414-1247/g4_00.dae
    OPTICKSDATA_DAEPATH_LXE=/usr/local/opticks/opticksdata/export/LXe/g4_00.dae


This .ini file is read by OpticksResource allowing opticks to access the .dae path 
at a higher level of just needing the tag.




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
   if [ ! -d "$nam/.hg" ]; then
        hg clone $url 
   else
        echo $msg ALREADY CLONED from $url to $(opticksdata-dir) 
   fi

}

opticksdata--()
{
   opticksdata-get
   opticksdata-export-ini
}

opticksdata-pull()
{
   local msg="$FUNCNAME :"
   local iwd=$PWD
   opticksdata-cd
   echo $msg PWD $PWD
   hg pull 
   hg up
   cd $iwd  
}


opticksdata-path(){ echo $(opticksdata-xpath $1).dae ; }
opticksdata-xpath(){
  local base=$(opticksdata-dir)/export
  case $1 in 
       dyb) echo $base/DayaBay_VGDX_20140414-1300/g4_00 ;;
      dlin) echo $base/Lingao_VGDX_20140414-1247/g4_00 ;;
      dfar) echo $base/Far_VGDX_20140414-1256/g4_00 ;;
      dpib) echo $base/dpib/cfg4 ;; 
       lxe) echo $base/LXe/g4_00 ;;
      jpmt) echo $base/juno/test3 ;;
  esac

#      dybf) echo $base/DayaBay_VGDX_20140414-1300/g4_00 ;;
#      jtst) echo $base/juno/test ;;
#      juno) echo $base/juno/nopmt ;;

}


opticksdata-tags-(){ cat << EOT
DYB
DFAR
DLIN
DPIB
JPMT
LXE
EOT
}

opticksdata-export(){
   local utag
   local ltag
   local path
   for utag in $(opticksdata-tags-) 
   do
      ltag=$(echo $utag | tr "A-Z" "a-z")
      path=$(opticksdata-path $ltag) 
      [ ! -f "$path" ] && echo $msg SKIP MISSING PATH for $utag && continue 
      printf "%5s %5s %s \n"  $utag $ltag $path
      export OPTICKSDATA_DAEPATH_$utag=$path
   done
}

opticksdata-dump(){
   env | grep OPTICKSDATA_DAEPATH_ | sort 
}

opticksdata-find(){    find ~/opticksdata -name '*.dae' -type f ; }
opticksdata-find-ls(){ find ~/opticksdata -name '*.dae' -type f -exec ls -l {} \; ; }
opticksdata-find-du(){ find ~/opticksdata -name '*.dae' -type f -exec du -h {} \; ; }


opticksdata-ini(){ echo $(opticks-prefix)/opticksdata/config/opticksdata.ini ; }
opticksdata-export-ini()
{
   local msg="=== $FUNCNAME :"

   opticksdata-export 

   local ini=$(opticksdata-ini)
   local dir=$(dirname $ini)
   mkdir -p $dir 

   echo $msg writing OPTICKS_DAEPATH_ environment to $ini
   env | grep OPTICKSDATA_DAEPATH_ | sort > $ini

   cat $ini
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


