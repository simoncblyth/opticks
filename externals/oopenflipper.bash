oopenflipper-src(){      echo externals/oopenflipper.bash ; }
oopenflipper-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(oopenflipper-src)} ; }
oopenflipper-vi(){       vi $(oopenflipper-source) ; }
oopenflipper-usage(){ cat << EOU

OpenFlipper
=============


EOU
}

oopenflipper-env(){  olocal- ; opticks- ; }

oopenflipper-info(){ cat << EOI

    name : $(oopenflipper-name)
    dist : $(oopenflipper-dist)



EOI
}


oopenflipper-vers(){ echo 3.1 ; }
oopenflipper-name(){ echo OpenFlipper-$(oopenflipper-vers) ; }
oopenflipper-url(){  echo http://www.openflipper.org/media/Releases/$(oopenflipper-vers)/$(oopenflipper-name).tar.gz ; }

oopenflipper-dist(){ echo $(dirname $(oopenflipper-dir))/$(basename $(oopenflipper-url)) ; }

oopenflipper-base(){ echo $(opticks-prefix)/externals/openflipper ; }
oopenflipper-prefix(){ echo $(opticks-prefix)/externals ; }
oopenflipper-idir(){ echo $(oopenflipper-prefix) ; }

oopenflipper-dir(){  echo $(oopenflipper-base)/$(oopenflipper-name) ; }
oopenflipper-bdir(){ echo $(oopenflipper-base)/$(oopenflipper-name).build ; }

oopenflipper-ecd(){  cd $(oopenflipper-edir); }
oopenflipper-cd(){   cd $(oopenflipper-dir)/$1 ; }
oopenflipper-bcd(){  cd $(oopenflipper-bdir); }
oopenflipper-icd(){  cd $(oopenflipper-idir); }


oopenflipper-get(){
   local dir=$(dirname $(oopenflipper-dir)) &&  mkdir -p $dir && cd $dir
   local url="$(oopenflipper-url)"
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxf $tgz 
}

oopenflipper-doc(){ oopenflipper-html ; }
oopenflipper-html(){ open $(oopenflipper-dir)/Documentation/index.html ; }

oopenflipper-find(){ oopenflipper-cd ; find src -type f -exec grep -H ${1:-DefaultTraits} {} \; ; }


oopenflipper-wipe(){
  local bdir=$(oopenflipper-bdir)
  rm -rf $bdir 

}

oopenflipper-edit(){ vi $(opticks-home)/cmake/Modules/FindOpenFlipper.cmake ; }

oopenflipper-cmake(){
  local iwd=$PWD
  local bdir=$(oopenflipper-bdir)
  mkdir -p $bdir

  [ -f "$bdir/CMakeCache.txt" ] && echo $msg already configured : oopenflipper-configure to reconfigure && return 

  oopenflipper-bcd


  # -G "$(opticks-cmake-generator)" \

  cmake $(oopenflipper-dir) \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$(oopenflipper-prefix) \
      -DBUILD_APPS=OFF 

  cd $iwd
}

oopenflipper-configure()
{
   oopenflipper-wipe
   oopenflipper-cmake $*
}


oopenflipper-make(){
  local iwd=$PWD
  oopenflipper-bcd

  cmake --build . --config Release --target ${1:-install}

  cd $iwd
}

oopenflipper--(){
  oopenflipper-get 
  oopenflipper-cmake
  oopenflipper-make install
}




