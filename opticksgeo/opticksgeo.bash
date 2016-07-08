opticksgeo-src(){      echo opticksgeo/opticksgeo.bash ; }
opticksgeo-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(opticksgeo-src)} ; }
opticksgeo-vi(){       vi $(opticksgeo-source) ; }
opticksgeo-usage(){ cat << EOU



EOU
}

opticksgeo-env(){      olocal- ; opticks- ; }

opticksgeo-dir(){  echo $(opticks-home)/opticksgeo ; }
opticksgeo-sdir(){ echo $(opticks-home)/opticksgeo ; }
opticksgeo-tdir(){ echo $(opticks-home)/opticksgeo/tests ; }
opticksgeo-idir(){ echo $(opticks-idir); } 
opticksgeo-bdir(){ echo $(opticks-bdir)/opticksgeo ; }  

opticksgeo-cd(){   cd $(opticksgeo-dir); }
opticksgeo-icd(){  cd $(opticksgeo-idir); }
opticksgeo-bcd(){  cd $(opticksgeo-bdir); }
opticksgeo-scd(){  cd $(opticksgeo-sdir); }
opticksgeo-tcd(){  cd $(opticksgeo-tdir); }

opticksgeo-wipe(){ local bdir=$(opticksgeo-bdir) ; rm -rf $bdir ; }

opticksgeo-name(){ echo OpticksGeometry ; }
opticksgeo-tag(){  echo OKGEO ; }

opticksgeo-apihh(){  echo $(opticksgeo-sdir)/$(opticksgeo-tag)_API_EXPORT.hh ; }
opticksgeo---(){     touch $(opticksgeo-apihh) ; opticksgeo--  ; }


opticksgeo--(){        opticks--     $(opticksgeo-bdir) ; }
opticksgeo-ctest(){    opticks-ctest $(opticksgeo-bdir) $* ; }
opticksgeo-genproj(){  opticksgeo-scd ; opticks-genproj $(opticksgeo-name) $(opticksgeo-tag) ; }
opticksgeo-gentest(){  opticksgeo-tcd ; opticks-gentest ${1:-OpticksGeometry} $(opticksgeo-tag) ; }

opticksgeo-txt(){   vi $(opticksgeo-sdir)/CMakeLists.txt $(opticksgeo-tdir)/CMakeLists.txt ; }


