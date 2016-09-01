okg-src(){      echo opticksgeo/okg.bash ; }
okg-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(okg-src)} ; }
okg-vi(){       vi $(okg-source) ; }
okg-usage(){ cat << EOU



EOU
}

okg-env(){      olocal- ; opticks- ; }

okg-dir(){  echo $(opticks-home)/opticksgeo ; }
okg-sdir(){ echo $(opticks-home)/opticksgeo ; }
okg-tdir(){ echo $(opticks-home)/opticksgeo/tests ; }
okg-idir(){ echo $(opticks-idir); } 
okg-bdir(){ echo $(opticks-bdir)/opticksgeo ; }  

okg-cd(){   cd $(okg-dir); }
okg-icd(){  cd $(okg-idir); }
okg-bcd(){  cd $(okg-bdir); }
okg-scd(){  cd $(okg-sdir); }
okg-tcd(){  cd $(okg-tdir); }

okg-wipe(){ local bdir=$(okg-bdir) ; rm -rf $bdir ; }

okg-name(){ echo OpticksGeometry ; }
okg-tag(){  echo OKGEO ; }

okg-apihh(){  echo $(okg-sdir)/$(okg-tag)_API_EXPORT.hh ; }
okg---(){     touch $(okg-apihh) ; okg--  ; }


okg--(){        opticks--     $(okg-bdir) ; }
okg-ctest(){    opticks-ctest $(okg-bdir) $* ; }
okg-genproj(){  okg-scd ; opticks-genproj $(okg-name) $(okg-tag) ; }
okg-gentest(){  okg-tcd ; opticks-gentest ${1:-OpticksGeometry} $(okg-tag) ; }

okg-txt(){   vi $(okg-sdir)/CMakeLists.txt $(okg-tdir)/CMakeLists.txt ; }


