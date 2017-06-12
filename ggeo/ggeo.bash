ggeo-rel(){      echo ggeo ; }
ggeo-src(){      echo ggeo/ggeo.bash ; }
ggeo-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ggeo-src)} ; }
ggeo-vi(){       vi $(ggeo-source) ; }
ggeo-usage(){ cat << \EOU


EOU
}

ggeo-env(){      olocal- ; opticks- ; }

ggeo-idir(){ echo $(opticks-idir); } 
ggeo-bdir(){ echo $(opticks-bdir)/$(ggeo-rel) ; }  

ggeo-sdir(){ echo $(opticks-home)/ggeo ; }
ggeo-tdir(){ echo $(opticks-home)/ggeo/tests ; }

ggeo-icd(){  cd $(ggeo-idir); }
ggeo-bcd(){  cd $(ggeo-bdir); }
ggeo-scd(){  cd $(ggeo-sdir)/$1; }
ggeo-tcd(){  cd $(ggeo-tdir) ; }
ggeo-cd(){  cd $(ggeo-sdir); }

ggeo-wipe(){
    local bdir=$(ggeo-bdir)
    rm -rf $bdir
}

ggeo-name(){ echo GGeo ; }
ggeo-tag(){  echo GGEO ; }

ggeo-apihh(){  echo $(ggeo-sdir)/$(ggeo-tag)_API_EXPORT.hh ; }
ggeo---(){     touch $(ggeo-apihh) ; ggeo--  ; }

ggeo--(){                   opticks-- $(ggeo-bdir) ; }
ggeo-t(){                   opticks-t $(ggeo-bdir) $* ; }

ggeo-genproj() { ggeo-scd ; opticks-genproj $(ggeo-name) $(ggeo-tag) ; }
ggeo-gentest() { ggeo-tcd ; opticks-gentest ${1:-GExample} $(ggeo-tag) ; }
ggeo-txt(){ vi $(ggeo-sdir)/CMakeLists.txt $(ggeo-tdir)/CMakeLists.txt ; }

   
ggeo-sln(){ echo $(ggeo-bdir)/$(ggeo-name).sln ; }
ggeo-slnw(){ vs- ; echo $(vs-wp $(ggeo-sln)) ; }
ggeo-vs(){  opticks-vs $(ggeo-sln) ; }


