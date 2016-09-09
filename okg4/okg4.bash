okg4-src(){      echo okg4/okg4.bash ; }
okg4-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(okg4-src)} ; }
okg4-vi(){       vi $(okg4-source) ; }
okg4-usage(){ cat << EOU

Integration of Opticks and Geant4 
===================================

EOU
}

okg4-env(){  
   olocal- 
   g4-
   opticks-
}



okg4-idir(){ echo $(opticks-idir); } 
okg4-bdir(){ echo $(opticks-bdir)/okg4 ; }
okg4-sdir(){ echo $(opticks-home)/okg4 ; }
okg4-tdir(){ echo $(opticks-home)/okg4/tests ; }

okg4-icd(){  cd $(okg4-idir); }
okg4-bcd(){  cd $(okg4-bdir); }
okg4-scd(){  cd $(okg4-sdir); }
okg4-tcd(){  cd $(okg4-tdir); }

okg4-dir(){  echo $(okg4-sdir) ; }
okg4-cd(){   cd $(okg4-dir); }


okg4-name(){ echo okg4 ; }
okg4-tag(){  echo OKG4 ; }

okg4-apihh(){  echo $(okg4-sdir)/$(okg4-tag)_API_EXPORT.hh ; }
okg4---(){     touch $(okg4-apihh) ; okg4--  ; }



okg4-wipe(){    local bdir=$(okg4-bdir) ; rm -rf $bdir ; } 

okg4--(){       opticks-- $(okg4-bdir) ; } 
okg4-t(){       opticks-t $(okg4-bdir) $* ; } 
okg4-genproj(){ okg4-scd ; oks- ; oks-genproj $(okg4-name) $(okg4-tag) ; } 
okg4-gentest(){ okg4-tcd ; oks- ; oks-gentest ${1:-CExample} $(okg4-tag) ; } 
okg4-txt(){     vi $(okg4-sdir)/CMakeLists.txt $(okg4-tdir)/CMakeLists.txt ; } 




