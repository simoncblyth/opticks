ggeo-rel(){      echo optix/ggeo ; }
ggeo-src(){      echo optix/ggeo/ggeo.bash ; }
ggeo-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ggeo-src)} ; }
ggeo-vi(){       vi $(ggeo-source) ; }
ggeo-usage(){ cat << \EOU


EOU
}

ggeo-env(){      elocal- ; opticks- ; }

ggeo-idir(){ echo $(opticks-idir); } 
ggeo-bdir(){ echo $(opticks-bdir)/$(ggeo-rel) ; }  

ggeo-sdir(){ echo $(env-home)/optix/ggeo ; }
ggeo-tdir(){ echo $(env-home)/optix/ggeo/tests ; }

ggeo-icd(){  cd $(ggeo-idir); }
ggeo-bcd(){  cd $(ggeo-bdir); }
ggeo-scd(){  cd $(ggeo-sdir); }
ggeo-cd(){  cd $(ggeo-sdir); }

ggeo-wipe(){
    local bdir=$(ggeo-bdir)
    rm -rf $bdir
}


ggeo-name(){ echo GGeo ; }
ggeo-tag(){  echo GGEO ; }
ggeo-genproj()
{
   ggeo-scd
   opticks-genproj $(ggeo-name) $(ggeo-tag) 
}


ggeo-config(){ echo Debug ; }
ggeo--(){
   local iwd=$PWD;
   ggeo-bcd;
   cmake --build . --config $(ggeo-config) --target ${1:-install};
   cd $iwd
}


   


