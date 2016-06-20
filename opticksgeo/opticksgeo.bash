opticksgeo-src(){      echo opticksgeo/opticksgeo.bash ; }
opticksgeo-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opticksgeo-src)} ; }
opticksgeo-vi(){       vi $(opticksgeo-source) ; }
opticksgeo-usage(){ cat << EOU



EOU
}

opticksgeo-env(){      elocal- ; opticks- ; }

opticksgeo-dir(){  echo $(env-home)/opticksgeo ; }
opticksgeo-sdir(){ echo $(env-home)/opticksgeo ; }
opticksgeo-idir(){ echo $(opticks-idir); } 
opticksgeo-bdir(){ echo $(opticks-bdir)/opticksgeo ; }  

opticksgeo-cd(){   cd $(opticksgeo-dir); }
opticksgeo-icd(){  cd $(opticksgeo-idir); }
opticksgeo-bcd(){  cd $(opticksgeo-bdir); }
opticksgeo-scd(){  cd $(opticksgeo-sdir); }

opticksgeo-wipe(){
    local bdir=$(opticksgeo-bdir)
    rm -rf $bdir
}

opticksgeo-config(){ echo Debug ; }
opticksgeo--()
{
   local iwd=$PWD;
   opticksgeo-bcd;
   cmake --build . --config $(opticksgeo-config) --target ${1:-install};
   cd $iwd
}



