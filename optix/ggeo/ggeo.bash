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
ggeo-scd(){  cd $(ggeo-sdir)/$1; }
ggeo-tcd(){  cd $(ggeo-tdir) ; }
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

ggeo-gentest()
{
   local iwd=$PWD
   ggeo-scd tests
   local cls=${1:-GMaterial}
   opticks-gentest $cls $(ggeo-tag) 
   cd $iwd
}



ggeo--(){  opticks-- $(ggeo-bdir) ; }
ggeo-txt(){ vi $(ggeo-sdir)/CMakeLists.txt $(ggeo-tdir)/CMakeLists.txt ; }


ggeo-ctest(){ opticks-ctest $(ggeo-bdir) $* ; }

   
ggeo-sln(){ echo $(ggeo-bdir)/$(ggeo-name).sln ; }
ggeo-slnw(){ vs- ; echo $(vs-wp $(ggeo-sln)) ; }

ggeo-vs(){ cat << EOC
  
# paste the below into PowerShell after checking profile with vip

vs-export

devenv /useenv $(ggeo-slnw)

 
EOC
}
