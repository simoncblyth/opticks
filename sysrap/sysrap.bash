sysrap-src(){      echo sysrap/sysrap.bash ; }
sysrap-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(sysrap-src)} ; }
sysrap-vi(){       vi $(sysrap-source) ; }
sysrap-usage(){ cat << \EOU

System Rap
============

Lowest level package, beneath BoostRap and 
**explicitly not using Boost**. 

A lower level pkg that BoostRap is required 
as nvcc, the CUDA compiler, has trouble compiling 
some Boost headers.

EOU
}

sysrap-env(){      olocal- ; opticks- ;  }

sysrap-dir(){  echo $(sysrap-sdir) ; }
sysrap-sdir(){ echo $(opticks-home)/sysrap ; }
sysrap-tdir(){ echo $(opticks-home)/sysrap/tests ; }
sysrap-idir(){ echo $(opticks-idir); }
sysrap-bdir(){ echo $(opticks-bdir)/sysrap ; }

sysrap-cd(){   cd $(sysrap-sdir); }
sysrap-scd(){  cd $(sysrap-sdir); }
sysrap-tcd(){  cd $(sysrap-tdir); }
sysrap-icd(){  cd $(sysrap-idir); }
sysrap-bcd(){  cd $(sysrap-bdir); }

sysrap-name(){ echo SysRap ; }
sysrap-tag(){  echo SYSRAP ; }

sysrap-apihh(){  echo $(sysrap-sdir)/$(sysrap-tag)_API_EXPORT.hh ; }
sysrap---(){     touch $(sysrap-apihh) ; sysrap--  ; }


sysrap-wipe(){    local bdir=$(sysrap-bdir) ; rm -rf $bdir ; }

sysrap--(){       opticks-- $(sysrap-bdir) ; } 
sysrap-t(){       opticks-t $(sysrap-bdir) $* ; } 
sysrap-genproj(){ sysrap-scd ; opticks-genproj $(sysrap-name) $(sysrap-tag) ; } 
sysrap-gentest(){ sysrap-tcd ; opticks-gentest ${1:-SCheck} $(sysrap-tag) ; } 
sysrap-txt(){     vi $(sysrap-sdir)/CMakeLists.txt $(sysrap-tdir)/CMakeLists.txt ; } 

sysrap-csg(){ head -20 $(sysrap-dir)/OpticksCSG.h ; }

