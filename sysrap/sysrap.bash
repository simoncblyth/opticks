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

sysrap-c(){    cd $(sysrap-sdir); }
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
sysrap-ts(){      opticks-ts $(sysrap-bdir) $* ; } 
sysrap-genproj(){ sysrap-scd ; opticks-genproj $(sysrap-name) $(sysrap-tag) ; } 
sysrap-gentest(){ sysrap-tcd ; opticks-gentest ${1:-SCheck} $(sysrap-tag) ; } 
sysrap-txt(){     vi $(sysrap-sdir)/CMakeLists.txt $(sysrap-tdir)/CMakeLists.txt ; } 

sysrap-csg(){ head -20 $(sysrap-dir)/OpticksCSG.h ; }


sysrap-csg-generate()
{
    local msg="$FUNCNAME : " 
    local iwd=$PWD
    sysrap-cd
    c_enums_to_python.py OpticksCSG.h 

    echo $msg To write above generated python to OpticksCSG.py ..

    local ans
    read -p "Enter YES ... " ans

    if [  "$ans" == "YES" ]; then 
       c_enums_to_python.py OpticksCSG.h > OpticksCSG.py 

       echo $msg checking the generated python is valid 
       python  OpticksCSG.py

    else
       echo $msg SKIP
    fi 

    cd $iwd
}


