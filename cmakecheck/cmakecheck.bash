# === func-gen- : cmakecheck/cmakecheck fgp cmakecheck/cmakecheck.bash fgn cmakecheck fgh cmakecheck
cmakecheck-src(){      echo cmakecheck/cmakecheck.bash ; }
cmakecheck-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(cmakecheck-src)} ; }
cmakecheck-vi(){       vi $(cmakecheck-source) ; }
cmakecheck-env(){      olocal- ; }
cmakecheck-usage(){ cat << EOU

::

    cmakecheck-;cmakecheck-configure -DDUMP=OFF -DBOOST_ROOT=$(boost-prefix) -D


EOU
}

cmakecheck-dir(){ echo $(cmakecheck-sdir) ; }
cmakecheck-sdir(){ echo $(opticks-home)/cmakecheck  ; }
cmakecheck-bdir(){ echo /tmp/opticks/cmakecheck  ; }

cmakecheck-cd(){  cd $(cmakecheck-dir); }
cmakecheck-scd(){  cd $(cmakecheck-sdir) ; }
cmakecheck-bcd(){  cd $(cmakecheck-bdir) ; }

cmakecheck-wipe(){  rm -rf $(cmakecheck-bdir) ; }

cmakecheck-cmake(){
   local iwd=$PWD
   local bdir=$(cmakecheck-bdir)
   mkdir -p $bdir
   cmakecheck-bcd
   cmake $(opticks-home)/cmakecheck $*
   cd $iwd
}

cmakecheck-configure(){
   cmakecheck-wipe
   cmakecheck-cmake $*
}

cmakecheck-txt(){ vi $(cmakecheck-sdir)/CMakeLists.txt ; }

