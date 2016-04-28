# === func-gen- : cmakecheck/cmakecheck fgp cmakecheck/cmakecheck.bash fgn cmakecheck fgh cmakecheck
cmakecheck-src(){      echo cmakecheck/cmakecheck.bash ; }
cmakecheck-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cmakecheck-src)} ; }
cmakecheck-vi(){       vi $(cmakecheck-source) ; }
cmakecheck-env(){      elocal- ; }
cmakecheck-usage(){ cat << EOU

::

    cmakecheck-;cmakecheck-configure -DDUMP=OFF -DBOOST_ROOT=$(boost-prefix) -D


EOU
}

cmakecheck-dir(){ echo $(cmakecheck-sdir) ; }
cmakecheck-sdir(){ echo $(env-home)/cmakecheck  ; }
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
   cmake $(env-home)/cmakecheck $*
   cd $iwd
}

cmakecheck-configure(){
   cmakecheck-wipe
   cmakecheck-cmake $*
}

cmakecheck-txt(){ vi $(cmakecheck-sdir)/CMakeLists.txt ; }

