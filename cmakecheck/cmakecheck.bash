# === func-gen- : cmakecheck/cmakecheck fgp cmakecheck/cmakecheck.bash fgn cmakecheck fgh cmakecheck
cmakecheck-src(){      echo cmakecheck/cmakecheck.bash ; }
cmakecheck-source(){   echo ${BASH_SOURCE:-$(env-home)/$(cmakecheck-src)} ; }
cmakecheck-vi(){       vi $(cmakecheck-source) ; }
cmakecheck-env(){      elocal- ; }
cmakecheck-usage(){ cat << EOU

::

    [blyth@ntugrid5 env]$ cmakecheck-configure


    CMake Error at /home/blyth/local/env/tools/cmake/cmake-3.5.2-Linux-x86_64/share/cmake-3.5/Modules/FindBoost.cmake:1657 (message):
      Unable to find the requested Boost libraries.

      Boost version: 1.41.0

      Boost include path: /usr/include

      Could not find the following Boost libraries:

              boost_log
              boost_log_setup

::

    [blyth@ntugrid5 env]$ cmakecheck-;cmakecheck-configure -DBOOST_ROOT=$(boost-prefix)

    -- Boost version: 1.60.0
    -- Found the following Boost libraries:
    --   system
    --   thread
    --   program_options
    --   log
    --   log_setup
    --   filesystem
    --   regex
    --   chrono
    --   date_time
    --   atomic




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

