brap-rel(){      echo boostrap ; }
brap-src(){      echo boostrap/brap.bash ; }
brap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(brap-src)} ; }
brap-vi(){       vi $(brap-source) ; }
brap-usage(){ cat << EOU


BoostRap
===========

Windows VS2015
---------------

Testing BLog.cc alone get link failure::


    "C:\usr\local\opticks\build\boostrap\BoostRap.vcxproj" (default target) (4) ->
    (Link target) ->
      LINK : fatal error LNK1104: cannot open file 'boost_log-vc140-mt-gd-1_61.lib' 
             [C:\usr\local\opticks\build\boostrap\BoostRap.vcxproj]

        6 Warning(s)
        1 Error(s)


Changing the config to Release in the brap-- "make" leads to missing another.

    "C:\usr\local\opticks\build\boostrap\BoostRap.vcxproj" (default target) (4) ->
    (Link target) ->
      LINK : fatal error LNK1104: cannot open file 'boost_log-vc140-mt-1_61.lib'
       [C:\usr\local\opticks\build\boostrap\BoostRap.vcxproj]


Where did the un-lib suffixed name come from ?
Found the cause BOOST_LOG_DYN_LINK in FindOpticksBoost.cmake::

     35     if(Boost_FOUND)
     36         set_property(GLOBAL PROPERTY gOpticksBoost_FOUND "YES")
     37         set_property(GLOBAL PROPERTY gOpticksBoost_LIBRARIES    ${Boost_LIBRARIES})
     38         set_property(GLOBAL PROPERTY gOpticksBoost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
     39      #   set_property(GLOBAL PROPERTY gOpticksBoost_DEFINITIONS  ${Boost_DEFINITIONS} -DBOOST_LOG_DYN_LINK)
     40     endif(Boost_FOUND)



BoostRap:LIBRARIES resulting from the cmake find are lib prefixed (that means static?) 
and all actually exist::

      optimized;C:/usr/local/opticks/externals/lib/libboost_system-vc140-mt-1_61.lib;
          debug;C:/usr/local/opticks/externals/lib/libboost_system-vc140-mt-gd-1_61.lib;
      optimized;C:/usr/local/opticks/externals/lib/libboost_thread-vc140-mt-1_61.lib;
          debug;C:/usr/local/opticks/externals/lib/libboost_thread-vc140-mt-gd-1_61.lib;
      optimized;C:/usr/local/opticks/externals/lib/libboost_program_options-vc140-mt-1_61.lib;
          debug;C:/usr/local/opticks/externals/lib/libboost_program_options-vc140-mt-gd-1_61.lib;
      optimized;C:/usr/local/opticks/externals/lib/libboost_log-vc140-mt-1_61.lib;
          debug;C:/usr/local/opticks/externals/lib/libboost_log-vc140-mt-gd-1_61.lib;
      optimized;C:/usr/local/opticks/externals/lib/libboost_log_setup-vc140-mt-1_61.lib;
          debug;C:/usr/local/opticks/externals/lib/libboost_log_setup-vc140-mt-gd-1_61.lib;
      optimized;C:/usr/local/opticks/externals/lib/libboost_filesystem-vc140-mt-1_61.lib;
          debug;C:/usr/local/opticks/externals/lib/libboost_filesystem-vc140-mt-gd-1_61.lib;
      optimized;C:/usr/local/opticks/externals/lib/libboost_regex-vc140-mt-1_61.lib;
          debug;C:/usr/local/opticks/externals/lib/libboost_regex-vc140-mt-gd-1_61.lib;
      optimized;C:/usr/local/opticks/externals/lib/libboost_chrono-vc140-mt-1_61.lib;
          debug;C:/usr/local/opticks/externals/lib/libboost_chrono-vc140-mt-gd-1_61.lib;
      optimized;C:/usr/local/opticks/externals/lib/libboost_date_time-vc140-mt-1_61.lib;
          debug;C:/usr/local/opticks/externals/lib/libboost_date_time-vc140-mt-gd-1_61.lib;
      optimized;C:/usr/local/opticks/externals/lib/libboost_atomic-vc140-mt-1_61.lib;
          debug;C:/usr/local/opticks/externals/lib/libboost_atomic-vc140-mt-gd-1_61.lib







*brap-test*
    for developing boost regex patterns, via search matching against cin

    ::

        brap-test "<[^>]*>"
        brap-test "\w*\d"


::

    simon:~ blyth$ brap-test "\"(.*?)\""
    search cin for text matching regex "(.*?)"
    pairs plucked using regexp : 2
                            path.h :           
                      other_path.h :         


*brap-enum*
    plucking names and values from an enum string

    ::
       
        brap-enum "^\s*(\\w+)\s*=\s*(.*?),*\s*?$"


::

    simon:thrust blyth$ brap-flags
    extract enum pairs from file $ENV_HOME/graphics/ggeoview/cu/photon.h
    udump : 13
                          CERENKOV : 1 :          1 :          1
                     SCINTILLATION : 2 :          2 :          2
                              MISS : 3 :          4 :          4
                       BULK_ABSORB : 4 :          8 :          8
                       BULK_REEMIT : 5 :         16 :         10
                      BULK_SCATTER : 6 :         32 :         20
                    SURFACE_DETECT : 7 :         64 :         40
                    SURFACE_ABSORB : 8 :        128 :         80
                  SURFACE_DREFLECT : 9 :        256 :        100
                  SURFACE_SREFLECT : a :        512 :        200
                  BOUNDARY_REFLECT : b :       1024 :        400
                 BOUNDARY_TRANSMIT : c :       2048 :        800
                         NAN_ABORT : d :       4096 :       1000
    simon:thrust blyth$ 



EOU
}

brap-sdir(){ echo $(env-home)/boostrap ; }




brap-dir(){  echo $(brap-sdir) ; }

brap-env(){      elocal- ; opticks- ;  }

brap-name(){ echo BoostRap ; }

brap-idir(){ echo $(opticks-idir); }
brap-bdir(){ echo $(opticks-bdir)/$(brap-rel) ; }

brap-scd(){  cd $(brap-sdir); }
brap-cd(){  cd $(brap-sdir); }

brap-icd(){  cd $(brap-idir); }
brap-bcd(){  cd $(brap-bdir); }



brap-flags(){ $(brap-idir)/bin/enum_regexsearchTest $* ; }

brap-wipe(){
   local bdir=$(brap-bdir)
   rm -rf $bdir
}


brap-txt(){ vi $(brap-dir)/CMakeLists.txt ; }
brap-make(){
   local iwd=$PWD
   brap-bcd
   make $*
   cd $iwd
}


brap-generate-exports(){

   brap-scd 
   importlib-
   importlib-exports BoostRap BRAP_API
}



brap-install(){
   brap-make install
}

brap-bin(){ echo $(brap-idir)/bin ; }
brap-export()
{
   echo -n
}

brap-run(){
   local bin=$(brap-bin)
   brap-export
   $bin $*
}

brap-lldb()
{
   local bin=$(brap-bin)
   brap-export
   lldb $bin -- $*
}

brap-full()
{
    brap-make clean
    brap-make
    brap-install
}




brap-libdir(){ echo $(brap-bdir)/$(brap-config) ; }

brap-config(){ echo Debug ; }
brap--()
{
   local iwd=$PWD
   brap-bcd ; 
   #make ${1:-install}  
   cmake \
          --build . \
          --config $(brap-config) \
          --target ${1:-install} 

   cd $iwd
}

brap-ctest(){ opticks-ctest $(brap-bdir) $* ; }

brap-hello()
{
   echo "<hello>world</hello>" | $(brap-bin)/regexsearchTest 
}

brap-test(){ $FUNCNAME- | $(brap-bin)/regexsearchTest "$1" ; }
brap-test-(){ cat << EOT
   <hello>world1</hello>
   <hello>world2</hello>
   <hello>world3</hello>
   <hello>world4</hello>

    \$ENV_HOME/graphics/ggeoview/cu/photon.h

    \$HOME/.opticks/GColors.json
    \$HOME/.opticks
    \$HOME/opticks/GColors.json


#incl "path.h"
#incl "other_path.h"

    __dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface 

EOT
}






brap-enum(){ $FUNCNAME- | $(brap-bin)/regexsearchTest "$1" ; }
brap-enum-(){ cat << EOE
enum
{
    NO_HIT                 = 0x1 << 0,
    BULK_ABSORB            = 0x1 << 1,
    SURFACE_DETECT         = 0x1 << 2,
    SURFACE_ABSORB         = 0x1 << 3,
    RAYLEIGH_SCATTER       = 0x1 << 4,
    REFLECT_DIFFUSE        = 0x1 << 5,
    REFLECT_SPECULAR       = 0x1 << 6,
    SURFACE_REEMIT         = 0x1 << 7,
    SURFACE_TRANSMIT       = 0x1 << 8,
    BULK_REEMIT            = 0x1 << 9,
    GENERATE_SCINTILLATION = 0x1 << 16, 
    GENERATE_CERENKOV      = 0x1 << 17, 
    BOUNDARY_SPOL          = 0x1 << 18, 
    BOUNDARY_PPOL          = 0x1 << 19, 
    BOUNDARY_REFLECT       = 0x1 << 20, 
    BOUNDARY_TRANSMIT      = 0x1 << 21, 
    BOUNDARY_TIR           = 0x1 << 22, 
    BOUNDARY_TIR_NOT       = 0x1 << 23, 
    NAN_ABORT              = 0x1 << 31
}; // processes

EOE
}



brap-photon(){  $(brap-bin)/enum_regexsearchTest \$ENV_HOME/graphics/ggeoview/cu/photon.h ; }

brap-hpp2hh()
{
   local msg="=== $FUNCNAME :"
   local path
   
   local cls=${1:-stringutil} 

   grep -l $cls\.hpp *.* | while read path 
   do 
      echo $msg $path
      perl -pi -e "s,$cls\.hpp,$cls.hh,mg" $path
   done
}


brap-hpp2hh-all()
{
   local cls=${1:-stringutil} 
   local iwd=$PWD
   local msg="=== $FUNCNAME : "
   opticks-
   local dir
   local base=$(opticks-home)
   opticks-dirs | while read dir 
   do
      cd ${base}/${dir}       &&  brap-hpp2hh $cls || echo $msg missing dir $dir
      cd ${base}/${dir}/tests  && brap-hpp2hh $cls || echo $msg missing dir ${dir}/tests 
   done
   cd $iwd
}





brap-recfg()
{
   local msg="=== $FUNCNAME :"
   local path
   grep -l \"Cfg\.hh\" *.* | while read path 
   do 
      echo $msg $path
      perl -pi -e 's,\"Cfg\.hh\","BCfg.hh",mg' $path
   done
}

brap-vicfg(){ cat << \EOU

To replace the Cfg class for BCfg use cmds like below in bash and vim:: 

    vi $(grep -l \ Cfg *.*)

    .,$s/ Cfg/ BCfg/gc

EOU
 


}
