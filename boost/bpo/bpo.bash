# === func-gen- : boost/bpo/bpo fgp boost/bpo/bpo.bash fgn bpo fgh boost/bpo
bpo-src(){      echo boost/bpo/bpo.bash ; }
bpo-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(bpo-src)} ; }
bpo-vi(){       vi $(bpo-source) ; }
bpo-env(){      elocal- ; }
bpo-usage(){ cat << EOU

Boost Program Options
=======================

* http://www.boost.org/doc/libs/1_55_0/doc/html/program_options/tutorial.html

::

    delta:src blyth$ port contents boost | grep program_options | grep example
      /opt/local/share/doc/boost/libs/program_options/example/Jamfile.v2
      /opt/local/share/doc/boost/libs/program_options/example/custom_syntax.cpp
      /opt/local/share/doc/boost/libs/program_options/example/first.cpp
      /opt/local/share/doc/boost/libs/program_options/example/multiple_sources.cfg
      /opt/local/share/doc/boost/libs/program_options/example/multiple_sources.cpp
      /opt/local/share/doc/boost/libs/program_options/example/option_groups.cpp
      /opt/local/share/doc/boost/libs/program_options/example/options_description.cpp
      /opt/local/share/doc/boost/libs/program_options/example/real.cpp
      /opt/local/share/doc/boost/libs/program_options/example/regex.cpp
      /opt/local/share/doc/boost/libs/program_options/example/response_file.cpp
      /opt/local/share/doc/boost/libs/program_options/example/response_file.rsp

::

    clang++ -I/opt/local/include -L/opt/local/lib -lboost_program_options-mt multiple_sources.cpp

EOU
}

bpo-boost-prefix(){ echo /opt/local ; }

bpo-dir(){  echo $(local-base)/env/boost/bpo ; }
bpo-sdir(){ echo $(opticks-home)/boost/bpo ; }
bpo-idir(){ echo $(bpo-boost-prefix)/include/boost/program_options ; }

bpo-cd(){   cd $(bpo-dir); }
bpo-scd(){  cd $(bpo-sdir); }
bpo-icd(){  cd $(bpo-idir); }


bpo-example-dir(){ echo /opt/local/share/doc/boost/libs/program_options/example ; }
bpo-example-cd(){  cd $(bpo-example-dir) ; }
bpo-example-make(){  
   bpo-scd
   local cpp=${1:-Config.cpp}
   local bin=$(bpo-dir)/${cpp/.cpp}
   local cmd="clang++ -I/opt/local/include -L/opt/local/lib -lboost_program_options-mt $cpp -o $bin"
   echo $cmd
   eval $cmd
}

