g4ok-source(){ echo $BASH_SOURCE ; }
g4ok-vi(){ vi $(g4ok-source) ; }
g4ok-env(){  olocal- ; opticks- ; }
g4ok-usage(){ cat << EOU
g4ok Usage 
===================

EOU
}

g4ok-dir(){ echo $(dirname $(g4ok-source)) ; }
g4ok-cd(){  cd $(g4ok-dir) ; }
g4ok-c(){   cd $(g4ok-dir) ; }
g4ok-bcd(){ cd $(g4ok-dir) ; om-cd ;  }

g4ok-conf(){  g4ok-cd ; om- ; om-conf $* ; }
g4ok-make(){  g4ok-cd ; om- ; om-make $* ; }
g4ok-test(){  g4ok-bcd ; om- ; om-test $* ; }

g4ok--(){   g4ok-make ; }
g4ok-t(){   g4ok-test ; }

g4ok-g(){ echo /usr/local/opticks/geocache/CX4GDMLTest_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/g4ok.gltf ; }
g4ok-gcd(){ cd $(dirname $(g4ok-g)) ; }
g4ok-ed(){  local g=$(g4ok-g) ; ls -l $g ; sleep 0.5 ; vi $g ; ls -l $g ; }

