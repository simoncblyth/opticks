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
g4ok-test(){  g4ok-cd ; om- ; om-test $* ; }


