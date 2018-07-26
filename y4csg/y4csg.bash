y4csg-source(){ echo $BASH_SOURCE ; }
y4csg-vi(){ vi $(y4csg-source) ; }
y4csg-env(){  olocal- ; opticks- ; }
y4csg-usage(){ cat << EOU
y4csg Usage 
===================

EOU
}

y4csg-dir(){ echo $(dirname $(y4csg-source)) ; }
y4csg-cd(){  cd $(y4csg-dir) ; }
y4csg-c(){   cd $(y4csg-dir) ; }
y4csg-bcd(){ cd $(y4csg-dir) ; om-cd ;  }

y4csg-conf(){  y4csg-cd ; om- ; om-conf $* ; }
y4csg-make(){  y4csg-cd ; om- ; om-make $* ; }
y4csg-test(){  y4csg-bcd ; om- ; om-test $* ; }

y4csg--(){   y4csg-make ; }
y4csg-t(){   y4csg-test ; }

y4csg-gcd(){ cd $(dirname $(y4csg-g)) ; }

