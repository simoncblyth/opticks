# this gets sourced by opticks-env from opticks-

bin-source(){ echo $BASH_SOURCE ; }
bin-dir(){    echo $(dirname $BASH_SOURCE) ; }
bin-cd(){     cd $(bin-dir) ; }
bin-vi(){     vi $BASH_SOURCE ; }
bin-env(){    echo -n ; }
bin-t(){      bin-cd ; om- ; om-test ; }
bin--(){     bin-cd ; om- ; om-- ; }



