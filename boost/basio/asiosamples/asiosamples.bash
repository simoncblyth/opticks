# === func-gen- : boost/basio/asiosamples/asiosamples fgp boost/basio/asiosamples/asiosamples.bash fgn asiosamples fgh boost/basio/asiosamples
asiosamples-src(){      echo boost/basio/asiosamples/asiosamples.bash ; }
asiosamples-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(asiosamples-src)} ; }
asiosamples-vi(){       vi $(asiosamples-source) ; }
asiosamples-env(){      elocal- ; }
asiosamples-usage(){ cat << EOU


ASIO Samples
===============

Examples (code samples) describing the construction of active objects on the
top of Boost.Asio. A code-based guide for client/server creation with usage of
active object pattern by means of Boost C++ Libraries.


See also
---------

* https://think-async.com/Asio/Examples



EOU
}
asiosamples-dir(){ echo $(local-base)/env/boost/basio/asiosamples ; }
asiosamples-cd(){  cd $(asiosamples-dir); }
asiosamples-mate(){ mate $(asiosamples-dir) ; }
asiosamples-get(){
   local dir=$(dirname $(asiosamples-dir)) &&  mkdir -p $dir && cd $dir

   git clone https://github.com/mabrarov/asio_samples asiosamples

}

asiosamples-find(){
   find $(asiosamples-dir) -type f -exec grep -H ${1:-post} {} \;
}
asiosamples-lfind(){
   find $(asiosamples-dir) -type f -exec grep -l ${1:-post} {} \;
}

