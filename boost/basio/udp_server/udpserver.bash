# === func-gen- : boost/basio/udp_server/udpserver fgp boost/basio/udp_server/udpserver.bash fgn udpserver fgh boost/basio/udp_server
udpserver-src(){      echo boost/basio/udp_server/udpserver.bash ; }
udpserver-source(){   echo ${BASH_SOURCE:-$(env-home)/$(udpserver-src)} ; }
udpserver-vi(){       vi $(udpserver-source) ; }
udpserver-env(){      elocal- ; }
udpserver-usage(){ cat << EOU

boost::asio UDP server
========================

Start server::

    udpserver-start

Test with python client::

    delta:~ blyth$ UDP_PORT=13 udp.py hello world
    sending [hello world] to host:port delta.local:13 


EOU
}

udpserver-sdir(){ echo $(env-home)/boost/basio/udp_server ; }
udpserver-idir(){ echo $(local-base)/env/boost/basio/udp_server ; }
udpserver-bdir(){ echo $(local-base)/env/boost/basio/udp_server.build ; }

udpserver-scd(){  cd $(udpserver-sdir); }
udpserver-icd(){  cd $(udpserver-idir); }
udpserver-bcd(){  cd $(udpserver-bdir); }


udpserver-start()
{
   local cmd="sudo $(udpserver-idir)/bin/UDPServer"
   echo $cmd
   eval $cmd
}

udpserver-test()
{
    UDP_PORT=13 udp.py ${1:-hello_world} 
}

udpserver-wipe(){
   local bdir=$(udpserver-bdir)
   rm -rf $bdir
}


udpserver-cmake(){
   local iwd=$PWD

   local bdir=$(udpserver-bdir)
   mkdir -p $bdir
  
   udpserver-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(udpserver-idir) \
       $(udpserver-sdir)

   cd $iwd
}

udpserver-make(){
   local iwd=$PWD

   udpserver-bcd 
   make $*

   cd $iwd
}

udpserver-install(){
   udpserver-make install
}

udpserver--()
{
    udpserver-wipe
    udpserver-cmake
    udpserver-make
    udpserver-install

}

