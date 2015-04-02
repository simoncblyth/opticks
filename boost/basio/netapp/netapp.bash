# === func-gen- : boost/basio/udp_server/udpserver fgp boost/basio/udp_server/udpserver.bash fgn udpserver fgh boost/basio/udp_server
netapp-src(){      echo boost/basio/udp_server/netapp.bash ; }
netapp-source(){   echo ${BASH_SOURCE:-$(env-home)/$(netapp-src)} ; }
netapp-vi(){       vi $(netapp-source) ; }
netapp-env(){      elocal- ; }
netapp-usage(){ cat << EOU

boost::asio UDP server
========================

Start server::

    netapp-start

Test with python client::

    delta:~ blyth$ UDP_PORT=13 udp.py hello world
    sending [hello world] to host:port delta.local:13 


EOU
}

netapp-sdir(){ echo $(env-home)/boost/basio/netapp ; }
netapp-idir(){ echo $(local-base)/env/boost/basio/netapp ; }
netapp-bdir(){ echo $(local-base)/env/boost/basio/netapp.build ; }

netapp-cd(){   cd $(netapp-sdir); }
netapp-scd(){  cd $(netapp-sdir); }
netapp-icd(){  cd $(netapp-idir); }
netapp-bcd(){  cd $(netapp-bdir); }


netapp-start()
{
   local cmd="sudo $(netapp-idir)/bin/NetApp"
   echo $cmd
   eval $cmd
}

netapp-test()
{
    UDP_PORT=13 udp.py ${1:-hello_world} 
    npysend.sh --tag 1
}

netapp-wipe(){
   local bdir=$(netapp-bdir)
   rm -rf $bdir
}


netapp-cmake(){
   local iwd=$PWD

   local bdir=$(netapp-bdir)
   mkdir -p $bdir
  
   netapp-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(netapp-idir) \
       $(netapp-sdir)

   cd $iwd
}

netapp-make(){
   local iwd=$PWD

   netapp-bcd 
   make $*

   cd $iwd
}

netapp-install(){
   netapp-make install
}

netapp--()
{
    netapp-wipe
    netapp-cmake
    netapp-make
    netapp-install

}

