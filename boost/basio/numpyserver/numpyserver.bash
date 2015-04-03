numpyserver-src(){      echo boost/basio/numpyserver/numpyserver.bash ; }
numpyserver-source(){   echo ${BASH_SOURCE:-$(env-home)/$(numpyserver-src)} ; }
numpyserver-vi(){       vi $(numpyserver-source) ; }
numpyserver-env(){      elocal- ; }
numpyserver-usage(){ cat << EOU

NumpyServer 
============

Asynchronous ZMQ and UDP server with NPY serialization decoding.

Example usage, as in main.cpp::

   int main()
   {
       numpydelegate nde ; 
       numpyserver<numpydelegate> srv(&nde, 8080, "tcp://127.0.0.1:5002");

       for(unsigned int i=0 ; i < 20 ; ++i )
       {

           // do other things like updating OpenGL windows 

           srv.poll();
           srv.sleep(1);
       }
       srv.stop();

       return 0 ;
   } 

Instanciation of **numpyserver** spins off a background network thread 
that listens for UDP and ZMQ connections.
The **numpyserver** template type and first argument identify 
the type and instance of the delegate to me messaged.
Other arguments identify the UDP port and ZMQ backend endpoint.

When messages arrive the numpydelegate *on_msg* or *on_npy* 
handlers are invoked back on the main thread during
calls to one of the pollers:: 

Prerequisites, ZMQ broker needs to be running on the backend::

    zmq-
    zmq-broker  

Start server (more correctly the worker in ZMQ parlance)::

    numpyserver

Test::

    numpyserver-test



issues
-------

not cleaning up properly
~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:numpyserver blyth$ numpyserver
          0x7fff742b1310 numpyserver::stop 
    libc++abi.dylib: terminating with uncaught exception of type boost::asio::zmq::exception: Socket operation on non-socket
    Abort trap: 6




structure
-----------

example main and numpydelegate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

main.cpp
numpydelegate.cpp
numpydelegate.hpp


header only boost::asio and asio-zmq implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

numpyserver.hpp

     numpyserver<Delegate> 

     Controls main thread boost::asio::io_service 
     and holds constitent net_manager<Delegate> 

net_manager.hpp

     net_manager<Delegate>

     Instanciation creates the network thread and
     a private local_io_service used only by the 
     network thread.

     Holds constituents:
  
     npy_server<Delegate>  using ZMQ/asio-zmq 
     udp_server<Delegate>  using boost::asio::ip::udp

npy_server.hpp

      npy_server<Delegate>
   
      Holds pointer to delegate and associated delegate_io_service.
 
      Instanciation connects to ZMQ backend and queues
      an async_read_message  with handle_req.
      On receiving requests the asio-zmq buffer is decoded
      and the Numpy array data is posted to the delegate via
      delegate_io_service.

      Currently the buffer is just echoed in reply.

      HOW TO ARRANGE REPLY OF PROCESSED NPY ?

      TODO: reply with processed array, the appropriate way 
      to do this depends on how fast the processing turns
      out to be. 

      Could have a separate gpu thread and gpu_io_service 
      to handle the OptiX launches ? Need to work out how 
      to get the results back to sender.

udp_server.hpp

      udp_server<Delegate>

      Equivalent for UDP messages, with posts to delegate on_msg 
      Currently a datetime string reply is returned 

numpy.hpp

      Numpy array serialization/deserialization 


Testing
----------

UDP just send test::

     UDP_PORT=8080 udp.py hello world

UDP testing with reply:: 

      UDP_PORT=8080 udpr.py hello

NPY/ZMQ test::

      npysend.sh 

EOU
}

numpyserver-sdir(){ echo $(env-home)/boost/basio/numpyserver ; }
numpyserver-idir(){ echo $(local-base)/env/boost/basio/numpyserver ; }
numpyserver-bdir(){ echo $(local-base)/env/boost/basio/numpyserver.build ; }

numpyserver-cd(){   cd $(numpyserver-sdir); }
numpyserver-scd(){  cd $(numpyserver-sdir); }
numpyserver-icd(){  cd $(numpyserver-idir); }
numpyserver-bcd(){  cd $(numpyserver-bdir); }

numpyserver-name(){ echo NumpyServer ; }
numpyserver-bin(){ echo $(numpyserver-idir)/bin/$(numpyserver-name) ; }

numpyserver-srun()
{
   local cmd="sudo $(numpyserver-bin)"
   echo $cmd
   eval $cmd
}

numpyserver-run(){
  local bin=$(numpyserver-bin) 
  $bin $* 
}
numpyserver(){ $(numpyserver-bin) $* ; }
numpyserver-lldb(){ lldb $(numpyserver-bin) $* ; }

numpyserver-test()
{
    UDP_PORT=8080 udpr.py ${1:-hello_world} 
    npysend.sh --tag 1
}

numpyserver-wipe(){
   local bdir=$(numpyserver-bdir)
   rm -rf $bdir
}


numpyserver-cmake(){
   local iwd=$PWD

   local bdir=$(numpyserver-bdir)
   mkdir -p $bdir
  
   numpyserver-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(numpyserver-idir) \
       $(numpyserver-sdir)

   cd $iwd
}

numpyserver-make(){
   local iwd=$PWD

   numpyserver-bcd 
   make $*

   cd $iwd
}

numpyserver-install(){
   numpyserver-make install
}

numpyserver--()
{
    numpyserver-wipe
    numpyserver-cmake
    numpyserver-make
    numpyserver-install

}

