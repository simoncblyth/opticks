##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

numpyserver-source(){   echo $BASH_SOURCE ; }
numpyserver-vi(){       vi $(numpyserver-source) ; }
numpyserver-env(){      olocal- ; }
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
the type and instance of the delegate to be messaged.
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




structure
-----------

Object Constituents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    numpyserver<Delegate>

         m_io_service    
         m_io_service_work
         net_manager<Delegate>   m_net_manager       
  
              m_ctx  (ZMQ)
              m_local_io_service
              m_local_io_service_work
              m_work_thread
              m_udp_server     udp_server<Delegate> 
              m_npy_server     npy_server<Delegate> 

                    m_socket
                    m_buffer 
                    m_delegate               Delegate*
                    m_delegate_io_service    boost::asio::io_service&
                    m_metadata               std::string
                    m_shape                  std::vector<int>   
                    m_data                   std::vector<float> 
              
                    Posts to the delegates on_npy method                          





header only boost::asio and asio-zmq implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

numpyserver.hpp

     numpyserver<Delegate> 

     Controls main thread boost::asio::io_service 
     and holds constitent net_manager<Delegate> 

     Main thread io_service is polled to see if the 
     net thread has any messages pending.  

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


UDP Reply to sender
---------------------

While numpyserver OR glfwtest is running, sendrecv UDP.::


    delta:~ blyth$ udpr.py --sendrecv -- --yfov 100
    2015-04-06 13:02:36,462 __main__ INFO     sendrecv [--yfov 100] to ('delta.local', 8080) 
    2015-04-06 13:02:36,731 __main__ INFO     revcfrom [('10.44.187.243', 8080)] [Mon Apr  6 13:02:36 2015
    ] 
    2015-04-06 13:02:36,743 __main__ INFO     revcfrom [('10.44.187.243', 8080)] [--yfov 100 returned from numpydelegate ] 


The time string is returned immediately from within the 
worker thread, the other string comes from the delegate 
on the main thread doing a send within the *on_msg* handler. 

This works as the *on_msg* handler arguments include the 
address and port of the sender.
The advantage here of passing values around 
and not maintaining state is presumably to make 
more robust when have concurrency. 


Testing
----------

UDP just send test::

     UDP_PORT=8080 udp.py hello world

UDP testing with reply:: 

      UDP_PORT=8080 udpr.py hello

NPY/ZMQ test::

      npysend.sh 



issues
-------

FIXED npy_server Abort at exit 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing in ZMQ context ensures the context outlives the socket
and avoids abort at exit::

    delta:numpyserver blyth$ numpyserver
    libc++abi.dylib: terminating with uncaught exception of type boost::asio::zmq::exception: Socket operation on non-socket
    Abort trap: 6



EOU
}

numpyserver-sdir(){ echo $(opticks-home)/numpyserver ; }
numpyserver-idir(){ echo $(local-base)/opticks/numpyserver ; }
numpyserver-bdir(){ echo $(local-base)/opticks/numpyserver.build ; }

numpyserver-cd(){   cd $(numpyserver-sdir); }
numpyserver-scd(){  cd $(numpyserver-sdir); }
numpyserver-icd(){  cd $(numpyserver-idir); }
numpyserver-bcd(){  cd $(numpyserver-bdir); }

numpyserver-name(){ echo NumpyServer ; }
numpyserver-bin(){ echo $(numpyserver-idir)/bin/$(numpyserver-name)Test ; }

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

