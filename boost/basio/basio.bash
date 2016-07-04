# === func-gen- : boost/basio/basio fgp boost/basio/basio.bash fgn basio fgh boost/basio
basio-src(){      echo boost/basio/basio.bash ; }
basio-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(basio-src)} ; }
basio-vi(){       vi $(basio-source) ; }
basio-env(){      olocal- ; }
basio-usage(){ cat << EOU

Boost ASIO
===========

* http://www.boost.org/doc/libs/1_57_0/doc/html/boost_asio.html
* http://www.boost.org/doc/libs/1_57_0/doc/html/boost_asio/overview/rationale.html
* http://think-async.com/Asio/


Sometimes, as is the case with networking, individual I/O operations 
can take a long time to complete. 
Boost.Asio provides the tools to manage these long running operations, 
without requiring programs to use concurrency models based on threads 
and explicit locking.


Conceptual Intro
----------------

* file:///opt/local/share/doc/boost/doc/html/boost_asio/overview/core/basics.html

Talks
-------

* :google:`Thinking Asynchronously Christopher Kohlhoff`

* https://www.youtube.com/watch?v=D-lTwGJRx0o

  * at about 57min, description of posting between threads

* http://theboostcpplibraries.com/boost.asio

Refs
-----

* http://www.digitalpeer.com/blog/asio-is-my-go-to-cpp-application-framework


Message passing from netThread to mainThread
----------------------------------------------

* http://stackoverflow.com/questions/17976568/boost-asio-pattern-with-gui-and-worker-thread

* http://stackoverflow.com/questions/17311512/howto-post-messages-between-threads-with-boostasio/17315567#17315567

* http://stackoverflow.com/questions/13680770/running-a-function-on-the-main-thread-from-a-boost-thread-and-passing-parameters

* http://stackoverflow.com/questions/13785640/using-boostasioio-servicepost

* http://stackoverflow.com/search?q=asio+io_service+post

* http://www.boost.org/doc/libs/1_45_0/doc/html/boost_asio/example/services/logger_service.hpp


Distributed Examples
----------------------

* https://think-async.com/Asio/Examples

* file:///opt/local/share/doc/boost/doc/html/boost_asio/examples/cpp03_examples.html

::

    basio-example-cd

::

    delta:cpp03 blyth$ basio-example-find post
    /opt/local/share/doc/boost/libs/asio/example/cpp03/chat/chat_client.cpp:    io_service_.post(boost::bind(&chat_client::do_write, this, msg));
    /opt/local/share/doc/boost/libs/asio/example/cpp03/chat/chat_client.cpp:    io_service_.post(boost::bind(&chat_client::do_close, this));
    /opt/local/share/doc/boost/libs/asio/example/cpp03/invocation/prioritised_handlers.cpp:  io_service.post(pri_queue.wrap(0, low_priority_handler));
    /opt/local/share/doc/boost/libs/asio/example/cpp03/serialization/connection.hpp:      socket_.get_io_service().post(boost::bind(handler, error));
    /opt/local/share/doc/boost/libs/asio/example/cpp03/services/logger_service.hpp:    work_io_service_.post(boost::bind(
    /opt/local/share/doc/boost/libs/asio/example/cpp03/services/logger_service.hpp:    work_io_service_.post(boost::bind(
    /opt/local/share/doc/boost/libs/asio/example/cpp03/windows/transmit_file.cpp:    // to be posted. When complete() is called, ownership of the OVERLAPPED-
    /opt/local/share/doc/boost/libs/asio/example/cpp11/chat/chat_client.cpp:    io_service_.post(
    /opt/local/share/doc/boost/libs/asio/example/cpp11/chat/chat_client.cpp:    io_service_.post([this]() { socket_.close(); });
    delta:cpp03 blyth$ 



cpp03/chat/chat_client.cpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* file:///opt/local/share/doc/boost/doc/html/boost_asio/example/cpp03/chat/chat_client.cpp

* single io_service from main
* chat_client object 

  * public methods write(msg) and close() 
    post arguments to the private methods 
    enabling these private methods to run on network thread

  * simple, as only the network thread has called io_service.run() so
    are sure that the task will run on the network thread

  * what this misses is how to get results back from the network thread


cpp03/serialization/connection.hpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Use of post with threads ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    delta:cpp03 blyth$ basio-example-lfind thread | grep -v Jam
    /opt/local/share/doc/boost/libs/asio/example/cpp03/chat/chat_client.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/echo/blocking_tcp_echo_server.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/fork/daemon.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/fork/process_per_connection.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/http/server2/io_service_pool.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/http/server2/main.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/http/server3/main.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/http/server3/server.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/http/server3/server.hpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/local/connect_pair.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/porthopper/client.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/services/logger_service.hpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/timeouts/blocking_tcp_client.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/timeouts/blocking_udp_client.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/tutorial/timer5/timer.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/tutorial/timer_dox.txt
    /opt/local/share/doc/boost/libs/asio/example/cpp11/chat/chat_client.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp11/echo/blocking_tcp_echo_server.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp11/futures/daytime_client.cpp

    delta:cpp03 blyth$ basio-example-lfind post
    /opt/local/share/doc/boost/libs/asio/example/cpp03/chat/chat_client.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/invocation/prioritised_handlers.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/serialization/connection.hpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/services/logger_service.hpp
    /opt/local/share/doc/boost/libs/asio/example/cpp03/windows/transmit_file.cpp
    /opt/local/share/doc/boost/libs/asio/example/cpp11/chat/chat_client.cpp


cpp03/services/logger_service.hpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* file:///opt/local/share/doc/boost/doc/html/boost_asio/example/cpp03/services/logger_service.hpp

::

    /// Service implementation for the logger.
    class logger_service
      : public boost::asio::io_service::service
    {


    /// Constructor creates a thread to run a private io_service.
    logger_service(boost::asio::io_service& io_service)
        : boost::asio::io_service::service(io_service),
          work_io_service_(),
          work_(new boost::asio::io_service::work(work_io_service_)),
          work_thread_(new boost::thread(
                boost::bind(&boost::asio::io_service::run, &work_io_service_)))
      {
      }


Public methods post to the private work_io_service_ and only work_thread_ has
invoked run for it, so the posted handler private methods will run on the worker thread.

::

      /// Log a message.
      void log(impl_type& impl, const std::string& message)
      {
        // Format the text to be logged.
        std::ostringstream os;
        os << impl->identifier << ": " << message;

        // Pass the work of opening the file to the background thread.
        work_io_service_.post(boost::bind(
              &logger_service::log_impl, this, os.str()));
      }



NumpyServer
--------------

See numpyserver-  

* posts from network thread up to a delegate thread and io_service, typically main thread





Other Examples
---------------

See asiosamples-


posting to boost::bind delegate ?
------------------------------------

* :google:`boost::asio post to delegate`

* http://docs.ros.org/indigo/api/socketcan_interface/html/asio__base_8h_source.html

::

    00010 namespace can{
    00011 
    00012 
    00013 template<typename FrameDelegate, typename StateDelegate, typename Socket> class AsioDriver : public DriverInterface{
    00014     static void call_delegate(const FrameDelegate &delegate, const Frame &msg){
    00015         delegate(msg);
    00016     }
    00017 
    00018     State state_;
    00019     boost::mutex state_mutex_;
    00020     boost::mutex socket_mutex_;
    00021     
    00022     FrameDelegate frame_delegate_;
    00023     StateDelegate state_delegate_;
    00024 protected:
    00025     boost::asio::io_service io_service_;
    00026     Socket socket_;
    00027     Frame input_;
    00028     
    00029     virtual void triggerReadSome() = 0;
    00030     virtual bool enqueue(const Frame & msg) = 0;
    00031     
    00032     void dispatchFrame(const Frame &msg){
    00033         io_service_.post(boost::bind(call_delegate, frame_delegate_,msg)); // copies msg
    00034     }


    00058     void frameReceived(const boost::system::error_code& error){
    00059         if(!error){
    00060             dispatchFrame(input_);
    00061             triggerReadSome();
    00062         }else{
    00063             setErrorCode(error);
    00064         }
    00065     }
    00066 
    00067     AsioDriver(FrameDelegate frame_delegate, StateDelegate state_delegate)
    00068     : frame_delegate_(frame_delegate), state_delegate_(state_delegate), socket_(io_service_)
    00069     {}
    00070 



* http://wiki.ros.org/socketcan_interface   Robot Operating System (ROS)






Boost inclusion
----------------

Note: Boost.Asio 1.10.6 will also be included in Boost 1.58.

Boost Asio that comes with macports boost 1.55
-------------------------------------------------

::

   port contents boost | grep asio

   file:///opt/local/share/doc/boost/doc/html/boost_asio.html


simple examples
~~~~~~~~~~~~~~~~

::

    cp /opt/local/share/doc/boost/doc/html/boost_asio/example/cpp03/echo/async_tcp_echo_server.cpp . 

    clang++ -I/opt/local/include -L/opt/local/lib -lboost_system-mt async_tcp_echo_server.cpp -o async_tcp_echo_server


How to integrate Boost ASIO into GUI app
------------------------------------------

* :google:`integrate boost asio into run loop`

* http://stackoverflow.com/questions/1001032/how-to-integrate-boost-asio-main-loop-in-gui-framework-like-qt4-or-gtk

* http://gbgames.dyndns.org/old-and-misc/c++/3dCustumGameEngine/src/mainClient.cpp

::


    void ioConnection(boost::asio::io_service * io_service){
        io_service->run();
    }

    int main()
    {
        try {
        boost::system::error_code ec;
        boost::asio::io_service io_service;

        //setup udp
        udp::socket us(io_service, udp::endpoint(udp::v4(), 0));
        udp::resolver uresolver(io_service);
        udp::resolver::query uquery(udp::v4(), "127.0.0.1", "13");
        udp::resolver::iterator uiterator = uresolver.resolve(uquery);

        //setup tcp
        tcp::resolver tresolver(io_service);
        tcp::resolver::query tquery(tcp::v4(), "127.0.0.1", "14");
        tcp::resolver::iterator iter = tresolver.resolve(tquery);
        tcp::resolver::iterator end;

        tcp::socket ts(io_service);
        ts.connect(*iter);
        CommandHandler::init(&ts,&us);
        tcpClient::pointer cl = tcpClient::create(&ts);
        cl->start();

        boost::thread ioThread(boost::bind(ioConnection, &io_service));
        


boost asio tutorial
--------------------

* http://www.boost.org/doc/libs/1_39_0/doc/html/boost_asio/tutorial/tuttimer3.html

* file:///opt/local/share/doc/boost/doc/html/boost_asio/tutorial/tuttimer4.html

boost::bind() converts our callback handler (now a member function) into a
function object that can be invoked as though it has the signature 
void(const boost::system::error_code&).

* file:///opt/local/share/doc/boost/doc/html/boost_asio/tutorial/tuttimer5.html

boost::asio::strand class to synchronise callback handlers in a multithreaded program

::

     66 int main()
     67 {
     68   boost::asio::io_service io;
     69   printer p(io);
     70   boost::thread t(boost::bind(&boost::asio::io_service::run, &io));
     71   io.run();
     72   t.join();
     73 
     74   return 0;
     75 }


daytime1
~~~~~~~~~~

* http://tf.nist.gov/tf-cgi/servers.cgi for list of daytime servers and their status

::

    delta:daytime1 blyth$ basio-example-make client.cpp 
    clang++ -I/opt/local/include -L/opt/local/lib -lboost_system-mt -lboost_thread-mt client.cpp -o client
    delta:daytime1 blyth$ ./client 
    Usage: client <host>
    delta:daytime1 blyth$ ./client 206.246.122.250

    57111 15-03-30 07:06:58 50 0 0 531.0 UTC(NIST) * 
    delta:daytime1 blyth$ pwd
    /usr/local/env/boost/basio/example/cpp03/tutorial/daytime1


daytime3 and daytime1
~~~~~~~~~~~~~~~~~~~~~~~

Start server in one terminal and run client from another, 
after stopping the server get connection refused::

    delta:daytime3 blyth$ sudo ./server

    delta:daytime1 blyth$ ./client 127.0.0.1
    Mon Mar 30 15:19:22 2015
    delta:daytime1 blyth$ 

    delta:daytime1 blyth$ ./client 127.0.0.1
    connect: Connection refused


boost::asio with zeromq
-------------------------

* http://iyedb.github.io/cpp11/en/2014/07/11/asio-zeromq-cpp11.html

* https://indico.cern.ch/event/281860/contribution/6/material/slides/0.pdf

* https://github.com/JonasKunze

* https://github.com/yayj/asio-zmq

azmq
~~~~~

C++ language binding library integrating ZeroMQ with Boost Asio

* https://github.com/zeromq/azmq
* https://rodgert.github.io/2014/12/24/boost-asio-and-zeromq-pt1/

Library dependencies are -

Boost 1.54 or later
ZeroMQ 4.0.x

boost::bind
------------

* http://www.boost.org/doc/libs/1_45_0/libs/bind/bind.html
* http://www.boost.org/doc/libs/1_45_0/libs/bind/bind.html#Purpose

Bakes function invokation into an object and allows
setting some arguments and retaining others.


OSX compilation issue with macports boost 1.55
-----------------------------------------------

* https://trac.macports.org/ticket/42282
* https://svn.boost.org/trac/boost/ticket/9610

::

    /opt/local/bin/clang++-mp-3.4  -ftemplate-depth-128 -Os -stdlib=libc++ -O3 -finline-functions -Wno-inline -Wall -pedantic -gdwarf-2 -fexceptions -arch i386 -arch x86_64 -Wextra -Wno-long-long -Wno-variadic-macros -Wunused-function -fpermissive -pedantic -DBOOST_ALL_NO_LIB=1 -DBOOST_ATOMIC_STATIC_LINK=1 -DBOOST_SYSTEM_STATIC_LINK=1 -DBOOST_THREAD_BUILD_LIB=1 -DBOOST_THREAD_DONT_USE_CHRONO -DBOOST_THREAD_POSIX -DNDEBUG  -I"." -c -o "bin.v2/libs/thread/build/darwin-4.2.1/release/address-model-32_64/architecture-x86/link-static/pch-off/threading-multi/pthread/once.o" "libs/thread/src/pthread/once.cpp"
    In file included from libs/thread/src/pthread/once.cpp:8:
    In file included from libs/thread/src/pthread/./once_atomic.cpp:9:
    In file included from ./boost/thread/once.hpp:20:
    In file included from ./boost/thread/pthread/once_atomic.hpp:20:
    In file included from ./boost/atomic.hpp:12:
    In file included from ./boost/atomic/atomic.hpp:17:
    In file included from ./boost/atomic/detail/platform.hpp:22:
    ./boost/atomic/detail/gcc-atomic.hpp:961:64: error: no matching constructor for initialization of 'storage_type' (aka
          'boost::atomics::detail::storage128_type')
        explicit base_atomic(value_type const& v) BOOST_NOEXCEPT : v_(0)
                                                                   ^  ~


Peers
------

* :google:`boost::asio alternatives`


SDL_net
~~~~~~~~

* http://www.libsdl.org/projects/SDL_net/

Community
----------

* http://www.digitalpeer.com/blog/asio-is-my-go-to-cpp-application-framework
* Packt book :google:`Boost.Asio C++ Network Programming, John Torjo`

* http://www.gamedev.net/blog/950/entry-2249317-a-guide-to-getting-started-with-boostasio/

* http://alexott.net/en/cpp/BoostAsioNotes.html

Event Loop Integration
------------------------

* http://www.zaphoyd.com/websocketpp/manual/common-patterns/io-strategies-and-event-loop-integration

* http://stackoverflow.com/questions/1001032/how-to-integrate-boost-asio-main-loop-in-gui-framework-like-qt4-or-gtk

Just ASIO without boost
---------------------------

* https://think-async.com
* https://github.com/chriskohlhoff/asio/
* http://think-async.com/Asio/AsioAndBoostAsio

Using ASIO with GLFW
----------------------

* :google:`glfw boost::asio`
* :google:`asio io_service glfw`

* https://bitbucket.org/voxelstorm/voxelstorm/wiki/VoxelStorm%20Dependencies





EOU
}
basio-dir(){ echo $(local-base)/env/boost/basio ; }
basio-sdir(){ echo $(opticks-home)/boost/basio ; }
basio-cd(){  cd $(basio-dir); }
basio-scd(){  mkdir -p $(basio-sdir) ; cd $(basio-sdir); }
basio-mate(){ mate $(basio-dir) ; }
basio-get(){
   local dir=$(dirname $(basio-dir)) &&  mkdir -p $dir && cd $dir

    

}


basio-contents(){ port contents boost  | grep asio ; }

basio-doc(){ open file:///opt/local/share/doc/boost/doc/html/boost_asio.html ; }

basio-example-idir(){ echo /opt/local/share/doc/boost/libs/asio/example ; }
basio-example-src(){  echo $(basio-example-idir)/cpp03 ; }
basio-example-dst(){  echo $(basio-dir)/example/cpp03 ; }
basio-example-get(){  
   local src=$(basio-example-src)
   local dst=$(basio-example-dst)

   mkdir -p $(dirname $dst)
   cp -r $src $dst/
}
basio-example-cd(){  cd $(basio-example-dst); }  

basio-example-find(){
  find $(basio-example-idir) -type f -exec grep -H ${1:-post} {} \;
} 
basio-example-lfind(){
  find $(basio-example-idir) -type f -exec grep -l ${1:-post} {} \;
} 



basio-example-make(){ 
   local cpp=$1
   local bin=${cpp/.cpp}
   [ "$cpp" == "$bin" ] && echo invalid cpp $cpp && return 

   local tmp=/tmp/env/boost/basio/$bin
   mkdir -p $(dirname $tmp)
   local cmd="clang++ -I/opt/local/include -L/opt/local/lib -lboost_system-mt -lboost_thread-mt $cpp -o $tmp "
   echo $cmd
   eval $cmd
 }



