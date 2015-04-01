# === func-gen- : boost/basio/basio fgp boost/basio/basio.bash fgn basio fgh boost/basio
basio-src(){      echo boost/basio/basio.bash ; }
basio-source(){   echo ${BASH_SOURCE:-$(env-home)/$(basio-src)} ; }
basio-vi(){       vi $(basio-source) ; }
basio-env(){      elocal- ; }
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

* http://theboostcpplibraries.com/boost.asio


Message passing from netThread to mainThread
----------------------------------------------

* http://stackoverflow.com/questions/17976568/boost-asio-pattern-with-gui-and-worker-thread

* http://stackoverflow.com/questions/17311512/howto-post-messages-between-threads-with-boostasio/17315567#17315567



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
basio-sdir(){ echo $(env-home)/boost/basio ; }
basio-cd(){  cd $(basio-dir); }
basio-scd(){  mkdir -p $(basio-sdir) ; cd $(basio-sdir); }
basio-mate(){ mate $(basio-dir) ; }
basio-get(){
   local dir=$(dirname $(basio-dir)) &&  mkdir -p $dir && cd $dir

    

}

basio-doc(){ open file:///opt/local/share/doc/boost/doc/html/boost_asio.html ; }

basio-example-src(){  echo /opt/local/share/doc/boost/libs/asio/example/cpp03 ; }
basio-example-dst(){  echo $(basio-dir)/example/cpp03 ; }
basio-example-get(){  
   local src=$(basio-example-src)
   local dst=$(basio-example-dst)

   mkdir -p $(dirname $dst)
   cp -r $src $dst/
}
basio-example-cd(){  cd $(basio-example-dst); }  


basio-example-make(){ 
   local cpp=$1
   local bin=${cpp/.cpp}
   [ "$cpp" == "$bin" ] && echo invalid cpp $cpp && return 

   local cmd="clang++ -I/opt/local/include -L/opt/local/lib -lboost_system-mt -lboost_thread-mt $cpp -o $bin "
   echo $cmd
   eval $cmd
 }



