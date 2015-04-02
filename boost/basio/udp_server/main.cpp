//  clang++ -I/opt/local/include -L/opt/local/lib -lboost_system-mt udp_server_test.cpp udp_server.cpp -o udp_server_test

/*
Test with python client::

    UDP_PORT=13 udp.py hello
*/


#include "udp_manager.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>


class App {

   boost::asio::io_service                          m_io_service ;
   boost::scoped_ptr<boost::asio::io_service::work> m_work ;
   udp_manager                                      m_udp_manager ; 

public:
   App(unsigned int port)  
      :
        m_udp_manager(m_io_service, port),   
        m_work(new boost::asio::io_service::work(m_io_service)) // prevents m_io_service::run from terminating 
   {
   }

   void poll_one()
   {
       std::cout << std::setw(20) << boost::this_thread::get_id() << " App::poll_one " << std::endl;
       m_io_service.poll_one();
   } 
   void run()
   {
       std::cout << std::setw(20) << boost::this_thread::get_id() << " App::run " << std::endl;
       m_io_service.run();
   } 

   void stop()
   {
       std::cout << std::setw(20) << boost::this_thread::get_id() << " App::stop " << std::endl;
       m_work.reset();
   } 



};


int main()
{
    std::cout << std::setw(20) << boost::this_thread::get_id() << " main " << std::endl;

    App app(13);

    for(unsigned int i=0 ; i < 20 ; ++i )
    {
        app.poll_one();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    app.stop();

    return 0;
}

