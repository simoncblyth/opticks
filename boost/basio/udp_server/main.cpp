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
   boost::scoped_ptr<boost::asio::io_service::work> m_io_service_work ;
   udp_manager<App>                                 m_udp_manager ; 

public:
   App(unsigned int port)  
      :
        m_udp_manager(this, port),   
        m_io_service_work(new boost::asio::io_service::work(m_io_service)) 
   {
   }

   void poll_one()
   {
       std::cout << std::setw(20) << boost::this_thread::get_id() << " App::poll_one " << std::endl;
       m_io_service.poll_one();
   } 

   void stop()
   {
       std::cout << std::setw(20) << boost::this_thread::get_id() << " App::stop " << std::endl;
       m_io_service_work.reset();
   } 


public:
   // methods that allow App to act as delegate to udp_manager 
   boost::asio::io_service& get_io_service()
   {
       return m_io_service ;  
   }

   void on_message(std::string msg)
   {
        std::cout << std::setw(20) << boost::this_thread::get_id() 
                  << " App::on_message " 
                  << " msg ["  << msg << "] "
                  << std::endl;
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

