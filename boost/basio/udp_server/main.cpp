/*
Test with udpserver-test OR  UDP_PORT=13 udp.py hello
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

   App(unsigned int udp_port, const char* zmq_backend)  
      :
        m_io_service_work(new boost::asio::io_service::work(m_io_service)),
        m_udp_manager(this, udp_port, zmq_backend)
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

   void on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata)
   {
        std::cout << std::setw(20) << boost::this_thread::get_id() 
                  << " App::on_npy " 
                  << " shape dimension " << shape.size()
                  << " data size "  << data.size()
                  << " metadata [" << metadata << "]" 
                  << std::endl;
   }





};


int main()
{
    std::cout << std::setw(20) << boost::this_thread::get_id() << " main " << std::endl;

    App app(13, "tcp://127.0.0.1:5002");

    for(unsigned int i=0 ; i < 20 ; ++i )
    {
        app.poll_one();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    app.stop();

    return 0;
}

