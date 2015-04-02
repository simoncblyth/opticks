#include "app.hpp"

#include <iostream>
#include <iomanip>
#include <boost/thread.hpp>

App::App()
{
}

void App::on_msg(std::string msg)
{

    std::cout << std::setw(20) << boost::this_thread::get_id() 
              << " App::on_msg " 
              << " msg ["  << msg << "] "
              << std::endl;
}

void App::on_npy(std::vector<int> shape, std::vector<float> data, std::string metadata)
{
    std::cout << std::setw(20) << boost::this_thread::get_id() 
              << " App::on_npy "
              << " shape dimension " << shape.size()
              << " data size " << data.size()
              << " metadata [" << metadata << "]" 
              << std::endl ; 

} 


