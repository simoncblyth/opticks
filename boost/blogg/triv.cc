/*

  simon:blogg blyth$ /tmp/triv 1>/dev/null
[2015-May-20 17:06:04.837115]: An informational severity message
[2015-May-20 17:06:04.837668]: A warning severity message
[2015-May-20 17:06:04.837749]: An error severity message
[2015-May-20 17:06:04.837815]: A fatal severity message
simon:blogg blyth$ /tmp/triv 2>/dev/null
simon:blogg blyth$ 


*/


//  https://gerrydevstory.com/2014/05/28/using-boost-logging/
#include <iostream>
 
#include "boost/log/trivial.hpp"
#include "boost/log/utility/setup.hpp"
 
//using namespace std;
 
int main() {
  // Output message to console

  boost::log::add_console_log(
    std::cerr, 
    boost::log::keywords::format = "[%TimeStamp%]: %Message%",
    boost::log::keywords::auto_flush = true
  );
 
  // Output message to file, rotates when file reached 1mb or at midnight every day. Each log file
  // is capped at 1mb and total is 20mb
  boost::log::add_file_log (
    boost::log::keywords::file_name = "MyApp_%3N.log",
    boost::log::keywords::rotation_size = 1 * 1024 * 1024,
    boost::log::keywords::max_size = 20 * 1024 * 1024,
    boost::log::keywords::time_based_rotation = boost::log::sinks::file::rotation_at_time_point(0, 0, 0),
    boost::log::keywords::format = "[%TimeStamp%]: %Message%",
    boost::log::keywords::auto_flush = true
  );
 
  boost::log::add_common_attributes();
 
  // Only output message with INFO or higher severity
  boost::log::core::get()->set_filter(
    boost::log::trivial::severity >= boost::log::trivial::info
  );
 
  // Output some simple log message
  BOOST_LOG_TRIVIAL(trace) << "A trace severity message";
  BOOST_LOG_TRIVIAL(debug) << "A debug severity message";
  BOOST_LOG_TRIVIAL(info) << "An informational severity message";
  BOOST_LOG_TRIVIAL(warning) << "A warning severity message";
  BOOST_LOG_TRIVIAL(error) << "An error severity message";
  BOOST_LOG_TRIVIAL(fatal) << "A fatal severity message";
}

