#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL

int main(int, char*[])
{
    LOG(trace) << "A trace severity message";
    LOG(debug) << "A debug severity message";
    LOG(info) << "An informational severity message";
    LOG(warning) << "A warning severity message";
    LOG(error) << "An error severity message";
    LOG(fatal) << "A fatal severity message";

    return 0;
}
