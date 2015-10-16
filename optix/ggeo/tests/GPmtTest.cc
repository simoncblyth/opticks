#include "GPmt.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

int main(int argc, char* argv[])
{
    GPmt* pmt = GPmt::load("/tmp/hemi-pmt-parts.npy");
    pmt->dump();
    pmt->Summary();

    return 0 ;
}


