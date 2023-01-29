// name=slog_lite_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -I$OPTICKS_PREFIX/externals/plog/include -o /tmp/$name && slog_lite=info /tmp/$name

#include "slog.h"

int main(int argc, char** argv)
{
    plog::Severity level = slog::envlevel("slog_lite", "DEBUG" ); 

    std::cout << slog::Desc(level) ; 
    std::cout << slog::Dump() ; 


    return 0 ; 
}
