#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    std::cout << SLOG::Banner() << std::endl ; 

    return 0 ; 
}
