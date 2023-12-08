#include <cassert>
#include <csignal>

#include "U4SensitiveDetector.hh"

int main(int argc, char** argv)
{
    U4SensitiveDetector* sd0 = new U4SensitiveDetector("sd0"); 
    U4SensitiveDetector* sd1 = new U4SensitiveDetector("sd1"); 

    bool sd0_expect = U4SensitiveDetector::Get("sd0") == sd0 ;
    bool sd1_expect = U4SensitiveDetector::Get("sd1") == sd1 ;

    assert( sd0_expect ); 
    assert( sd1_expect ); 

    if(!sd0_expect) std::raise(SIGINT); 
    if(!sd1_expect) std::raise(SIGINT); 

    return 0 ; 
}
