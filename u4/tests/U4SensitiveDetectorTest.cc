#include <cassert>

#include "U4SensitiveDetector.hh"

int main(int argc, char** argv)
{
    U4SensitiveDetector* sd0 = new U4SensitiveDetector("sd0"); 
    U4SensitiveDetector* sd1 = new U4SensitiveDetector("sd1"); 

    assert( U4SensitiveDetector::Get("sd0") == sd0 ); 
    assert( U4SensitiveDetector::Get("sd1") == sd1 ); 

    return 0 ; 
}
