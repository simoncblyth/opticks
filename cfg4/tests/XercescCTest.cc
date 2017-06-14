// clang XercescCTest.cc -I/opt/local/include -L/opt/local/lib -lxerces-c -lc++ -o /tmp/XercescCTest

#include <iostream>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/dom/DOM.hpp>

int main(int argc, char** argv)
{
    std::cout << argc << " " << argv[0] << std::endl ; 

    //xercesc_3_1::XMLPlatformUtils::Initialize();
    xercesc_2_8::XMLPlatformUtils::Initialize();
 
    return 0 ; 
}
