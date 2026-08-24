#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstring>
#include <array>

struct Demo
{
    Demo(const char* name_, int value_)
       :
       name(strdup(name_)),
       value(value_)
    {
    }
    std::string desc() const
    {
        std::stringstream ss ;
        ss
           << std::setw(10) << name
           << " : "
           << std::setw(10) << value
           ;
        return ss.str();
    }

    const char* name ;
    int value ;
};


int main(int argc, char** argv)
{
    std::array<Demo*, 10> arr ;
    arr.fill(NULL);

    arr[5] = new Demo("yo", 42) ;

    for(unsigned i=0 ; i < arr.size() ; i++) std::cout << i << " : " << ( arr[i] ? arr[i]->desc() : "-" ) << "\n" ;

    // arr[10] = new Demo("hmm", 42) ; // THIS IS OFF THE END - gcc11 DID NOT NOTICE, gcc15 DID

    return 0 ;
}


