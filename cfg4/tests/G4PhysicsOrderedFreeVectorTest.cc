#include <iostream>
#include <fstream>
#include <sstream>

#include "G4PhysicsOrderedFreeVector.hh"
#include "PLOG.hh"

class redirecter
// http://wordaligned.org/articles/cpp-streambufs
{
public:
    redirecter(std::ostream & dst, std::ostream & src)
        : 
        src(src), 
        sbuf(src.rdbuf(dst.rdbuf())) 
    {
    }

    ~redirecter() { src.rdbuf(sbuf); }
private:
    std::ostream & src;
    std::streambuf * const sbuf;
};

void test_redirected( G4PhysicsOrderedFreeVector& vec, bool ascii )
{
    std::ofstream fp("/dev/null", std::ios::out); 

    std::stringstream ss ;     
    redirecter rdir(ss,fp);    // redirect writes to the file to instead go to the stringstream 
    
    vec.Store(fp, ascii );

    std::cout <<  ss.str() << std::endl ; 
}

void test_caveman( G4PhysicsOrderedFreeVector& vec, bool ascii )
{
    std::vector<char> buf(512);
    for(unsigned j=0 ; j < buf.size() ; j++ ) buf[j] = '*' ; 

    std::ofstream fp("/dev/null", std::ios::out); 

    fp.rdbuf()->pubsetbuf(buf.data(),buf.size());

    vec.Store(fp, ascii );

    for(unsigned j=0 ; j < buf.size() ; j++ )
    {
        std::cout << " " << buf[j] ; 
        if( (j + 1) % 16 == 0 ) std::cout << std::endl ; 
    }
}





int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    size_t n = 5 ; 
    double e[n] ; 
    double v[n] ; 
    for(unsigned i=0 ; i < n ; i++ )
    {
        e[i] = double(i)*100 + 0.1 ;  
        v[i] = double(i)*1000 + 0.2 ;  
    } 
    
    G4PhysicsOrderedFreeVector vec(e, v, n ); 
    std::cout << vec << std::endl ; 

    // Making an ofstream writing method write into a buffer 

    bool ascii = false ; 

    //test_caveman(   vec, ascii ); 
    test_redirected( vec, ascii ); 



    return 0 ; 
}


