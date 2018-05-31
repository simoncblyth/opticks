
#include "SDirect.hh"
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "SSys.hh"
#include "SYSRAP_LOG.hh"
#include "PLOG.hh"


void test_cout_cerr_redirect(const char* msg)
{
    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {
        cout_redirect out_(coutbuf.rdbuf());
        cerr_redirect err_(cerrbuf.rdbuf());
        
        SSys::Dump(msg); 
        
        // dtors of the redirect structs reset back to standard cout/cerr streams  
    }        

    std::string out = coutbuf.str(); 
    std::string err = cerrbuf.str(); 

    LOG(info) << " captured cout " << out.size()  ; 
    std::cout << "[" << std::endl << out << "]" << std::endl  ; 

    LOG(info) << " captured cerr " << err.size() ; 
    std::cout << "[" << std::endl << err << "]" << std::endl  ; 

    SSys::Dump(msg); 
}


void method_expecting_to_write_to_file( std::ofstream& fp, std::vector<std::string>& msgv )
{
    for(unsigned i=0 ; i < msgv.size() ; i++ )
    {
        const char* pt = msgv[i].c_str() ;
        fp.write( const_cast<char*>(pt) , sizeof(pt)); 
    }
} 

void test_stream_redirect()
{
    std::ofstream fp("/dev/null", std::ios::out); 
    std::stringstream ss ;          

    stream_redirect rdir(ss,fp); // stream_redirect such that writes to the file instead go to the stringstream 

    std::vector<std::string> msgv ; 
    msgv.push_back("hello"); 
    msgv.push_back("world"); 
 
    method_expecting_to_write_to_file(fp, msgv);

    std::cout <<  ss.str() << std::endl ; 
}


int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 

    LOG(info) << argv[0] ; 

    SSys::Dump(argv[0]); 

    //test_cout_cerr_redirect(argv[0]); 
    test_stream_redirect(); 


    return 0 ; 
}

