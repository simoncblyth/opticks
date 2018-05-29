#include <sstream>
#include <string>
#include "SSys.hh"

#include "SYSRAP_LOG.hh"
#include "PLOG.hh"


int test_tpmt()
{
    return SSys::run("tpmt.py");
}

int test_RC(int irc)
{
    //assert( irc < 0xff && irc >= 0 ) ; 
    std::stringstream ss ; 
    ss << "python -c 'import sys ; sys.exit(" << irc << ")'" ;
    std::string s = ss.str();
    return SSys::run(s.c_str());
}


void test_RC()
{
    int rc(0);
    for(int irc=0 ; irc < 500 ; irc+=10 )
    {
        int xrc = irc & 0xff ;   // beyond 0xff return codes get truncated 
        rc = test_RC(irc);       
        assert( rc == xrc ); 
    } 
}

int test_OKConfCheck()
{
    int rc = SSys::OKConfCheck();
    assert( rc == 0 );
    return rc ; 
}



int main(int argc , char** argv )
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 

    int rc(0) ;

    rc = test_OKConfCheck();

    //rc = test_tpmt();

    //rc = test_RC(77);

    LOG(info) << argv[0] << " rc " << rc ; 

    return rc  ; 
}

