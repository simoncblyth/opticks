#include "BStr.hh"
#include <iomanip>
#include <cstdio>
#include <boost/algorithm/string/replace.hpp>

#include "PLOG.hh"

void test_afterLastOrAll_(const char* s, const char* x )
{
    char* a = BStr::afterLastOrAll(s) ; 
    bool match = strcmp(a,x) == 0 ; 

    if(!match)
    LOG(fatal) << " MISMATCH " 
               << " s " << std::setw(30) << s 
               << " x " << std::setw(30) << x 
               << " a " << std::setw(30) << a
                ;

    assert(match); 
}


void test_afterLastOrAll()
{
    test_afterLastOrAll_("/hello/dear/world/take/me","me") ; 
    test_afterLastOrAll_("me","me") ; 
    test_afterLastOrAll_("me/","me/") ; 
}

void test_DAEIdToG4_(const char* daeid, const char* x_g4name)
{
    char* g4name = BStr::DAEIdToG4(daeid);
    bool match = strcmp( g4name, x_g4name  ) == 0 ;

    if(!match)
       LOG(fatal) << "MISMATCH"
                  << " daeid " << daeid 
                  << " g4name " << g4name 
                  << " x_g4name " << x_g4name 
                  ;

    assert(match);
}

void test_DAEIdToG4()
{
    test_DAEIdToG4_("__dd__Geometry__PoolDetails__lvLegInIWSTub0xc400e40", "/dd/Geometry/PoolDetails/lvLegInIWSTub" );
}



void test_patternPickField()
{
    std::string str = "aaaa__bbbb__cccccccccccccc__d__e" ;
    std::string ptn = "__" ;

    for(int field=-5 ; field < 5 ; field++ )
    {
        printf("patternPickField(%s,%s,%d) --> ", str.c_str(), ptn.c_str(), field  );
        std::string pick = BStr::patternPickField(str, ptn,field);
        printf(" %s \n", pick.c_str());
    }
}

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    test_patternPickField();
    test_afterLastOrAll();
    test_DAEIdToG4();
    return 0 ; 
}

