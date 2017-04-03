#include "BStr.hh"
#include <iomanip>

#include <iterator>
#include <iostream>

#include <cstdio>
#include <boost/algorithm/string/replace.hpp>

#include "PLOG.hh"


void test_ijoin()
{
    std::vector<int> elem ; 
    std::string ij = BStr::ijoin(elem, ',');
    assert( strcmp(ij.c_str(), "") == 0 );
}



void test_fsplit()
{
    const char* line = "1.12,10.0,-100.1,-200,+20.5" ;

    std::vector<float> elem ; 
    BStr::fsplit(elem, line, ',');

    LOG(info) << " fsplit [" << line << "] into elem count " << elem.size() ; 
    for(unsigned i=0 ; i < elem.size() ; i++) std::cout << elem[i] << std::endl ; 

    assert( elem.size() == 5 );
    assert( elem[0] == 1.12f );
    assert( elem[1] == 10.f );
    assert( elem[2] == -100.1f );
    assert( elem[3] == -200.f );
    assert( elem[4] == 20.5f );

}


void test_isplit()
{
    const char* line = "1,10,100,-200" ;

    std::vector<int> elem ; 
    BStr::isplit(elem, line, ',');

    LOG(info) << " isplit [" << line << "] into elem count " << elem.size() ; 

    assert(elem.size() == 4);
    assert(elem[0] == 1 );
    assert(elem[1] == 10 );
    assert(elem[2] == 100 );
    assert(elem[3] == -200 );

    std::string ij = BStr::ijoin(elem, ',');

    LOG(info) << " ijoin elem into [" << ij << "]" ;

    assert( strcmp( ij.c_str(), line) == 0);
}


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

/*
    test_patternPickField();
    test_afterLastOrAll();
    test_DAEIdToG4();
    test_isplit();
    test_ijoin();
*/
    test_fsplit();

    return 0 ; 
}

