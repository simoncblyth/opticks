#include "stringutil.hpp"


void test_patternPickField()
{
    std::string str = "aaaa__bbbb__cccccccccccccc__d__e" ;
    std::string ptn = "__" ;

    for(int field=-5 ; field < 5 ; field++ )
    {
        printf("patternPickField(%s,%s,%d) --> ", str.c_str(), ptn.c_str(), field  );
        std::string pick = patternPickField(str, ptn,field);
        printf(" %s \n", pick.c_str());
    }
}



int main()
{
    test_patternPickField();
    return 0 ; 
}

