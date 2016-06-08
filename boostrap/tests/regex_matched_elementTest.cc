#include "regexsearch.hh"
#include "stdio.h"

int main(int argc, char** argv)
{
    std::string o = regex_matched_element(" __hello__world__is__what__i__want ");
    printf("%s\n", o.c_str());
    return 0 ; 
}
