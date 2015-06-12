#include "regexsearch.hh"
#include "stdio.h"

int main(int argc, char** argv)
{
    std::string o = regex_extract_quoted("#incl \"some_path.h\"");
    printf("%s\n", o.c_str());
    return 0 ; 
}
