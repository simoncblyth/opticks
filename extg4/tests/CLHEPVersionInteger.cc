#include "CLHEP/ClhepVersion.h"
#include <stdio.h>

unsigned CLHEPVersionInteger()
{
     int maj = CLHEP::Version::Major(); 
     int sma = CLHEP::Version::SubMajor(); 
     int min = CLHEP::Version::Minor(); 
     int smi = CLHEP::Version::SubMinor(); 
     unsigned ver = maj*1000 + sma*100 + min*10 + smi ; 
     return ver ; 
}

int main()
{
    printf("%d\n", CLHEPVersionInteger()); 
    return 0 ; 
}


