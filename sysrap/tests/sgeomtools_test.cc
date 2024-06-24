/**

   ~/o/sysrap/tests/sgeomtools_test.sh 

**/


#include "sgeomtools.h"
#include "scuda_double.h"


struct DiskExtent
{
    static constexpr const double rmin = 0.9 ; 
    static constexpr const double rmax = 1.0 ; 

    double startPhi ; 
    double deltaPhi ; 
    double endPhi ; 

    double2 pmin = {} ; 
    double2 pmax = {} ; 

    DiskExtent(double _startPhi, double _deltaPhi) ; 

    std::string desc() const ; 

    static int Main(); 
    static int test_0(); 
    static int test_1(); 
    static int test_2(); 
    static int test_3(); 
    static int test_4(); 

};
    
inline DiskExtent::DiskExtent(double _startPhi, double _deltaPhi) 
    :
    startPhi(_startPhi),
    deltaPhi(_deltaPhi),
    endPhi(startPhi+deltaPhi)
{
    sgeomtools::DiskExtent(rmin, rmax, startPhi, deltaPhi, pmin, pmax ); 
}

    
inline std::string DiskExtent::desc() const 
{
    std::stringstream ss ; 

    ss 
         << "DiskExtent::desc"
         << "\n"
         << " startPhi " << startPhi
         << " startPhi/M_PI " << startPhi/M_PI
         << "\n"
         << " endPhi " << endPhi
         << " endPhi/M_PI " << endPhi/M_PI
         << "\n"
         << " deltaPhi " << deltaPhi
         << " deltaPhi/M_PI " << deltaPhi/M_PI
         << "\n"
         << " pmax " << pmax 
         << "\n"
         << " pmin " << pmin 
         << "\n"
         ;

    std::string str = ss.str() ; 
    return str ; 
}



/**
DiskExtent::test_0
-------------------
                        (10,10)
           +-----+-----+
           | .       . |
           |.         .|
     pi -x +-----+-----+x  phi=0
      (-10,0)

**/

int DiskExtent::test_0()
{
    DiskExtent t(0.,M_PI); 
    std::cout << "test_0\n" << t.desc() ;     
    return 0 ; 
}


/**
DiskExtent::test_1
-------------------
               pi/2     (10,10)
           +-----+-----+
           | .   |   . |
           |.    |    .|
     pi -x +-----+-----+x  phi=0
      (-10,0)

**/

int DiskExtent::test_1()
{
    DiskExtent t(0.,M_PI/2);  
    std::cout << "test_1 (1st quad)\n" << t.desc() ;     
    return 0 ; 
}

int DiskExtent::test_2()
{
    DiskExtent t(M_PI/2,M_PI/2);  
    std::cout << "test_2 (2nd quad)\n" << t.desc() ;     
    return 0 ; 
}

int DiskExtent::test_3()
{
    DiskExtent t(M_PI,M_PI/2);  
    std::cout << "test_3 (3rd quad)\n" << t.desc() ;     
    return 0 ; 
}

int DiskExtent::test_4()
{
    DiskExtent t(1.5*M_PI,M_PI/2);  
    std::cout << "test_4 (4th quad)\n" << t.desc() ;     
    return 0 ; 
}



int DiskExtent::Main()
{
    int rc = 0 ; 
    rc += test_0();
    rc += test_1();
    rc += test_2();
    rc += test_3();
    rc += test_4();
    return rc ; 
}

int main()
{
    return DiskExtent::Main() ; 
}
