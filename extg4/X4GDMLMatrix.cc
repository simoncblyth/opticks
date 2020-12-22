#include <sstream>
#include "X4GDMLMatrix.hh"

X4GDMLMatrix::X4GDMLMatrix(const G4GDMLMatrix& matrix )
    :
    m_matrix(matrix)
{
}


std::string X4GDMLMatrix::desc(unsigned edgeitems) const 
{
    size_t rows = m_matrix.GetRows() ;  // eg 39 
    size_t cols = m_matrix.GetCols() ;  // eg 2  

    std::stringstream ss ; 
    ss 
        << " rows " << rows
        << " cols " << cols
        << " " 
        ;

    double mn0 = std::numeric_limits<double>::max();   
    double mx0 = std::numeric_limits<double>::lowest(); 
    double mn1 = std::numeric_limits<double>::max();   
    double mx1 = std::numeric_limits<double>::lowest(); 

    for(size_t r=0 ; r < rows ; r++ ) 
    {
        for(size_t c=0 ; c < cols ; c++)
        {
            double v = m_matrix.Get(r,c) ; 

            if( c == 0 )
            {
                if(v > mx0) mx0 = v ;  
                if(v < mn0) mn0 = v ; 
            }
            else if( c == 1 )
            {
                if(v > mx1) mx1 = v ;  
                if(v < mn1) mn1 = v ; 
            }

            if( r < edgeitems || r > rows - edgeitems ) 
            {
                ss << v << " " ; 
            }
            else if( r == edgeitems ) 
            {
                ss << "... " ; 
            }
        }
    }

    ss 
        << " mn0 " << mn0 
        << " mx0 " << mx0 
        << " mn1 " << mn1 
        << " mx1 " << mx1 
        ;

    return ss.str(); 

}
 
