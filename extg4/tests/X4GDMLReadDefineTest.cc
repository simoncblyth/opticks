
#include <map>
#include <cassert>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "G4GDMLReadDefine.hh"   // for G4GDMLMatrix
#include "G4GDMLEvaluator.hh"

/*

<matrix coldim="2" name="EFFICIENCY0x1d79780" values="1.512e-06 0.0001 1.5498e-06 0.0001 1.58954e-06 0.000440306 1.63137e-06 0.000782349 1.67546e-06 0.00112439 1.722e-06 0.00146644 1.7712e-0      6 0.00180848 1.8233e-06 0.00272834 1.87855e-06 0.00438339 1.93725e-06 0.00692303 1.99974e-06 0.00998793 2.0664e-06 0.0190265 2.13766e-06 0.027468 2.214e-06 0.0460445 2.296e-06 0.0652553 2.38431e      -06 0.0849149 2.47968e-06 0.104962 2.583e-06 0.139298 2.69531e-06 0.170217 2.81782e-06 0.19469 2.952e-06 0.214631 3.0996e-06 0.225015 3.26274e-06 0.24 3.44401e-06 0.235045 3.64659e-06 0.21478 3.      87451e-06 0.154862 4.13281e-06 0.031507 4.42801e-06 0.00478915 4.76862e-06 0.00242326 5.16601e-06 0.000850572 5.63564e-06 0.000475524 6.19921e-06 0.000100476 6.88801e-06 7.50165e-05 7.74901e-06       5.00012e-05 8.85601e-06 2.49859e-05 1.0332e-05 0 1.23984e-05 0 1.5498e-05 0 2.0664e-05 0"/>

*/

int main(int argc, char** argv)
{
    std::map<G4String,G4GDMLMatrix> matrixMap; 
    G4GDMLEvaluator eval ;  

    G4String name = "EFFICIENCY0x1d79780";
    G4int coldim  = 2 ; 
    G4String values = "1.512e-06 0.0001 1.5498e-06 0.0001 1.58954e-06 0.000440306 1.63137e-06 0.000782349 1.67546e-06 0.00112439 1.722e-06 0.00146644 1.7712e-06 0.00180848 1.8233e-06 0.00272834 1.87855e-06 0.00438339 1.93725e-06 0.00692303 1.99974e-06 0.00998793 2.0664e-06 0.0190265 2.13766e-06 0.027468 2.214e-06 0.0460445 2.296e-06 0.0652553 2.38431e-06 0.0849149 2.47968e-06 0.104962 2.583e-06 0.139298 2.69531e-06 0.170217 2.81782e-06 0.19469 2.952e-06 0.214631 3.0996e-06 0.225015 3.26274e-06 0.24 3.44401e-06 0.235045 3.64659e-06 0.21478 3.87451e-06 0.154862 4.13281e-06 0.031507 4.42801e-06 0.00478915 4.76862e-06 0.00242326 5.16601e-06 0.000850572 5.63564e-06 0.000475524 6.19921e-06 0.000100476 6.88801e-06 7.50165e-05 7.74901e-06 5.00012e-05 8.85601e-06 2.49859e-05 1.0332e-05 0 1.23984e-05 0 1.5498e-05 0 2.0664e-05 0"  ;  
    
    std::stringstream MatrixValueStream(values);
    std::vector<double> valueList;

    while (!MatrixValueStream.eof())
    {   
        std::string MatrixValue;
        MatrixValueStream >> MatrixValue;
        std::cout << MatrixValue << std::endl ;
        valueList.push_back(eval.Evaluate(MatrixValue));
    }   
    assert( valueList.size() == 39*2 );  
    for(unsigned i=0 ; i < valueList.size() ; i++ ) std::cout << std::fixed << std::setprecision(4) << valueList[i] << std::endl ; 

    eval.DefineMatrix(name,coldim,valueList);

    G4GDMLMatrix matrix(valueList.size()/coldim,coldim);

    for (size_t i=0;i<valueList.size();i++)
    {   
        matrix.Set(i/coldim,i%coldim,valueList[i]);
    }   

    assert( matrix.GetCols() == 2 ); 
    assert( matrix.GetRows() == 39 ); 
 

    matrixMap[name] = matrix;

    // some difference in handling of non-unique names could explain the zeros
    // but this works for materials in 1062, so must be something surface specific


    return 0 ; 
}
