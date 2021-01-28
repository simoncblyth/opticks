#include "SSys.hh"
#include "OPTICKS_LOG.hh"


void test_ImportOpticks()
{
    LOG(info);
    SSys::run("python -c 'import opticks'"); 
}

void test_ImportNumPy()
{
    LOG(info);
    SSys::run("python -c 'import numpy'"); 
}

void test_python_numpy()
{
    LOG(info); 
    SSys::run("python -c 'import numpy as np ; print(np.__version__)'" ); 
}

void test_ResolvePython()
{
    const char* p = SSys::ResolvePython(); 
    LOG(info) << " p " << p ; 
}

void test_RunPythonScript()
{
    const char* script = "np.py" ; 
    int rc = SSys::RunPythonScript(script); 
    LOG(info) 
       << " script " << script 
       << " rc " << rc
       ;
}

void test_RunPythonCode()
{
    const char* code = "import numpy as np ; print(np.__version__)" ; 
    int rc = SSys::RunPythonCode(code); 
    LOG(info) 
       << " code " << code 
       << " rc " << rc
       ;
}

int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    test_ImportOpticks();
    test_ImportNumPy();
    test_python_numpy(); 
    test_ResolvePython(); 
    test_RunPythonScript(); 
    test_RunPythonCode(); 

    return 0  ; 
}

