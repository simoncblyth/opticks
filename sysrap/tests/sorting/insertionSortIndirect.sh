#!/bin/bash -l 

name=insertionSortIndirect 


create()
{
    local e_path=/tmp/e.npy
    if [ -f "$e_path" ]; then
       ${OPTICKS_PYTHON:-python} -c "import numpy as np ; np.set_printoptions(suppress=True) ; e = np.load('$e_path') ; print(e) " 
    else
       ${OPTICKS_PYTHON:-python} -c "import numpy as np ; np.save('$e_path', np.random.sample(1000).astype(np.float32)) " 
    fi 
}

create


gcc $name.cc -std=c++11 -lstdc++ -I$OPTICKS_PREFIX/include/SysRap -o /tmp/$name 
[ $? -ne 0 ] && echo $msg compile error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 2 


if [ -n "$I" ]; then 
   ${IPYTHON:-ipython} -i $name.py 
else
   ${OPTICKS_PYTHON:-python}  $name.py 
fi 



exit 0 


