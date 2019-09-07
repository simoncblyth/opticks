/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

/*
Test with eg::

        udpserver-test 
        UDP_PORT=13 udp.py hello


        npysend.sh --tag 1  # requires zmq- ; zmq-broker 

*/

#include "numpydelegate.hpp"
#include "numpyserver.hpp"

int main()
{
    numpydelegate nde ;  // example numpydelegate contains defaults 
    numpyserver<numpydelegate> srv(&nde);
    //nde.setServer(&srv); // needed for replying to posts from delegate calls
    
    for(unsigned int i=0 ; i < 20 ; ++i )
    {
        srv.poll();
        srv.sleep(1);
        //std::string msg = std::to_string(i) ;
        //srv.send( msg );
    }

    return 0;
}

