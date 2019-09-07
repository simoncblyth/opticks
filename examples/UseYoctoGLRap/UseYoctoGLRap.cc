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

// yoctoglrap/tests/YOGMakerTest.cc 

#include <string>
#include <iostream>
#include <fstream>

#include "YOGMaker.hh"

int main(int argc, char** argv)
{
    auto gltf = YOG::Maker::make_gltf_example() ; 

    std::string path = "/tmp/YOGMaker.gltf" ; 
    bool save_bin = false ; 
    bool save_shaders = false ; 
    bool save_images = false ; 

    save_gltf(path, gltf.get(), save_bin, save_shaders, save_images);

    std::cout << "writing " << path << std::endl ; 

    std::ifstream fp(path);
    std::string line;
    while(std::getline(fp, line)) std::cout << line << std::endl ; 
    return 0 ; 
}


/*

ygltf buffers use
    std::vector<unsigned char> 

hmm how to put NPY data arrays into ygltf buffers 


Need a way to predict the byteLength that the NPY will 
occupy in a file and the byteOffset to the start of the data 

Total header length is typically 80 or 90 bytes depending on array shape::

    epsilon:GBndLib blyth$ xxd -l 80 GBndLibOptical.npy
    00000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
    00000010: 7227 3a20 273c 7534 272c 2027 666f 7274  r': '<u4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2831 3233  e, 'shape': (123
    00000040: 2c20 342c 2034 292c 207d 2020 2020 200a  , 4, 4), }     .

    epsilon:GBndLib blyth$ xxd -l 80 GBndLibIndex.npy
    00000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
    00000010: 7227 3a20 273c 7534 272c 2027 666f 7274  r': '<u4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2831 3233  e, 'shape': (123
    00000040: 2c20 3429 2c20 7d20 2020 2020 2020 200a  , 4), }        .

    epsilon:GBndLib blyth$ xxd -l 96 GBndLib.npy
    00000000: 934e 554d 5059 0100 5600 7b27 6465 7363  .NUMPY..V.{'desc
    00000010: 7227 3a20 273c 6634 272c 2027 666f 7274  r': '<f4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2831 3233  e, 'shape': (123
    00000040: 2c20 342c 2032 2c20 3339 2c20 3429 2c20  , 4, 2, 39, 4), 
    00000050: 7d20 2020 2020 2020 2020 2020 2020 200a  }              .

    epsilon:0 blyth$ xxd -l 80 vertices.npy
    00000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
    00000010: 7227 3a20 273c 6634 272c 2027 666f 7274  r': '<f4', 'fort
    00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    00000030: 652c 2027 7368 6170 6527 3a20 2832 3034  e, 'shape': (204
    00000040: 3436 342c 2033 292c 207d 2020 2020 200a  464, 3), }     .
    epsilon:0 blyth$ 

    epsilon:0 blyth$ echo $(( 204464*3*4 + 80 ))
    2453648
    epsilon:0 blyth$ l vertices.npy
    -rw-r--r--  1 blyth  staff  - 2453648 Apr  4 21:59 vertices.npy
    epsilon:0 blyth$ 




* https://docs.scipy.org/doc/numpy/neps/npy-format.html

Format Specification: Version 1.0
The first 6 bytes are a magic string: exactly “x93NUMPY”.

The next 1 byte is an unsigned byte: the major version number of the file format, e.g. x01.

The next 1 byte is an unsigned byte: the minor version number of the file format, e.g. x00. 
Note: the version of the file format is not tied to the version of the numpy package.

The next 2 bytes form a little-endian unsigned short int: the length of the header data HEADER_LEN.

The next HEADER_LEN bytes form the header data describing the array’s format. 
It is an ASCII string which contains a Python literal expression of a dictionary. 
It is terminated by a newline (‘n’) and padded with spaces (‘x20’) to make 
the total length of the magic string + 4 + HEADER_LEN 
be evenly divisible by 16 for alignment purposes.


::

    magic   : 6 bytes
    version : 2 bytes
    headlen : 2 bytes   -> 10 bytes so far
     
    for headlen 0x0046 = 70 bytes, total of 80 bytes for the offset, 
    which matches 16*5 = 80 

*/

