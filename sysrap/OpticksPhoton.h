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

#pragma once

enum
{
    CERENKOV           = 0x1 <<  0,
    SCINTILLATION      = 0x1 <<  1,
    TORCH              = 0x1 <<  2,
    BULK_ABSORB        = 0x1 <<  3,
    BULK_REEMIT        = 0x1 <<  4,
    BULK_SCATTER       = 0x1 <<  5,
    SURFACE_DETECT     = 0x1 <<  6,
    SURFACE_ABSORB     = 0x1 <<  7,
    SURFACE_DREFLECT   = 0x1 <<  8,
    SURFACE_SREFLECT   = 0x1 <<  9,
    BOUNDARY_REFLECT   = 0x1 << 10,
    BOUNDARY_TRANSMIT  = 0x1 << 11,
    NAN_ABORT          = 0x1 << 12,
    EFFICIENCY_COLLECT = 0x1 << 13,
    EFFICIENCY_CULL    = 0x1 << 14,
    MISS               = 0x1 << 15,
    __NATURAL          = 0x1 << 16,
    __MACHINERY        = 0x1 << 17,
    __EMITSOURCE       = 0x1 << 18,
    PRIMARYSOURCE      = 0x1 << 19,
    GENSTEPSOURCE      = 0x1 << 20,
    DEFER_FSTRACKINFO  = 0x1 << 21
};

//
// FFS(flag) for the above is 1,2,3,4 etc..
// but that needs to fit within 4 bits for seq recording.
// Due to this limitation moved the rarely seen MISS to 0x1 << 15
// beyond seq recording to allow more important flags like
// EFFICIENCY_COLLECT to be seq recorded. See sseq_test::truncation
//
// Could use FFS(flag)-1 to gain another item, but "guest"
// flags of zero are a common bug that is useful to be able
// to record into the seq.
//



