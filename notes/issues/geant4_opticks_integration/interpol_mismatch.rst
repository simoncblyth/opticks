Interpol Mismatch
===================

::

   simon:tests blyth$ ipython -i OInterpolationTest_identity.py    ## perfect match when lookup at the input positions 
   simon:tests blyth$ ipython -i OInterpolationTest_interpol.py 


Probably all materials with non constant props are mismatching::

         .            Bakelite  True  
                           BPE  True  
         ADTableStainlessSteel  True  
                        MixGas  True  
                           PPE  True  
                       RadRock  True  
                          Foam  True  
                  OpaqueVacuum  True  
                           PVC  True  
                        Vacuum  True  
                        Silver  True  
            UnstStainlessSteel  True  
                         Tyvek  True  
                           ESR  True  
                StainlessSteel  True  
                           Air  True  
                          Iron  True  
                          Rock  True  
                         Nylon  True  
                     Aluminium  True  
                         Co_60  True  
                         Ge_68  True  
                          C_13  True  
                      Nitrogen  True  

            LiquidScintillator  False  [(21, 3, 0), (22, 0, 0), (24, 0, 0), (25, 0, 0), (42, 3, 0), (46, 3, 0), (71, 3, 0), (75, 3, 0)] 
                       Acrylic  False  [(20, 3, 0), (21, 0, 0), (22, 3, 0), (23, 0, 0), (37, 3, 0), (38, 0, 0), (44, 3, 0), (53, 0, 0), (54, 0, 0), (60, 3, 0), (61, 0, 0), (63, 0, 0), (67, 3, 0), (70, 3, 0)] 
                     GdDopedLS  False  [(23, 3, 0), (25, 3, 0), (26, 3, 0), (45, 3, 0), (72, 3, 0), (74, 3, 0)] 
                        Teflon  False  [(24, 3, 0), (26, 0, 0), (41, 3, 0)] 
                         Water  False  [(47, 3, 0), (48, 0, 0), (78, 0, 0)] 
                      OwsWater  False  [(14, 3, 0), (15, 0, 0), (99, 0, 0), (100, 0, 0), (101, 0, 0), (102, 0, 0), (103, 0, 0), (104, 0, 0), (105, 0, 0), (106, 0, 0), (107, 0, 0), (108, 0, 0), (109, 0, 0), (110, 0, 0), (111, 0, 0), (112, 0, 0), (113, 0, 0), (114, 0, 0), (115, 0, 0), (116, 0, 0), (117, 0, 0), (118, 0, 0), (119, 0, 0), (120, 0, 0)] 
                      Bialkali  False  [(29, 3, 0), (34, 3, 0)] 
                     DeadWater  False  [(12, 3, 0), (13, 0, 0), (121, 0, 0)] 
                   NitrogenGas  False  [(69, 3, 0), (70, 0, 0), (71, 0, 0), (72, 0, 0)] 
                         Pyrex  False  [(27, 3, 0), (28, 0, 0), (33, 3, 0), (81, 3, 0), (99, 3, 0)] 
                    MineralOil  False  [(19, 3, 0), (20, 0, 0), (27, 0, 0), (31, 0, 0), (32, 0, 0), (36, 0, 0), (37, 0, 0), (41, 0, 0), (42, 0, 0), (79, 3, 0)] 
                      IwsWater  False  [(16, 3, 0), (17, 0, 0), (17, 3, 0), (18, 0, 0), (47, 0, 0), (68, 0, 0), (76, 0, 0), (80, 0, 0), (81, 0, 0), (82, 0, 0), (83, 0, 0), (84, 0, 0), (85, 0, 0), (86, 0, 0), (87, 0, 0), (88, 0, 0), (89, 0, 0), (90, 0, 0), (91, 0, 0), (92, 0, 0), (93, 0, 0), (94, 0, 0), (95, 0, 0), (96, 0, 0), (97, 0, 0)] 



Seems that the tex settings just dont interpolate::


    In [7]: t[21,3,0,-5:]   ## last 5 from tex in standard 20nm steps 
    Out[7]: 
    array([[      1.479,    1862.275,  380174.5  ,       0.   ],
           [      1.479,     978.836,  420184.094,       0.   ],
           [      1.478,    3680.715,  460193.688,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    1966.001,  500000.   ,       0.   ]], dtype=float32)


    In [6]: o[21,3,0,-30:]   ## last 30 of 1nm steps, via tex interpol ... seems settings not interpolating
    Out[6]: 
    array([[      1.478,    3680.715,  460193.688,       0.   ],
           [      1.478,    3680.715,  460193.688,       0.   ],
           [      1.478,    3680.715,  460193.688,       0.   ],
           [      1.478,    3680.715,  460193.688,       0.   ],
           [      1.478,    3680.715,  460193.688,       0.   ],
           [      1.478,    3680.715,  460193.688,       0.   ],
           [      1.478,    3680.715,  460193.688,       0.   ],
           [      1.478,    3680.715,  460193.688,       0.   ],
           [      1.478,    3680.715,  460193.688,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    1966.001,  500000.   ,       0.   ]], dtype=float32)


    In [18]: c[21, 3, 0, -30:]   ## G4 interpolating away...
    Out[18]: 
    array([[      1.478,    3433.531,  481848.375,       0.   ],
           [      1.478,    3411.401,  483872.125,       0.   ],
           [      1.478,    3389.326,  485897.812,       0.   ],
           [      1.478,    3367.307,  487923.812,       0.   ],
           [      1.478,    3345.343,  489948.531,       0.   ],
           [      1.478,    3323.434,  491970.469,       0.   ],
           [      1.478,    3301.58 ,  493988.062,       0.   ],
           [      1.478,    3279.781,  495999.844,       0.   ],
           [      1.478,    3258.037,  498004.312,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3171.323,  500141.906,      -0.   ],
           [      1.478,    3106.461,  500276.781,      -0.   ],
           [      1.478,    3041.761,  500403.25 ,      -0.   ],
           [      1.478,    2977.222,  500519.938,      -0.   ],
           [      1.478,    2912.843,  500625.406,      -0.   ],
           [      1.478,    2848.623,  500718.375,      -0.   ],
           [      1.478,    2784.563,  500797.5  ,      -0.   ],
           [      1.478,    2720.662,  500861.469,      -0.   ],
           [      1.478,    2656.918,  500909.031,      -0.   ],
           [      1.478,    2593.332,  500938.875,      -0.   ],
           [      1.478,    2529.903,  500949.812,      -0.   ],
           [      1.478,    2466.63 ,  500940.562,      -0.   ],
           [      1.478,    2403.512,  500909.969,      -0.   ],
           [      1.478,    2340.55 ,  500856.781,      -0.   ],
           [      1.478,    2277.742,  500779.906,      -0.   ],
           [      1.478,    2215.088,  500678.156,      -0.   ],
           [      1.478,    2152.588,  500550.375,      -0.   ],
           [      1.478,    2090.24 ,  500395.5  ,      -0.   ],
           [      1.478,    2028.045,  500212.406,      -0.   ],
           [      1.478,    1966.001,  500000.   ,       0.   ]], dtype=float32)


::

    390 void OConfig::configureSampler(optix::TextureSampler& sampler, optix::Buffer& buffer)
    391 {
    392     LOG(trace) << "OPropertyLib::configureSampler" ;
    393 
    394     // cuda-pdf p43 // default is to clamp to the range
    395     RTwrapmode wrapmode = RT_WRAP_REPEAT ;
    396     //RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_EDGE ;  // <--- seems not supported 
    397     //RTwrapmode wrapmode = RT_WRAP_MIRROR ;
    398     //RTwrapmode wrapmode = RT_WRAP_CLAMP_TO_BORDER ;  // return zero when out of range
    399     sampler->setWrapMode(0, wrapmode);
    400     sampler->setWrapMode(1, wrapmode);
    401 
    402     //RTfiltermode filtermode = RT_FILTER_NEAREST ; 
    403     RTfiltermode filtermode = RT_FILTER_LINEAR ;
    404     RTfiltermode minification = filtermode ;
    405     RTfiltermode magnification = filtermode ;
    406     RTfiltermode mipmapping = RT_FILTER_NONE ;
    407 
    408     sampler->setFilteringModes(minification, magnification, mipmapping);
    409 
    410     //RTtexturereadmode readmode = RT_TEXTURE_READ_NORMALIZED_FLOAT ;
    411     RTtexturereadmode readmode = RT_TEXTURE_READ_ELEMENT_TYPE ;    // No conversion
    412     sampler->setReadMode(readmode);
    413 
    414     //RTtextureindexmode indexingmode = RT_TEXTURE_INDEX_ARRAY_INDEX ;  // by inspection : zero based array index offset by 0.5 (fails to validate in OptiX 400)
    415     RTtextureindexmode indexingmode = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES ;
    416     sampler->setIndexingMode(indexingmode);
    417 
    418 
    419     sampler->setMaxAnisotropy(1.0f);
    420     sampler->setMipLevelCount(1u);
    421     sampler->setArraySize(1u);
    422     //   from 3.8 pdf: OptiX currently supports only a single MIP level and a single element texture array.
    423 
    424     unsigned int texture_array_idx = 0u ;
    425     unsigned int mip_level = 0u ;
    426     sampler->setBuffer(texture_array_idx, mip_level, buffer);  // deprecated in OptiX 4
    427 
    428 }


Seems the advertized lineral interpolation is not happening  ?



::

    7195   /**
    7196   * @brief Sets the filtering modes of a texture sampler
    7197   *
    7198   * @ingroup TextureSampler
    7199   *
    7200   * <B>Description</B>
    7201   *
    7202   * @ref rtTextureSamplerSetFilteringModes sets the minification, magnification and MIP mapping filter modes for \a texturesampler.
    7203   * RTfiltermode must be one of the following values:
    7204   *
    7205   *  - @ref RT_FILTER_NEAREST
    7206   *  - @ref RT_FILTER_LINEAR
    7207   *  - @ref RT_FILTER_NONE
    7208   *
    7209   * These filter modes specify how the texture sampler will interpolate
    7210   * buffer data that has been attached to it.  \a minification and
    7211   * \a magnification must be one of @ref RT_FILTER_NEAREST or
    7212   * @ref RT_FILTER_LINEAR.  \a mipmapping may be any of the three values but
    7213   * must be @ref RT_FILTER_NONE if the texture sampler contains only a
    7214   * single MIP level or one of @ref RT_FILTER_NEAREST or @ref RT_FILTER_LINEAR
    7215   * if the texture sampler contains more than one MIP level.
    7216   *
    7217   * @param[in]   texturesampler   The texture sampler object to be changed
    7218   * @param[in]   minification     The new minification filter mode of the texture sampler
    7219   * @param[in]   magnification    The new magnification filter mode of the texture sampler
    7220   * @param[in]   mipmapping       The new MIP mapping filter mode of the texture sampler
    7221   *
    7222   * <B>Return values</B>
    7223   *
    7224   * Relevant return values:
    7225   * - @ref RT_SUCCESS
    7226   * - @ref RT_ERROR_INVALID_CONTEXT
    7227   * - @ref RT_ERROR_INVALID_VALUE
    7228   *
    7229   * <B>History</B>
    7230   *
    7231   * @ref rtTextureSamplerSetFilteringModes was introduced in OptiX 1.0.
    7232   *
    7233   * <B>See also</B>
    7234   * @ref rtTextureSamplerGetFilteringModes
    7235   *
    7236   */
    7237   RTresult RTAPI rtTextureSamplerSetFilteringModes(RTtexturesampler texturesampler, RTfiltermode  minification, RTfiltermode  magnification, RTfiltermode mipmapping);
    7238 



Issue not with tex settings but with boundary_lookup.h.
After avoiding int-ization on 20nm boundaries, some interpol is happening::

    In [1]: t[21,3,0,-5:]
    Out[1]: 
    array([[      1.479,    1862.275,  380174.5  ,       0.   ],
           [      1.479,     978.836,  420184.094,       0.   ],
           [      1.478,    3680.715,  460193.688,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    1966.001,  500000.   ,       0.   ]], dtype=float32)

    In [2]: o[21,3,0,-30:]
    Out[2]: 
    array([[      1.478,    3435.965,  482118.25 ,       0.   ],
           [      1.478,    3413.4  ,  484139.688,       0.   ],
           [      1.478,    3392.57 ,  486005.594,       0.   ],
           [      1.478,    3370.004,  488027.   ,       0.   ],
           [      1.478,    3347.439,  490048.438,       0.   ],
           [      1.478,    3324.873,  492069.844,       0.   ],
           [      1.478,    3302.308,  494091.25 ,       0.   ],
           [      1.478,    3281.478,  495957.156,       0.   ],
           [      1.478,    3258.912,  497978.594,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3171.837,  500000.   ,       0.   ],
           [      1.478,    3107.327,  500000.   ,       0.   ],
           [      1.478,    3047.78 ,  500000.   ,       0.   ],
           [      1.478,    2983.27 ,  500000.   ,       0.   ],
           [      1.478,    2918.76 ,  500000.   ,       0.   ],
           [      1.478,    2854.25 ,  500000.   ,       0.   ],
           [      1.478,    2789.741,  500000.   ,       0.   ],
           [      1.478,    2730.193,  500000.   ,       0.   ],
           [      1.478,    2665.684,  500000.   ,       0.   ],
           [      1.478,    2601.174,  500000.   ,       0.   ],
           [      1.478,    2536.664,  500000.   ,       0.   ],
           [      1.478,    2472.154,  500000.   ,       0.   ],
           [      1.478,    2412.607,  500000.   ,       0.   ],
           [      1.478,    2348.097,  500000.   ,       0.   ],
           [      1.478,    2283.587,  500000.   ,       0.   ],
           [      1.478,    2219.077,  500000.   ,       0.   ],
           [      1.478,    2154.568,  500000.   ,       0.   ],
           [      1.478,    2095.02 ,  500000.   ,       0.   ],
           [      1.478,    2030.51 ,  500000.   ,       0.   ],
           [      1.478,    1966.001,  500000.   ,       0.   ]], dtype=float32)

    In [3]: c[21,3,0,-30:]    ## G4 is inventing noise in its interpolation ???
    Out[3]: 
    array([[      1.478,    3433.531,  481848.375,       0.   ],
           [      1.478,    3411.401,  483872.125,       0.   ],
           [      1.478,    3389.326,  485897.812,       0.   ],
           [      1.478,    3367.307,  487923.812,       0.   ],
           [      1.478,    3345.343,  489948.531,       0.   ],
           [      1.478,    3323.434,  491970.469,       0.   ],
           [      1.478,    3301.58 ,  493988.062,       0.   ],
           [      1.478,    3279.781,  495999.844,       0.   ],
           [      1.478,    3258.037,  498004.312,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3171.323,  500141.906,      -0.   ],
           [      1.478,    3106.461,  500276.781,      -0.   ],
           [      1.478,    3041.761,  500403.25 ,      -0.   ],
           [      1.478,    2977.222,  500519.938,      -0.   ],
           [      1.478,    2912.843,  500625.406,      -0.   ],
           [      1.478,    2848.623,  500718.375,      -0.   ],
           [      1.478,    2784.563,  500797.5  ,      -0.   ],
           [      1.478,    2720.662,  500861.469,      -0.   ],
           [      1.478,    2656.918,  500909.031,      -0.   ],
           [      1.478,    2593.332,  500938.875,      -0.   ],
           [      1.478,    2529.903,  500949.812,      -0.   ],
           [      1.478,    2466.63 ,  500940.562,      -0.   ],
           [      1.478,    2403.512,  500909.969,      -0.   ],
           [      1.478,    2340.55 ,  500856.781,      -0.   ],
           [      1.478,    2277.742,  500779.906,      -0.   ],
           [      1.478,    2215.088,  500678.156,      -0.   ],
           [      1.478,    2152.588,  500550.375,      -0.   ],
           [      1.478,    2090.24 ,  500395.5  ,      -0.   ],
           [      1.478,    2028.045,  500212.406,      -0.   ],
           [      1.478,    1966.001,  500000.   ,       0.   ]], dtype=float32)

::

    In [1]: c[21,3,0,-30:]    ## switch off spline interpol in CPropLib::addProperty
    Out[1]: 
    array([[      1.478,    3433.531,  482336.25 ,       0.   ],
           [      1.478,    3411.401,  484318.719,       0.   ],
           [      1.478,    3389.326,  486296.188,       0.   ],
           [      1.478,    3367.307,  488268.656,       0.   ],
           [      1.478,    3345.343,  490236.188,       0.   ],
           [      1.478,    3323.434,  492198.75 ,       0.   ],
           [      1.478,    3301.58 ,  494156.406,       0.   ],
           [      1.478,    3279.781,  496109.156,       0.   ],
           [      1.478,    3258.037,  498057.   ,       0.   ],
           [      1.478,    3236.347,  500000.   ,       0.   ],
           [      1.478,    3171.323,  500000.   ,       0.   ],
           [      1.478,    3106.461,  500000.   ,       0.   ],
           [      1.478,    3041.761,  500000.   ,       0.   ],
           [      1.478,    2977.222,  500000.   ,       0.   ],
           [      1.478,    2912.843,  500000.   ,       0.   ],
           [      1.478,    2848.623,  500000.   ,       0.   ],
           [      1.478,    2784.563,  500000.   ,       0.   ],
           [      1.478,    2720.662,  500000.   ,       0.   ],
           [      1.478,    2656.918,  500000.   ,       0.   ],
           [      1.478,    2593.332,  500000.   ,       0.   ],
           [      1.478,    2529.903,  500000.   ,       0.   ],
           [      1.478,    2466.63 ,  500000.   ,       0.   ],
           [      1.478,    2403.512,  500000.   ,       0.   ],
           [      1.478,    2340.55 ,  500000.   ,       0.   ],
           [      1.478,    2277.742,  500000.   ,       0.   ],
           [      1.478,    2215.088,  500000.   ,       0.   ],
           [      1.478,    2152.588,  500000.   ,       0.   ],
           [      1.478,    2090.24 ,  500000.   ,       0.   ],
           [      1.478,    2028.045,  500000.   ,       0.   ],
           [      1.478,    1966.001,  500000.   ,       0.   ]], dtype=float32)



