stree_MOI_SID_frame_addressing_is_off_by_one_as_revealed_by_the_staked_PMT.rst
===============================================================================

Avoid confusion, just document the off-by-one::

    In [1]: idtab
    Out[1]: 
    [idtab instanceId - MOI=SID:$(( nnn-1 )) to access frames
    array([[     0, 461340,      0],
           [ 50056,  74539,   9021],
           [ 50086,  12487,   9001],
           [ 50087,   1890,  15209],
           [ 50146,    559,  17249],
           [ 50001,    134, 311415],
           [ 50147,     51,  25163]])
    ]idtab instanceId - MOI=SID:$(( nnn-1 )) to access frames






cxt_min.sh simtracing with the below frame shows the overlap::

   moi=-6000,0,19700,500 ; moi_note="$LINENO close to where the stake goes thru mask_PMT_20inch_vetosMask_virtual"


Identity tables show which volumes involved::

    In [2]: iitab
    Out[2]: 
    [iitab
    array([[     0, 456008,      0],
           [ 43252,  74539,   9021],
           [ 43282,  12487,   9001],
           [ 49185,   5332,   9066],
           [ 43283,   1890,  15209],
           [ 43342,    559,  17249],
           [ 43197,    134, 311415],
           [ 43343,     51,  25163]])
    ]iitab

    In [3]: idtab
    Out[3]: 
    [idtab
    array([[     0, 461340,      0],
           [ 50056,  74539,   9021],     ## THIS ID LOOKS TO BE +1 WITH 0 MEANING NOT-A-SENSOR
           [ 50086,  12487,   9001],
           [ 50087,   1890,  15209],
           [ 50146,    559,  17249],
           [ 50001,    134, 311415],
           [ 50147,     51,  25163]])
    ]idtab

    In [4]: gptab
    Out[4]: 
    [gptab
    array([['266', '85914', '9010', 's_EMF_bar_box_810mm'],
           ['2221', '82571', '2', 'GLb3.up09_FlangeI_Web_FlangeII'],
           ['1519', '70091', '0', 'sOuterWaterInCD_T'],
           ['209', '66432', '9030', 'sAirGap'],
           ['2836', '63510', '3', 'sInnerReflectorInCD_T'],
           ['3096', '56375', '9001', 'mask_PMT_20inch_vetosMask_virtual'],
           ['1518', '52815', '9000', 'sOuterReflectorInCD_T'],
           ['3099', '25494', '332000', 'PMT_20inch_veto_inner_solid_1_2'],
           ['213', '9023', '11196', 'sTyvek_shell'],
           ['2191', '7753', '67005', 'GLb4.up10_FlangeI_Web_FlangeII'],
           ['3097', '6197', '277003', 'mask_PMT_20inch_vetosMask'],
           ['256', '5937', '260010', 's_EMF_bar_box_810mm'],
           ['225', '4641', '9018', 's_EMFsupport_ring3'],
           ['3265', '3451', '9066', 's_EMFcoil_holder_ring11_seg1'],
           ['301', '2234', '14071', 's_EMF_bar_box_930mm'],
           ['2251', '1937', '19', 'GLb2.up08_FlangeI_Web_FlangeII'],
           ['3098', '1594', '390004', 'PMT_20inch_veto_pmt_solid_1_2'],
           ['226', '883', '12008', 's_EMFsupport_ring4'],
           ['3270', '882', '11085', 's_EMFcoil_holder_ring12_seg1'],
           ['224', '724', '260018', 's_EMFsupport_ring2'],
           ['3262', '615', '260030', 's_EMFcoil_holder_ring10_seg1'],
           ['2161', '412', '204347', 'GLb3.up11_FlangeI_Web_FlangeII'],
           ['2281', '406', '119', 'GLb2.up07_FlangeI_Web_FlangeII'],
           ['3275', '384', '13283', 's_EMFcoil_holder_ring13_seg1'],
           ['227', '379', '14091', 's_EMFsupport_ring5'],
           ['2311', '333', '479', 'GLb2.up06_FlangeI_Web_FlangeII'],
           ['2341', '13', '5227', 'GLb1.up05_FlangeI_Web_FlangeII']], dtype='<U44')
    ]gptab



Ray trace viz using frames selected using above idenity indices::
 
     MOI=PRIM:266 cxr_min.sh               ## finds the staked PMT, whacky global PRIM frame
     MOI=INST:43252 EYE=3,3,3 cxr_min.sh   ## also finds it, natural INST frame

Oops this SID finds adjacent PMT::

     MOI=SID:50056 EYE=3,3,3 cxr_min.sh   ## finds adjacent PMT

Subtracting one from SID index finds it::

     MOI=SID:50055 EYE=3,3,3 cxr_min.sh   


GEOM st::

    In [10]: f.inst[43252]
    Out[10]: 
    array([[   -0.961,     0.005,    -0.278,     0.   ],
           [   -0.005,    -1.   ,     0.   ,     0.   ],
           [   -0.278,     0.001,     0.961,     0.   ],
           [-5700.183,    29.893, 19693.629,     0.   ]])


    In [9]: f.inst[43252,:,3].view(np.uint64)
    Out[9]: array([43252,     4, 50055, 43256], dtype=uint64)

    ## inst_idx, gas_idx, sen_id, sen_idx



The +1 for sensor_identifier is old, whats new is SID addressing::

    367     QAT4_METHOD int get_IAS_OptixInstance_instanceId() const
    368     {
    369         const int& sensor_identifier = q2.i.w ;
    370         assert( sensor_identifier >= 0 );  // 0 means not a sensor GPU side, so subtract 1 to get actual sensorId
    371         return sensor_identifier ;
    372     }
    373 
    374     /**
    375     sqat4::setIdentity
    376     -------------------
    377 
    378     Canonical usage from CSGFoundry::addInstance  where sensor_identifier gets +1
    379     with 0 meaning not a sensor.
    380     **/
    381 
    382     QAT4_METHOD void setIdentity(int ins_idx, int gas_idx, int sensor_identifier_1, int sensor_index )
    383     {
    384         assert( sensor_identifier_1 >= 0 );
    385 
    386         q0.i.w = ins_idx ;               // formerly unsigned and "+ 1"
    387         q1.i.w = gas_idx ;
    388         q2.i.w = sensor_identifier_1 ;   // now +1 with 0 meaning not-a-sensor
    389         q3.i.w = sensor_index ;
    390     }


Change variable name to clarify::

    QAT4_METHOD int get_IAS_OptixInstance_instanceId() const
    {   
        const int& sensor_identifier_1 = q2.i.w ;
        assert( sensor_identifier_1 >= 0 );  // 0 means not a sensor GPU side
        return sensor_identifier_1 ;    // NB this is +1,  subtract 1 to get original sensorId of sensors
    }   




