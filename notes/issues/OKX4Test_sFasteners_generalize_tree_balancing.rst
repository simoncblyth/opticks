OKX4Test_sFasteners_generalize_tree_balancing
================================================

Following generalization can raytrace the balanced tree::

   LV=16 x4gen-csg   


Tree balancing lost all the bolts for this deep tree::

    epsilon:npy blyth$ TEST=NTreeBalanceTest om-t


    2018-08-31 17:10:46.885 INFO  [4668622] [test_process@189] tree initial 
    NTreeAnalyse height 11 count 25
                                                                                          un            

                                                                                  un              di    

                                                                          un          sp      sp      sp

                                                                  un          sp                        

                                                          un          sp                                

                                                  un          sp                                        

                                          un          sp                                                

                                  un          sp                                                        

                          un          sp                                                                

                  un          sp                                                                        

          di          sp                                                                                

      sp      sp                                                                                        


    2018-08-31 17:10:46.887 INFO  [4668622] [*NTreeBalance<nnode>::create_balanced@40] op_mask union intersection 
    2018-08-31 17:10:46.887 INFO  [4668622] [*NTreeBalance<nnode>::create_balanced@41] hop_mask union 
    2018-08-31 17:10:46.887 INFO  [4668622] [*NTreeBalance<nnode>::create_balanced@65]  bileafs 2 otherprim 9
    2018-08-31 17:10:46.887 ERROR [4668622] [NTreeBuilder<nnode>::init@169]  num_subs 2 num_otherprim 9 num_prim 13 height 4 mode MIXED operator union
    2018-08-31 17:10:46.887 INFO  [4668622] [test_process@194] tree result 
    NTreeAnalyse height 5 count 35
                                                                                  un                                                            

                                                  un                                                              un                            

                                  un                              un                              un                              un            

                  un                      un              un              un              un              un              un              un    

          in              in          sp      sp      sp      sp      sp      sp      sp      sp      sp      ze      ze      ze      ze      ze

      sp     !sp      sp     !sp                                                                                                                



With pruning::

    2018-08-31 17:14:16.879 INFO  [4670430] [*NTreeBalance<nnode>::create_balanced@65]  bileafs 2 otherprim 9
    2018-08-31 17:14:16.879 ERROR [4670430] [NTreeBuilder<nnode>::init@169]  num_subs 2 num_otherprim 9 num_prim 13 height 4 mode MIXED operator union
    2018-08-31 17:14:16.879 INFO  [4670430] [test_process@194] tree result 
    NTreeAnalyse height 5 count 25
                                                                                  un                    

                                                  un                                              un    

                                  un                              un                      un          sp

                  un                      un              un              un          sp      sp        

          in              in          sp      sp      sp      sp      sp      sp                        

      sp     !sp      sp     !sp                                                                        



This is not optimal, the bileafs need to be higher. Switching population order does that::


    2018-08-31 17:21:11.718 INFO  [4674115] [test_process@194] tree result 
    NTreeAnalyse height 4 count 25
                                                                  un                                    

                                  un                                                      un            

                  un                              un                      un                      in    

          un              un              un              un          sp          in          sp     !sp

      sp      sp      sp      sp      sp      sp      sp      sp              sp     !sp                




