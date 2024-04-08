The repo is forked for EE451 Project, aiming to apply MobileNetV2 Network with Cpp CUDA and optimize its performance by parallelism. The origin readme.md is localed at pythonVersion/README.md

## Before you started
Attention, TensorFlow v2.16.1 cannot find all the cuda lib. Try to downgrade or try

```python
TF_CPP_MAX_VLOG_LEVEL=3 python -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))"
```
Some lib coule be no found and you should add them manually.

## Model Training
The model is too big for my 3070Ti or CARC so I switch to the A100 on Colab...

### Model structure
```
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]        0         []                            
                                                                                                  
 Conv1 (Conv2D)              (None, 112, 112, 32)         864       ['input_1[0][0]']             
                                                                                                  
 bn_Conv1 (BatchNormalizati  (None, 112, 112, 32)         128       ['Conv1[0][0]']               
 on)                                                                                              
                                                                                                  
 Conv1_relu (ReLU)           (None, 112, 112, 32)         0         ['bn_Conv1[0][0]']            
                                                                                                  
 expanded_conv_depthwise (D  (None, 112, 112, 32)         288       ['Conv1_relu[0][0]']          
 epthwiseConv2D)                                                                                  
                                                                                                  
 expanded_conv_depthwise_BN  (None, 112, 112, 32)         128       ['expanded_conv_depthwise[0][0
  (BatchNormalization)                                              ]']                           
                                                                                                  
 expanded_conv_depthwise_re  (None, 112, 112, 32)         0         ['expanded_conv_depthwise_BN[0
 lu (ReLU)                                                          ][0]']                        
                                                                                                  
 expanded_conv_project (Con  (None, 112, 112, 16)         512       ['expanded_conv_depthwise_relu
 v2D)                                                               [0][0]']                      
                                                                                                  
 expanded_conv_project_BN (  (None, 112, 112, 16)         64        ['expanded_conv_project[0][0]'
 BatchNormalization)                                                ]                             
                                                                                                  
 block_1_expand (Conv2D)     (None, 112, 112, 96)         1536      ['expanded_conv_project_BN[0][
                                                                    0]']                          
                                                                                                  
 block_1_expand_BN (BatchNo  (None, 112, 112, 96)         384       ['block_1_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_1_expand_relu (ReLU)  (None, 112, 112, 96)         0         ['block_1_expand_BN[0][0]']   
                                                                                                  
 block_1_pad (ZeroPadding2D  (None, 113, 113, 96)         0         ['block_1_expand_relu[0][0]'] 
 )                                                                                                
                                                                                                  
 block_1_depthwise (Depthwi  (None, 56, 56, 96)           864       ['block_1_pad[0][0]']         
 seConv2D)                                                                                        
                                                                                                  
 block_1_depthwise_BN (Batc  (None, 56, 56, 96)           384       ['block_1_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_1_depthwise_relu (Re  (None, 56, 56, 96)           0         ['block_1_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_1_project (Conv2D)    (None, 56, 56, 24)           2304      ['block_1_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_1_project_BN (BatchN  (None, 56, 56, 24)           96        ['block_1_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_2_expand (Conv2D)     (None, 56, 56, 144)          3456      ['block_1_project_BN[0][0]']  
                                                                                                  
 block_2_expand_BN (BatchNo  (None, 56, 56, 144)          576       ['block_2_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_2_expand_relu (ReLU)  (None, 56, 56, 144)          0         ['block_2_expand_BN[0][0]']   
                                                                                                  
 block_2_depthwise (Depthwi  (None, 56, 56, 144)          1296      ['block_2_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_2_depthwise_BN (Batc  (None, 56, 56, 144)          576       ['block_2_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_2_depthwise_relu (Re  (None, 56, 56, 144)          0         ['block_2_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_2_project (Conv2D)    (None, 56, 56, 24)           3456      ['block_2_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_2_project_BN (BatchN  (None, 56, 56, 24)           96        ['block_2_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_2_add (Add)           (None, 56, 56, 24)           0         ['block_1_project_BN[0][0]',  
                                                                     'block_2_project_BN[0][0]']  
                                                                                                  
 block_3_expand (Conv2D)     (None, 56, 56, 144)          3456      ['block_2_add[0][0]']         
                                                                                                  
 block_3_expand_BN (BatchNo  (None, 56, 56, 144)          576       ['block_3_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_3_expand_relu (ReLU)  (None, 56, 56, 144)          0         ['block_3_expand_BN[0][0]']   
                                                                                                  
 block_3_pad (ZeroPadding2D  (None, 57, 57, 144)          0         ['block_3_expand_relu[0][0]'] 
 )                                                                                                
                                                                                                  
 block_3_depthwise (Depthwi  (None, 28, 28, 144)          1296      ['block_3_pad[0][0]']         
 seConv2D)                                                                                        
                                                                                                  
 block_3_depthwise_BN (Batc  (None, 28, 28, 144)          576       ['block_3_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_3_depthwise_relu (Re  (None, 28, 28, 144)          0         ['block_3_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_3_project (Conv2D)    (None, 28, 28, 32)           4608      ['block_3_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_3_project_BN (BatchN  (None, 28, 28, 32)           128       ['block_3_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_4_expand (Conv2D)     (None, 28, 28, 192)          6144      ['block_3_project_BN[0][0]']  
                                                                                                  
 block_4_expand_BN (BatchNo  (None, 28, 28, 192)          768       ['block_4_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_4_expand_relu (ReLU)  (None, 28, 28, 192)          0         ['block_4_expand_BN[0][0]']   
                                                                                                  
 block_4_depthwise (Depthwi  (None, 28, 28, 192)          1728      ['block_4_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_4_depthwise_BN (Batc  (None, 28, 28, 192)          768       ['block_4_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_4_depthwise_relu (Re  (None, 28, 28, 192)          0         ['block_4_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_4_project (Conv2D)    (None, 28, 28, 32)           6144      ['block_4_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_4_project_BN (BatchN  (None, 28, 28, 32)           128       ['block_4_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_4_add (Add)           (None, 28, 28, 32)           0         ['block_3_project_BN[0][0]',  
                                                                     'block_4_project_BN[0][0]']  
                                                                                                  
 block_5_expand (Conv2D)     (None, 28, 28, 192)          6144      ['block_4_add[0][0]']         
                                                                                                  
 block_5_expand_BN (BatchNo  (None, 28, 28, 192)          768       ['block_5_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_5_expand_relu (ReLU)  (None, 28, 28, 192)          0         ['block_5_expand_BN[0][0]']   
                                                                                                  
 block_5_depthwise (Depthwi  (None, 28, 28, 192)          1728      ['block_5_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_5_depthwise_BN (Batc  (None, 28, 28, 192)          768       ['block_5_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_5_depthwise_relu (Re  (None, 28, 28, 192)          0         ['block_5_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_5_project (Conv2D)    (None, 28, 28, 32)           6144      ['block_5_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_5_project_BN (BatchN  (None, 28, 28, 32)           128       ['block_5_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_5_add (Add)           (None, 28, 28, 32)           0         ['block_4_add[0][0]',         
                                                                     'block_5_project_BN[0][0]']  
                                                                                                  
 block_6_expand (Conv2D)     (None, 28, 28, 192)          6144      ['block_5_add[0][0]']         
                                                                                                  
 block_6_expand_BN (BatchNo  (None, 28, 28, 192)          768       ['block_6_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_6_expand_relu (ReLU)  (None, 28, 28, 192)          0         ['block_6_expand_BN[0][0]']   
                                                                                                  
 block_6_pad (ZeroPadding2D  (None, 29, 29, 192)          0         ['block_6_expand_relu[0][0]'] 
 )                                                                                                
                                                                                                  
 block_6_depthwise (Depthwi  (None, 14, 14, 192)          1728      ['block_6_pad[0][0]']         
 seConv2D)                                                                                        
                                                                                                  
 block_6_depthwise_BN (Batc  (None, 14, 14, 192)          768       ['block_6_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_6_depthwise_relu (Re  (None, 14, 14, 192)          0         ['block_6_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_6_project (Conv2D)    (None, 14, 14, 64)           12288     ['block_6_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_6_project_BN (BatchN  (None, 14, 14, 64)           256       ['block_6_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_7_expand (Conv2D)     (None, 14, 14, 384)          24576     ['block_6_project_BN[0][0]']  
                                                                                                  
 block_7_expand_BN (BatchNo  (None, 14, 14, 384)          1536      ['block_7_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_7_expand_relu (ReLU)  (None, 14, 14, 384)          0         ['block_7_expand_BN[0][0]']   
                                                                                                  
 block_7_depthwise (Depthwi  (None, 14, 14, 384)          3456      ['block_7_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_7_depthwise_BN (Batc  (None, 14, 14, 384)          1536      ['block_7_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_7_depthwise_relu (Re  (None, 14, 14, 384)          0         ['block_7_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_7_project (Conv2D)    (None, 14, 14, 64)           24576     ['block_7_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_7_project_BN (BatchN  (None, 14, 14, 64)           256       ['block_7_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_7_add (Add)           (None, 14, 14, 64)           0         ['block_6_project_BN[0][0]',  
                                                                     'block_7_project_BN[0][0]']  
                                                                                                  
 block_8_expand (Conv2D)     (None, 14, 14, 384)          24576     ['block_7_add[0][0]']         
                                                                                                  
 block_8_expand_BN (BatchNo  (None, 14, 14, 384)          1536      ['block_8_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_8_expand_relu (ReLU)  (None, 14, 14, 384)          0         ['block_8_expand_BN[0][0]']   
                                                                                                  
 block_8_depthwise (Depthwi  (None, 14, 14, 384)          3456      ['block_8_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_8_depthwise_BN (Batc  (None, 14, 14, 384)          1536      ['block_8_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_8_depthwise_relu (Re  (None, 14, 14, 384)          0         ['block_8_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_8_project (Conv2D)    (None, 14, 14, 64)           24576     ['block_8_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_8_project_BN (BatchN  (None, 14, 14, 64)           256       ['block_8_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_8_add (Add)           (None, 14, 14, 64)           0         ['block_7_add[0][0]',         
                                                                     'block_8_project_BN[0][0]']  
                                                                                                  
 block_9_expand (Conv2D)     (None, 14, 14, 384)          24576     ['block_8_add[0][0]']         
                                                                                                  
 block_9_expand_BN (BatchNo  (None, 14, 14, 384)          1536      ['block_9_expand[0][0]']      
 rmalization)                                                                                     
                                                                                                  
 block_9_expand_relu (ReLU)  (None, 14, 14, 384)          0         ['block_9_expand_BN[0][0]']   
                                                                                                  
 block_9_depthwise (Depthwi  (None, 14, 14, 384)          3456      ['block_9_expand_relu[0][0]'] 
 seConv2D)                                                                                        
                                                                                                  
 block_9_depthwise_BN (Batc  (None, 14, 14, 384)          1536      ['block_9_depthwise[0][0]']   
 hNormalization)                                                                                  
                                                                                                  
 block_9_depthwise_relu (Re  (None, 14, 14, 384)          0         ['block_9_depthwise_BN[0][0]']
 LU)                                                                                              
                                                                                                  
 block_9_project (Conv2D)    (None, 14, 14, 64)           24576     ['block_9_depthwise_relu[0][0]
                                                                    ']                            
                                                                                                  
 block_9_project_BN (BatchN  (None, 14, 14, 64)           256       ['block_9_project[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_9_add (Add)           (None, 14, 14, 64)           0         ['block_8_add[0][0]',         
                                                                     'block_9_project_BN[0][0]']  
                                                                                                  
 block_10_expand (Conv2D)    (None, 14, 14, 384)          24576     ['block_9_add[0][0]']         
                                                                                                  
 block_10_expand_BN (BatchN  (None, 14, 14, 384)          1536      ['block_10_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_10_expand_relu (ReLU  (None, 14, 14, 384)          0         ['block_10_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_10_depthwise (Depthw  (None, 14, 14, 384)          3456      ['block_10_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_10_depthwise_BN (Bat  (None, 14, 14, 384)          1536      ['block_10_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_10_depthwise_relu (R  (None, 14, 14, 384)          0         ['block_10_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_10_project (Conv2D)   (None, 14, 14, 96)           36864     ['block_10_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_10_project_BN (Batch  (None, 14, 14, 96)           384       ['block_10_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_11_expand (Conv2D)    (None, 14, 14, 576)          55296     ['block_10_project_BN[0][0]'] 
                                                                                                  
 block_11_expand_BN (BatchN  (None, 14, 14, 576)          2304      ['block_11_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_11_expand_relu (ReLU  (None, 14, 14, 576)          0         ['block_11_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_11_depthwise (Depthw  (None, 14, 14, 576)          5184      ['block_11_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_11_depthwise_BN (Bat  (None, 14, 14, 576)          2304      ['block_11_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_11_depthwise_relu (R  (None, 14, 14, 576)          0         ['block_11_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_11_project (Conv2D)   (None, 14, 14, 96)           55296     ['block_11_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_11_project_BN (Batch  (None, 14, 14, 96)           384       ['block_11_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_11_add (Add)          (None, 14, 14, 96)           0         ['block_10_project_BN[0][0]', 
                                                                     'block_11_project_BN[0][0]'] 
                                                                                                  
 block_12_expand (Conv2D)    (None, 14, 14, 576)          55296     ['block_11_add[0][0]']        
                                                                                                  
 block_12_expand_BN (BatchN  (None, 14, 14, 576)          2304      ['block_12_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_12_expand_relu (ReLU  (None, 14, 14, 576)          0         ['block_12_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_12_depthwise (Depthw  (None, 14, 14, 576)          5184      ['block_12_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_12_depthwise_BN (Bat  (None, 14, 14, 576)          2304      ['block_12_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_12_depthwise_relu (R  (None, 14, 14, 576)          0         ['block_12_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_12_project (Conv2D)   (None, 14, 14, 96)           55296     ['block_12_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_12_project_BN (Batch  (None, 14, 14, 96)           384       ['block_12_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_12_add (Add)          (None, 14, 14, 96)           0         ['block_11_add[0][0]',        
                                                                     'block_12_project_BN[0][0]'] 
                                                                                                  
 block_13_expand (Conv2D)    (None, 14, 14, 576)          55296     ['block_12_add[0][0]']        
                                                                                                  
 block_13_expand_BN (BatchN  (None, 14, 14, 576)          2304      ['block_13_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_13_expand_relu (ReLU  (None, 14, 14, 576)          0         ['block_13_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_13_pad (ZeroPadding2  (None, 15, 15, 576)          0         ['block_13_expand_relu[0][0]']
 D)                                                                                               
                                                                                                  
 block_13_depthwise (Depthw  (None, 7, 7, 576)            5184      ['block_13_pad[0][0]']        
 iseConv2D)                                                                                       
                                                                                                  
 block_13_depthwise_BN (Bat  (None, 7, 7, 576)            2304      ['block_13_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_13_depthwise_relu (R  (None, 7, 7, 576)            0         ['block_13_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_13_project (Conv2D)   (None, 7, 7, 160)            92160     ['block_13_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_13_project_BN (Batch  (None, 7, 7, 160)            640       ['block_13_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_14_expand (Conv2D)    (None, 7, 7, 960)            153600    ['block_13_project_BN[0][0]'] 
                                                                                                  
 block_14_expand_BN (BatchN  (None, 7, 7, 960)            3840      ['block_14_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_14_expand_relu (ReLU  (None, 7, 7, 960)            0         ['block_14_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_14_depthwise (Depthw  (None, 7, 7, 960)            8640      ['block_14_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_14_depthwise_BN (Bat  (None, 7, 7, 960)            3840      ['block_14_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_14_depthwise_relu (R  (None, 7, 7, 960)            0         ['block_14_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_14_project (Conv2D)   (None, 7, 7, 160)            153600    ['block_14_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_14_project_BN (Batch  (None, 7, 7, 160)            640       ['block_14_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_14_add (Add)          (None, 7, 7, 160)            0         ['block_13_project_BN[0][0]', 
                                                                     'block_14_project_BN[0][0]'] 
                                                                                                  
 block_15_expand (Conv2D)    (None, 7, 7, 960)            153600    ['block_14_add[0][0]']        
                                                                                                  
 block_15_expand_BN (BatchN  (None, 7, 7, 960)            3840      ['block_15_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_15_expand_relu (ReLU  (None, 7, 7, 960)            0         ['block_15_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_15_depthwise (Depthw  (None, 7, 7, 960)            8640      ['block_15_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_15_depthwise_BN (Bat  (None, 7, 7, 960)            3840      ['block_15_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_15_depthwise_relu (R  (None, 7, 7, 960)            0         ['block_15_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_15_project (Conv2D)   (None, 7, 7, 160)            153600    ['block_15_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_15_project_BN (Batch  (None, 7, 7, 160)            640       ['block_15_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 block_15_add (Add)          (None, 7, 7, 160)            0         ['block_14_add[0][0]',        
                                                                     'block_15_project_BN[0][0]'] 
                                                                                                  
 block_16_expand (Conv2D)    (None, 7, 7, 960)            153600    ['block_15_add[0][0]']        
                                                                                                  
 block_16_expand_BN (BatchN  (None, 7, 7, 960)            3840      ['block_16_expand[0][0]']     
 ormalization)                                                                                    
                                                                                                  
 block_16_expand_relu (ReLU  (None, 7, 7, 960)            0         ['block_16_expand_BN[0][0]']  
 )                                                                                                
                                                                                                  
 block_16_depthwise (Depthw  (None, 7, 7, 960)            8640      ['block_16_expand_relu[0][0]']
 iseConv2D)                                                                                       
                                                                                                  
 block_16_depthwise_BN (Bat  (None, 7, 7, 960)            3840      ['block_16_depthwise[0][0]']  
 chNormalization)                                                                                 
                                                                                                  
 block_16_depthwise_relu (R  (None, 7, 7, 960)            0         ['block_16_depthwise_BN[0][0]'
 eLU)                                                               ]                             
                                                                                                  
 block_16_project (Conv2D)   (None, 7, 7, 320)            307200    ['block_16_depthwise_relu[0][0
                                                                    ]']                           
                                                                                                  
 block_16_project_BN (Batch  (None, 7, 7, 320)            1280      ['block_16_project[0][0]']    
 Normalization)                                                                                   
                                                                                                  
 Conv_1 (Conv2D)             (None, 7, 7, 1280)           409600    ['block_16_project_BN[0][0]'] 
                                                                                                  
 Conv_1_bn (BatchNormalizat  (None, 7, 7, 1280)           5120      ['Conv_1[0][0]']              
 ion)                                                                                             
                                                                                                  
 out_relu (ReLU)             (None, 7, 7, 1280)           0         ['Conv_1_bn[0][0]']           
                                                                                                  
 global_max_pooling2d (Glob  (None, 1280)                 0         ['out_relu[0][0]']            
 alMaxPooling2D)                                                                                  
                                                                                                  
 dense (Dense)               (None, 128)                  163968    ['global_max_pooling2d[0][0]']
                                                                                                  
 dense_1 (Dense)             (None, 64)                   8256      ['dense[0][0]']               
                                                                                                  
 dense_2 (Dense)             (None, 18)                   1170      ['dense_1[0][0]'] 
```

## Convert trained Model and read model with C++ 

```python
import tensorflow as tf
model = tf.keras.models.load_model('path_to_your_model/model.h5')
model.save('path_to_export_savedmodel', save_format='tf')
```

