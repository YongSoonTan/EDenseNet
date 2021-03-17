# EDenseNet  ["Hand Gesture Recognition via Enhanced Densely Connected Convolutional Neural Network"](https://doi.org/10.1016/j.eswa.2021.114797)
To be updated soon.
Implementation of EDenseNet (in tensorflow 1.15). For data augmentation, please refer to this ["repos"](https://github.com/YongSoonTan/CNN-SPP). 

if you find this code useful for your research, please consider citing (this citation will be updated again later):

    @article{tan2021hand,
      title={Hand Gesture Recognition via Enhanced Densely Connected Convolutional Neural Network},
      author={Tan, Yong Soon and Lim, Kian Ming and Lee, Chin Poo},
      journal={Expert Systems with Applications},
      pages={114797},
      year={2021},
      publisher={Elsevier}
    }
    
if GPU memory is not an issue, during testing, you can run all test images at once, just remove for loop in line 262 and line 378, and dedent the block of codes, and set the test images, labels and relevant variables accordingly. 

 ## Datasets (Please refer to this ["repos"](https://github.com/YongSoonTan/CNN-SPP))
 
 For ASL dataset, augmented training set is not provided, as they are too large to upload (2GB for each fold). However, training set without augmented data is provided, each fold of the training sets is compressed into 3 parts. you can reproduce training sets with augmented data using the Data_Aug.py file provided.
 
 For ASL with digits and NUS hand gesture dataset, training set with and without augmented data are both provided, where augmented training sets are compressed into parts.
 
 Data augmentation needs to be applied to the each fold of the training sets, images are not augmented in real-time during training. 
 To generate augmented data for ASL dataset, please read instruction on line 16 in Data_Aug.py.

| ASL                                                                                              
|---------------------------------------------------------------------------------------------------------
![ASL](https://github.com/YongSoonTan/CNN-SPP/blob/main/ASL.jpg)

| ASL with digits                                                                                               | NUS hand gesture                                            
|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------
| ![ASL_with_digits](https://github.com/YongSoonTan/CNN-SPP/blob/main/ASL_with_digits.jpg) | ![NUS](https://github.com/YongSoonTan/CNN-SPP/blob/main/NUS.jpg) |

| Data augmentation                                                                                              
|---------------------------------------------------------------------------------------------------------
![ASL](https://github.com/YongSoonTan/CNN-SPP/blob/main/Data_Augmentation.jpg)
