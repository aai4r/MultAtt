# MultAtt

## Multimodal Attention (MultAtt) Module


## Dataset Download
  - CMU-MOSI: https://drive.google.com/file/d/16XtyD9kNwJlyimVSWJdDNTPpyQn_amOV/view?usp=sharing
  - CAER-S: https://caer-dataset.github.io/
  
## Preprocessing
  ### CMU-MOSI
  Already pre-trained features
  - Audio: COVAREP
  - Video: Facet
  - Text: BERT
  ### CAER-S
  Face cropping and masking by autocrop (https://github.com/leblancfg/autocrop/blob/master/LICENSE)
  1. Autocrop by runinng 
    ```
    ./CAER/autocrop/main.py
    ```
  2. Manually crop by running 
    ```
    ./CAER/autocrop/manual_crop.py 
    ```
    for the failed images
    
## Training & Evaluation
  ### CMU-MOSI
    run ./CMU-MOSI/train_transformer.py
    
  ### CAER-S
    run ./CAER/train.py
    
