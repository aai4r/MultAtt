# MultAtt

## Multimodal Attention (MultAtt) Module

<div>
<img width="270" src="https://user-images.githubusercontent.com/59352765/101871283-a6076d00-3bc6-11eb-93a0-e7c6cfe190c4.png">
<img width="550" src="https://user-images.githubusercontent.com/59352765/101871317-b3bcf280-3bc6-11eb-9734-62a7cf69cf85.png">
  
**Multimodal Sentiment Analysis**
  
  Dataset: CMU-MOSI
  
**Context-aware Emotion Recognition**
  
  Dataset: CAER-S

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
    
