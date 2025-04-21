# STEAD: Spatio-Temporal Efficient Anomaly Detection for Time and Compute Sensitive Applications

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stead-spatio-temporal-efficient-anomaly-1/anomaly-detection-in-surveillance-videos-on)](https://paperswithcode.com/sota/anomaly-detection-in-surveillance-videos-on?p=stead-spatio-temporal-efficient-anomaly-1)
This repo is the official implementation of [STEAD: Spatio-Temporal Efficient Anomaly Detection for Time and Compute Sensitive Applications](https://arxiv.org/abs/2503.07942)  

### Pretrained models available in the saved_models folder

**Extracted X3D Features for UCF-Crime dataset**

[**UCF-Crime X3D Features on Google drive**](https://drive.google.com/file/d/1LBTddU2mKuWvpbFOrqylJrZQ4u-U-zxG/view?usp=sharing)  

Feature extraction code also available for modification  

#### Prepare the environment: 
        pip install -r requirements.txt
#### Test: Run 
        python test.py
#### Train: Modify the option.py and run 
        python main.py

## Citation
    @misc{gao2025steadspatiotemporalefficientanomaly,
          title={STEAD: Spatio-Temporal Efficient Anomaly Detection for Time and Compute Sensitive Applications}, 
          author={Andrew Gao and Jun Liu},
          year={2025},
          eprint={2503.07942},
          archivePrefix={arXiv},
          primaryClass={cs.CV},
          url={https://arxiv.org/abs/2503.07942}, 
    }
