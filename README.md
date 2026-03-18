# TimeAPN: Adaptive Amplitude-Phase Non-Stationarity Normalization for Time Series Forecasting
This repo is the official Pytorch implementation of our paper.

## Introduction
In this paper, we propose TimeAPN, an Adaptive Amplitude–Phase Non-Stationarity Normalization framework that explicitly models and predicts non-stationary factors from both the time and frequency domains. Specifically, TimeAPN first models the mean sequence jointly in the time and frequency domains, and then forecasts its evolution over future horizons. Meanwhile, phase information is extracted in the frequency domain, and the phase discrepancy between the predicted and ground-truth future sequences is explicitly modeled to capture temporal misalignment. Furthermore, TimeAPN incorporates amplitude information into an adaptive normalization mechanism, enabling the model to effectively account for abrupt fluctuations in signal energy. The predicted non-stationary factors are subsequently integrated with the backbone forecasting outputs through a collaborative de-normalization process to reconstruct the final non-stationary time series. The
proposed framework is model-agnostic and can be seamlessly integrated with various forecasting backbones. 
![image](https://github.com/y563642-max/timeapn/blob/main/figs/framework.png)

We perform comparative experiments across seven widely-used datasets using popular forecasting backbones, and further evaluate TimeAPN's performance against other reversible non-stationary approaches.

Multivariate forecasting results:
![image](https://github.com/y563642-max/timeapn/blob/main/figs/multivariate.jpg)

Comparison with other reversible non-stationary approaches:
![image]()

## Usage
### Environment and dataset setup
```
pip install -r requirements.txt
mkdir datasets
```

All the datasets are available at the Google Driver provided by Autoformer. Many thanks to their efforts and devotion!

## Running
We provide ready-to-use scripts for different backbone models. For example, to run TimeAPN with DLinear:
```
bash ./scripts/DLinear.sh # scripts for DLinear
```

## Acknowledgement
This repo is built on the pioneer works. We appreciate the following GitHub repos a lot for their valuable code base or datasets:

[FEDformer](https://github.com/MAZiqing/FEDformer)

[DLinear](https://github.com/cure-lab/LTSF-Linear)

[PatchTST](https://github.com/yuqinie98/PatchTST)

[S_Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba)

[DDN](https://github.com/Hank0626/DDN)

## Citation
If you find our work helpful, please consider citing our paper:
