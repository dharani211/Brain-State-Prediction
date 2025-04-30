Functional magnetic resonance imaging (fMRI) serves as an essential tool for studying human brain activity by measuring blood oxygen level-dependent (BOLD) signals. Recent developments in deep learning, especially transformer-based
models, have demonstrated significant potential in forecasting brain states using resting-state fMRI (rs-fMRI) data. However, challenges such as error accumulation and temporal resolution
constraints remain significant. Building on existing research, this project proposes an enhancement to transformer-based brain state prediction by incorporating multi-scale temporal attention mechanisms, which can better capture long-term dependencies
while maintaining fine-grained temporal resolution. Prior research has demonstrated the potential of transformers in brain state modeling, such as the BrainLM model, which achieved superior accuracy but required extensive training on
large datasets. Additionally, the use of autoregressive transformer models has been explored in various neuroscience applications, with results indicating promising but limited scalability. Ourproposed approach will introduce hierarchical attention layers to
model temporal granularity at multiple scales, aiming to improve predictive accuracy and robustness. This study will utilize high-resolution fMRI data from the Human Connectome Project (HCP) to implement a multi-scale
transformer model that adaptively modulates attention mecha-nisms across various temporal scales. The proposed model is expected to achieve lower mean squared error (MSE) and higher
correlation with true fMRI data, compared to baseline models. Future work will explore personalization of predictions using transfer learning and explainability techniques to provide deeper
insights into functional connectivity patterns.

For the Project Manual please check below link:
https://docs.google.com/document/d/1T0j8c8Qw1wXt-2h8eE6TbnL8yjMbZlO_YoPjWHjAMCo/edit?usp=sharing
