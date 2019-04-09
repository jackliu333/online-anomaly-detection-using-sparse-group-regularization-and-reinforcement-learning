# Research Porposal
Anomaly detection, or change point detection, has been a popular research topic in recent years. Most statistical methods uses adaptive forecasting as baseline model based on certain window size, and classifies the new data point as anomalous if it deviates from baseline prediction by some threshold. The choice of window size and threshold thus affect the quality of the anomaly detection model, and should change based on domain and time period. To avoid such continuous parameter tuning, we introduce reinforcement learning to consistently learn the optimal policy on parameter tuning and anomaly detection. In addition, we employ group sparse lasso for anomaly estimation, which enables us to find the location of the anomaly with statistical significance.

### Reinforcement learning
We use reinforcement learning for anomaly detection that features:
- no assumption on the concept of anomaly, thus learn the concept purely from the training dataset
- threshold free, i.e. no need to define window size as in adpative forecasting approach
- dynamically improving, with the accumulation of the anomaly detection experience by learning new anomalies and consistently enhancing its knowledge for anomaly detection

