# Research Porposal
Anomaly detection, or change point detection, has been a popular research topic in recent years. Most statistical methods uses adaptive forecasting as baseline model based on certain window size, and classifies the new data point as anomalous if it deviates from baseline prediction by some threshold. The choice of window size and threshold thus affect the quality of the anomaly detection model, and should change based on domain and time period. To avoid such continuous parameter tuning, we introduce reinforcement learning to consistently learn the optimal policy on parameter tuning and anomaly detection. In addition, we employ group sparse lasso for anomaly estimation, which enables us to find the location of the anomaly with statistical significance.

### Reinforcement learning
We use reinforcement learning for anomaly detection that features:
- no assumption on the concept of anomaly, thus learn the concept purely from the training dataset
- threshold free, i.e. no need to define window size as in adpative forecasting approach
- dynamically improving, with the accumulation of the anomaly detection experience by learning new anomalies and consistently enhancing its knowledge for anomaly detection

We define anomaly detector $\pi$ as the conditional probabiliyt distribution $\pi:=p(A|S)$ where $S$ and $A$ denote the set of states in the target system and the set of actions respectively. Action is divided into two classes: changing window size for historical data or making a anomaly classification on current data point. Reward is defined as the given time series value or classification score obtained in advance, depending on the specific use case. State is defined as the currently selected data points and associated value. 

Anomaly detector performance is measured by $V_{\pi}= \sum_{s \in S}d^{\pi}(s)\sum_{a \in A}Q(s,a)\cdot \pi(s,a)$, where $d^{\pi}(s)$ is the probability of the target system being in state s following policy $\pi$, and $Q(s,a)$ represents the accumulated reward started from state $s$ with action $a$. The optimal policy is selected by $\pi^*=argmax_{\pi}V_{\pi}$.

Exeprience $E$ is a set of tuples $<s,a,r,s'>$. By learning from E, i.e. using experience replay, the RL agent is able to consistently improving its policy. 

### Anomaly estimation via sparse group regularization

We consider linear regression model
\begin{equation}
    y_t=\beta^T_tx_t+\epsilon_t,t=1,2,..,n
\end{equation}
where $x_t$ is a $p$ dimensional vector, $\beta_t$ is a $p$ dimensional sparse coefficients vector, $p$ represents the number of pixel values in a video frame, and $\epsilon$ is observation noise which is assumed to be i.i.d. following $N(0,\sigma^2)$. By directly modelling on each pixel value, we assume that that values of $\beta_t$ change over time, the linear model experiences $K^*$ times of changes in $\beta_t$, and the set of change points are denoted as $T^*=\{t_k^*,k=1,...,K*\}$. For $1 \leq k \leq K^*+1$, denote $\beta_t=\alpha^*_k$ for $t^*_{k-1}\leq t \leq t^*_k-1$ and $\alpha^*_k\neq\alpha^*_{k-1}$, where $t^*_0=1$, $t^*_{K^*+1}=n+1$, and $\{\alpha^*_k,k=1,...,K^*\}$ are the true values of coefficients that are fixed but unknown. The goal is to estimate the change points $\{t_k^*\}$, the coefficients $\{\alpha^*_k\}$ and the number of change points $K^*$ through $n$ pairs of observed and predicted data $(x_t,y_t)$.


Let $K_{max}$ be a known upper bound on the number of change points and $K_{max} \leq n$. The multiple change points estimation problem can be written as
\begin{equation}
    \begin{aligned}
    \underset{\beta}{min} \frac{1}{n}\sum^n_{t=1}(y_t-\beta^T_tx_t)^2 \\
    s.t. \sum^{n-1}_{t=1}1\{\beta_{t+1}\neq \beta_t\} \leq K_{max}
    \end{aligned}
\end{equation}


Let $\theta_1=\beta_1$ and $\theta_t=\beta_t-\beta_{t-1}$ for $t=2,..,n$, $\beta=(\beta^T_1,\beta^T_2,...,\beta^T_n)^T$,$\theta=(\theta^T_1,\theta^T_2,...,\theta^T_n)^T$, where $\beta$ and $\theta$ are both $np$ dimensional column vectors. Under the sparsity assumption, most of $\{\theta_t\}$ are zero vectors with at most $K_{max}$ non-zero $\{\theta_t\}$ vectors, of which most entries are also zero.Thus if we view $\{\theta_t\}$ as groups within $\theta$, then it has the following group sparse structure: most of the groups are zero, and for those non-zero groups, most of the entries within the group are zero.

Let $Y=(y_1,y_2,...,y_n)^T$, $E=(\epsilon_1,\epsilon_2,...,\epsilon_n)^T$, and 


$$
X = \left(
\begin{array}{cccc}
     x^T_1  \\
      \\
      \\
      \\
\end{array}
\begin{array}{cccc}
      \\
     x^T_2 \\
      \\
      \\
\end{array}
\begin{array}{cccc}
      \\
      \\
      ... \\
      \\
\end{array}
\begin{array}{cccc}
      \\
      \\
      \\
     x^T_n  \\
\end{array}
\right )
$$



$$
\tilde A = \left(
\begin{array}{cccc}
     I_p  \\
     I_p \\
     ... \\
     I_p \\
\end{array}
\begin{array}{cccc}
         \\
      I_p \\
      ... \\
      I_p \\
\end{array}
\begin{array}{cccc}
       \\
       \\
      ... \\
      ... \\
\end{array}
\begin{array}{cccc}
      \\
      \\
      \\
      I_p \\
\end{array}
\right )
$$

where $I_p$ is an identity matrix of size $p$ x $p$, $Y$ and $E$ are vectors of dimension $n$, and $X$ and $\tilde A$ are matrices of size $n$ x $np$ and $np$ x $np$ respectively. Define

$$
\tilde X = X\tilde A = \left(
\begin{array}{ccccc}
     x^T_1  \\
     x^T_2 \\
     x^T_3 \\
     ... \\
     x^T_n \\
\end{array}
\begin{array}{ccccc}
         \\
      x^T_2 \\
      x^T_3 \\
      ... \\
      ... \\
\end{array}
\begin{array}{ccccc}
       \\
       \\
      x^T_3 \\
      ... \\
      ... \\
\end{array}
\begin{array}{ccccc}
      \\
      \\
      \\
      ... \\
      ... \\
\end{array}
\begin{array}{ccccc}
      \\
      \\
      \\
      \\
      x^T_n \\
\end{array}
\right )
$$

Thus the model can be rewritten as
\begin{equation}
    Y=X\beta+E=\tilde X\theta+E
\end{equation}

To obtain the estimates on the number of locations of change points and linear coefficients of each region, let $\beta_0=0_{p}$, the objective function with group LASSO regularization is 

\begin{equation}
    \underset{\beta}{min} \frac{1}{n}\Vert Y-X\beta \Vert^2+\gamma\lambda_n\sum^n_{t=1}\Vert \beta_t-\beta_{t-1} \Vert_2 + (1-\gamma)\lambda_n\sum^n_{t=1} \Vert \beta_t - \beta_{t-1} \Vert_1
\end{equation}

which can also be rewritten as

\begin{equation}
    \underset{\beta}{min} \frac{1}{n}\Vert Y-\tilde X\theta \Vert^2+\gamma\lambda_n\sum^n_{t=1}\Vert \theta_t \Vert_2 + (1-\gamma)\lambda_n\sum^n_{t=1} \Vert \theta_t \Vert_1
\end{equation}

where $\Vert \cdot \Vert_2$ is $l_2$ norm, $\Vert \cdot \Vert_1$ is $l_1$ norm, $\lambda_n$ is regularization penalty weight, and $\gamma \in (0,1)$ adjusts the relative weight between inter and intra group sparsity of the solution obtained from the optimization problem.


### References

Change-point Detection Methods for Body-Worn Video <br>
Multiple Change-Points Estimation in Linear Regression Models via Sparse Group Lasso <br>
Towards Experienced Anomaly Detector through Reinforcement Learning <br>
Classification with Costly Features using Deep Reinforcement Learning
