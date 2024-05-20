### 基于level-2限价订单簿和分笔交易数据的研究

更多高频系列研究可见公众号：2AMquant

该项目进展：

1. Start：Sep.2019 
2. Better Visual display by Pyechart：Dec.2019 

## 结果摘选：

### 

在分析订单簿的研究中，主要从两个角度研究市场交易行为：一种是通过价差、深度、宽度、斜率、订单不平衡等限价订单簿的状态指标，其特征变量决定了短期价格走势；另一种是通过订单簿事件研究市场交易行为。关于订单簿特征变量的描述可见文末的附录部分。

<img src="https://github.com/yudai-il/High-Frequency/blob/master/Results/%E5%9B%BE%E7%89%87%201.png"/>

Aspects of market liquidity（Bervas，2006）；图来源：High-Frequency Trading

### Order Imbalance 


订单不平衡指标反映了市场供需关系和价格买卖压力，侧面反映了市场结构、市场偏好和投资者结构等问题。目前关于订单不平衡是构建基于买卖压力失衡策略的重要指标，对其的研究普遍有如下观点：

1.	正负体现了供需关系，指标小于零意味着供给大于需求；

2.	聚变意味着市场供需出现较大缺口，股票的换手率加大；

3.	均值回复，指标的波动性较大，即订单流不平衡的方向改变较快、较频繁，随着统计时间间隔的跨度增大，订单流不平衡的波动性加大；

4.	存在波动性“聚集”现象。

5.	自相关性/长记忆性，“聚集”，相邻的符号相同，造成的原因可能是1.流动性交易者“分拆订单”的交易行为，避免信息释放速度较快和减小冲击成本；2.信息的连续性；反应了订单流不平衡的收敛速度。
<p>

<img src="https://github.com/yudai-il/High-Frequency/blob/master/Results/%E5%9B%BE%E7%89%87%202.png"/>

Limit book distribution and subsequent price moves；图来源：High-Frequency Trading


### 买卖压力指标序列和自相关性


<img src="https://github.com/yudai-il/High-Frequency/blob/master/Results/%E5%9B%BE%E7%89%87%203.png"/>

### 买卖压力（Press）的盘口拓展分析



天风证券在《买卖压力失衡，利用高频数据拓展盘口数据》一文中指出，可使用高频数据加工降频至低频数据，以拓展盘口数据。本文根据该研究报告构建如下指标，当压力指标位于过去20日平均压力指标的1.96倍以上时，认为是利好信号（见下图b），下图c展示了第一档的深度和宽度不平衡移动平均序列，可以看出当价格序列横盘震荡时，深度不平衡指标处于低位。因此在根据买卖压力构建趋势策略时，需要结合其他指标进一步区分真突破与假突破。

<img src = "https://github.com/yudai-il/High-Frequency/blob/master/Results/%E5%9B%BE%E7%89%87%204.png"/>



## 参考文献：
1.	Cao, C., Hansch, O., & Wang, X. (2009). The information content of an open limit‐order book. Journal of Futures Markets, 29(1), 16-41. 

2.	Cont R , Kukanov A , Stoikov S . The Price Impact of Order Book Events[J]. Social Science Electronic Publishing.

3.	Shen D . Order Imbalance Based Strategy in High Frequency Trading[J]. 2015.

4.	Chordia T , Roll R , Subrahmanyam A . Order imbalance, liquidity, and market returns[J]. Journal of Financial Economics, 2002, 65.

5.	Chordia, T. & Subrahmanyam, A. 2004, "Order imbalance and individual stock returns: Theory and evidence", Journal of Financial Economics, vol. 72, no. 3, pp. 485-518.

6.	Chordia T , Subrahmanyam A . Order imbalance and individual stock returns: Theory and evidence[J]. Journal of Financial Economics, 2004, 72.

7.	Ravi R , Sha Y . Autocorrelated Order-Imbalance and Price Momentum in the Stock Market[J]. International Journal of Economics and Finance, 2014, 6(10).

8.	Charles M. C. Lee & Ready, M.J. 1991, "Inferring Trade Direction from Intraday Data", The Journal of Finance, vol. 46, no. 2, pp. 733-746.

9.	Irene Aldridge .High-Frequency Trading

10.	天风证券-买卖压力失衡-利用高频数据拓展盘口数据

11.	海通证券-听海外高频交易专家讲解美国的高频交易

12.	平安证券-从海外经验看中国高频交易的发展








