# Heatwave-and-sentiment-study
The goal is to build a delay model and explore the changing relationship between heatwave and emotions in various states across the United States. We believe that human emotions have a delay in response to heat.

The heat here is 0 and 1, and the mood changes are not binary values. At the same time, the data set contains various other variables.

We wanted to use the Bayesian model in R, but we used python notebook in colab to complete this operation.

0322_INLA.ipynb==
步骤一：
安装R包INLA及其依赖包。这一步是比较费时间的

步骤二：
使用INLA运行贝叶斯层次模型，这一步会打印模型摘要

会打印类似这种信息
Model hyperparameters:
                                         mean    sd 0.025quant 0.5quant
Precision for the Gaussian observations 32.43 0.184      32.07    32.43
                                        0.975quant  mode
Precision for the Gaussian observations      32.79 32.43

Marginal log-Likelihood:  19891.00 

Marginal log-Likelihood是关键

步骤三：
运行带有滞后HeatCount的模型
    # 在R中创建滞后变量
    ro.r('r_df$HeatCount_lag1 <- c(NA, head(r_df$HeatCount, -1))')
    ro.r('r_df$HeatCount_lag2 <- c(NA, NA, head(r_df$HeatCount, -2))')
    ro.r('r_df$HeatCount_lag3 <- c(NA, NA, NA, head(r_df$HeatCount, -3))')

步骤四：
运行包含不同滞后周期的模型
for lag_weeks in range(1, 5):
    lagged_formula = "SentimentScore ~ HeatCount"
    for i in range(1, lag_weeks + 1):
        lagged_formula += f" + HeatCount_lag{i*7}"  # 每周滞后7天
    model_name = f"滞后{lag_weeks}周模型"
    print(f"运行包含滞后{lag_weeks}周的模型")
    mlik = run_inla_model(lagged_formula, df, model_name)
    mliks.append(mlik)
    models.append(model_name)

绘制边际对数似然值变化趋势

核心说明：
使用R的INLA库通过Python进行贝叶斯统计建模。它涉及加载数据集，设置滞后变量，并使用INLA（集成嵌套拉普拉斯逼近）方法运行模型。代码的关键步骤和原理如下：

1. 引入Python和R的接口：使用`rpy2`库在Python中调用R代码。这使得可以在Python环境中使用R的功能，特别是INLA包，它用于贝叶斯统计分析。

2. 数据转换：使用`pandas2ri`模块将Pandas的DataFrame转换为R的data.frame格式，确保数据可以在R中使用。

3. 加载INLA库：通过调用`ro.r('library(INLA)')`加载R中的INLA库，它用于执行贝叶斯统计分析。

4. 定义模型运行函数：`run_inla_model`函数封装了模型运行的过程。它接受模型公式、数据集和模型名称作为参数，并在R环境中执行统计分析。

5. 处理滞后变量：如果模型公式中包含滞后变量（如`HeatCount_lag1`），则在数据集中创建这些变量。滞后变量用于分析时间序列数据中某个变量的过去值对当前值的影响。

6. 运行INLA模型：通过构造一个INLA函数调用字符串并在R环境中执行它，来运行贝叶斯回归模型。这个过程包括指定模型公式、数据集和家族分布（这里使用的是高斯分布）。

7. 模型评估：通过打印模型的总结统计信息和计算边际对数似然（Marginal Log-Likelihood, MLL），来评估模型。MLL是衡量模型拟合数据好坏的一个指标，它考虑了模型复杂性和对数据的拟合程度。

比较边际对数似然的原因：
- 模型选择：边际对数似然是比较不同模型拟合数据好坏的一个重要指标。高的边际对数似然表示模型对数据有更好的拟合。
- 复杂性惩罚：边际对数似然自然地平衡了模型的复杂性和拟合度。它惩罚过于复杂的模型，避免过拟合现象。

边际对数似然说明的问题：
- 模型适应性：边际对数似然可以帮助判断模型是否适合数据。一个高的值意味着模型能够较好地解释观察到的数据。
- 模型比较：通过比较不同模型的边际对数似然，可以选择最佳模型。这在存在多个候选模型时尤其有用，帮助识别哪个模型更好地捕捉数据中的模式和结构。

如果这个模型的边际对数似然较高，这确实表明模型有效地捕捉了数据中的模式。它表明的是，包括 HeatCount 及其滞后项在内的整个模型更好地描述了 SentimentScore 的变化。

步骤五
将全国的数据集按照州为单位进行划分，然后对每个州进行类似操作，去研究各个州之间的异同。

步骤六
在得到结果之后，我们需要考虑了空间上是否相邻，分析spatial autocorrelation
