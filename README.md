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
