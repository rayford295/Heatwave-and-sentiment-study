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

如果这个模型的边际对数似然较高，这确实表明模型有效地捕捉了数据中的模式。它表明的是，包括 HeatCount 及其滞后项在内的整个模型更好地描述了 SentimentScore 的变化。根据模型的边际对数似然（Marginal Log-Likelihood, MLL）值的高低来判断延迟模型的最佳选择。边际对数似然值提供了一种衡量模型拟合数据优劣的方法，反映了模型对数据的解释能力。

步骤五
将全国的数据集按照州为单位进行划分，然后对每个州进行类似操作，去研究各个州之间的异同。

步骤六
在得到结果之后，我们需要考虑了空间上是否相邻，分析spatial autocorrelation


信息准则（如AIC、BIC）和边际对数似然（Marginal Log-Likelihood, MLL）是用来评估统计模型好坏的指标，特别是在进行模型选择时。这些指标可以帮助选择最佳的滞后期，即确定在预测模型中应该考虑多远的过去数据。

AIC（赤池信息准则）
- AIC是衡量统计模型相对质量的指标。它基于模型复杂度（参数数量）和模型对数据拟合程度的平衡。
- AIC计算公式为：\(AIC = 2k - 2ln(L)\)，其中\(k\)是模型中参数的数量，\(L\)是模型的最大似然估计。
- 在模型选择时，AIC较低的模型被认为更好，因为它指示模型有更好的拟合度，同时避免了过度拟合。

BIC（贝叶斯信息准则）
- BIC与AIC类似，也是衡量模型质量和复杂度的指标，但在惩罚模型复杂度时更加严格。
- BIC计算公式为：\(BIC = ln(n)k - 2ln(L)\)，其中\(n\)是数据点的数量。
- 同样，在模型选择时，BIC较低的模型被认为更好。

边际对数似然（Marginal Log-Likelihood, MLL）
- MLL是在贝叶斯框架下评估模型拟合度的指标。它考虑了模型的复杂性和对数据的拟合程度。
- 在贝叶斯模型中，MLL高通常意味着模型对数据有更好的解释。
- 在比较不同模型时，MLL较高的模型通常被认为是更好的模型。

运行带有不同滞后期的模型时，可以计算每个模型的AIC、BIC和MLL。选择这些指标最优的模型（即AIC和BIC较低，MLL较高的模型）可以认为是在给定数据上表现最好的模型。这个过程帮助确定在分析中应该使用多长时间的滞后期。

空间依赖性意味着一个区域的观测值可能与其邻近区域的观测值相似。
这通常通过在模型中添加一个空间随机效应项来实现，这个效应项反映了空间位置之间的相关性。
INLA提供了多种空间模型，如Besag模型（用于区域数据）、Stochastic Partial Differential Equation (SPDE) 模型（用于连续空间数据）等。
例如，如果数据是按区域划分的（如州、县），可以使用Besag模型来建模区域之间的空间依赖性。


怎么去判断第几天是延迟的最佳结果
检查延迟效应的后验分布
对于每个延迟周期(如滞后1天、2天等)的模型,INLA会给出对应滞后效应的后验分布估计。查看每个延迟效应的后验均值和95%可信区间/概率区间:

如果区间不覆盖0,说明该滞后项对sentiment score有显著影响
后验均值的大小反映了该滞后效应的强度
可信区间的宽度反映了估计的不确定性
通过对比不同延迟期的后验分布,可以发现哪个延迟效应最为显著和强劲。

计算延迟效应的模型贡献度
INLA可以计算出每个效应(如滞后项)对模型预测的贡献,用来量化其重要性。可以对比不同滞后项对预测的贡献大小,数值越大说明其作用越关键。

比较交叉验证的预测精度
在留出法交叉验证中,分别用包含不同滞后项的模型进行打分预测,计算均方根误差(RMSE)等指标。预测精度越高的模型,所包含的延迟结构越合理。

检查边际对数似然(MLL)的变化
逐步增加延迟周期,观察MLL的变化趋势。MLL值提高最大时对应的延迟期,可能就是最优延迟结构。


要分析各州之间sentiment score和heatwave的关系,几个方面进行思考和建模:

加入随机效应项
在INLA模型中,可以为州这一分组变量加入随机效应项。这种随机效应允许不同州有自己的基线水平,用于捕捉州与州之间的异质性。模型公式可以是:

SentimentScore ~ HeatCount + HeatCount滞后项 + f(State, model="iid")
其中f(State, model="iid")是对州的随机效应建模,假设各州效应独立同分布。通过该项可以估计出每个州自身的效应强度,反映了不同州之间的差异。

加入空间效应及其与时间的交互
借助INLA的SPDE(Stochastic Partial Differential Equation)方法,可以直接对空间结构进行建模。可将之前的模型扩展为:

SentimentScore ~ HeatCount + HeatCount滞后项 + f(State, model="iid") + 
                 f(空间效应, model="spde", group=County) +
                 f(空间效应,时间效应, model="spde")
f(空间效应, model="spde", group=County)用于捕捉相邻县之间的空间相关性
f(空间效应,时间效应, model="spde")则捕捉空间和时间的交互作用,同一地点在不同时间具有自相关性
通过这种方式,我们可以估计sentiment score在空间和时间上的结构化变化模式。

加入协变量及其与州的交互
可以考虑加入一些协变量(如人口、气候等),并与州进行交互作用,看是否存在"协变量*州"的交互效应:

SentimentScore ~ HeatCount*f(State,model="iid") +
                 HeatCount滞后项 + 人口特征*f(State,model="iid") + ...
这意味着不同州对heatcount和人口等协变量的响应可能不同。

引入动态空间效应
除了全局的空间相关结构外,还可以进一步假设空间效应会随时间动态变化,即每个时间点都有一个独立的空间效应面:

SentimentScore ~ HeatCount + HeatCount滞后项 + f(State, model="iid") +
                 f(空间效应, time=时间指示变量, model="spde", group=County)
这允许模型自动学习每个时间点上空间相关模式的差异。

0426meeting
当前情况总结


1.Start from simple linear models 
BASE Model: for the entire US by county; You can try to add in the below spatial/temporal/fixed effects using back/forward entry to find out the best combinations. 
Sentiment ~ Spatial effect (input of state polygon and county polygon) + temporal effect (year, month, week, weekend, and holiday) + fixed effect (vulnerability, events)
这一步主要是确定base model是什么，我们这里最后确定的是时间+空间+Vulnerability index (THEMES)

2.ADVANCED Model with no lag: 
BASE MODEL + heatwave 
BASE MODEL + air pollution 
BASE MODEL + rainfall 
BASE MODEL + heatwave + air pollution
BASE MODEL + heatwave + rainfall
BASE MODEL + air pollution + rainfall
BASE MODEL + heatwave + air pollution + rainfall
第二步的核心就是探索heatwave, air pollution, 和rainfall这些变量在基础模型的情况下产生的影响。对比基础模型，从而说明这些变量是需要考虑的

3.After select the above best model then add DLNM: ADVANCED Model with DLNM
DLNM with lag 1-14 days
这一步就是增加延迟模型，我们认为这个结果可能是具有延迟效应的

4.需要考虑空间自相关，我们这里有country的shapefile文件和每个country的geoID，这种情况下，我可以确定出每个县的相邻关系，我们想看这些因素heatwave, air pollution, 和rainfall是否满足空间自相关



PyMC3和INLA（Integrated Nested Laplace Approximations）是两种不同的贝叶斯统计模型框架

计算方法
INLA：专为估计潜在高斯场模型（通常用于空间和时间数据）的边缘后验分布而设计。INLA使用拉普拉斯近似来近似这些分布，从而避免了传统MCMC的复杂计算。这使得INLA在处理大型层次模型（特别是在具有空间和时间结构的模型）时，能够比传统MCMC方法更快地得到结果。
PyMC3：依赖于MCMC算法（特别是Hamiltonian Monte Carlo和其变种No-U-Turn Sampler）来抽样估计后验分布。这种方法在统计推断上通常更为精确，因为它不依赖于近似方法，但计算上通常比INLA更慢，尤其是在模型较为复杂或数据集较大时。

进行PyMC3时候产生的问题
Theano 库自 2017 年以来就没有更新，且不再维护。随着 Python 和其它依赖库（如 NumPy）的版本更新，Theano 的一些功能已经不再兼容新版本的库。这是导致错误的主要原因之一。
这里可能无法使用PyMC3

INLA：非常适合估计潜在高斯模型，尤其是在具有明显空间或时间相关性的数据上。然而，它的使用范围相对有限，主要针对特定类型的模型。对于适合其方法的模型，通常比基于采样的方法（如MCMC）计算更快，尤其是在高维潜在空间模型中。
PyMC3：提供了更广泛的模型构建和定制选项。使用PyMC3，研究人员可以构建各种贝叶斯模型，包括但不限于层次模型、非参数模型、和复杂的多层模型。PyMC3的灵活性使其能够应用于多种不同的统计分析任务。虽然计算时间更长，但在模型复杂性和样本大小增加时，提供了更细致的后验分布估计。


代码中没有直接使用 iteritems()，但仍遇到了 AttributeError: 'DataFrame' object has no attribute 'iteritems' 这一错误，那么问题可能是由于 rpy2 库在与 Pandas 的交互中引起的。这可能是因为 rpy2 使用了在旧版本 Pandas 中存在的方法，而这些方法在新版本中已经被移除或更改。

The errors regarding using R's INLA library have been resolved and I can run the model successfully. However, when comparing the results of the basic model and the basic model + heatwave or the basic model + precipitation we set previously, the model’s result information criteria (such as AIC, BIC) and marginal log-likelihood (Marginal Log-Likelihood, MLL ) has not changed much. 
Code link：
https://colab.research.google.com/drive/1hR03T-icNB3pHHdSxZTAgq4en5lS8p5Z?usp=sharing

0612
以加州为例，直接看整个加州的各县之间的包含地理属性的INLA模型

现在有加州的数据集叫"/content/TestData_California.csv"，这里有加州的county的geoid，heatwave，降水等数据。还有全美国的shapefile文件，"/content/County.shp"，包含所有县的geoid。这两个数据集的连接之处是geoid，但是需要注意的是/content/TestData_California.csv这个数据集的县的id需要前面的前缀加一个0，才跟shapefile的数据集相吻合。我们需要shapefile知道加州县的临界关系。其次，我们使用INLA库来看SentimentScore和不同变量之间的关系，我们希望考虑进去空间位置关系。


# 更新模型公式以包含空间效应
spatial_model_formula = """
SentimentScore ~ as.factor(CountyName) + as.factor(Year) + as.factor(Month) +
                  Week + Weekend + Holiday + VulnerabilityIndex +
                  f(spatial_effect, model = "besag", graph = lw)
"""

# 定义包含空间效应的模型
models = {
    "Spatial Model + Heatwave": spatial_model_formula + " + HeatCount",
    "Spatial Model + Air Pollution": spatial_model_formula + " + AirPolllution_Interpolate",
    "Spatial Model + Precipitation": spatial_model_formula + " + Precipitation",
    "Spatial Model + All Environmental Factors": spatial_model_formula + " + HeatCount + AirPolllution_Interpolate + Precipitation"
}

使用R语言来完成这个工作

# 加载 Shapefile
shapefile_path = "/content/County.shp"
gdf = gpd.read_file(shapefile_path)

# 加载原始数据
data_path = "/content/TestData_California.csv"
df = pd.read_csv(data_path)

# 将 GeoDataFrame 中的 'GEOID' 列转换为字符串类型
gdf['GEOID'] = gdf['GEOID'].astype(str)

# 只给 DataFrame 中的 GEOID 列前补充前缀 '0'
df['GEOID'] = '0' + df['GEOID'].astype(str)

# 合并数据集
merged_df = gdf.merge(df, left_on='GEOID', right_on='GEOID')

交互各个州的气候图，以及后续分析


在使用inla模型的时候，出现了46id的州，base模型的问题


