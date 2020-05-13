import pandas as pd
import numpy as np

from High_Frequency.limit_order_book.analysis_unit import *


data = pd.read_pickle("High_Frequency/example_data/海天味业Level-2.pkl")


# min_bar = pd.read_pickle("High_Frequency/example_data/海天味业_min_price.pkl")

wp_2_10 = weighted_price(data, 2, 10)

wp_2_4 = weighted_price(data, 2, 4)

wp_5_10 = weighted_price(data, 5, 10)


mid_price = calc_mid_price(data)

close_price = data['close']

merged_price = pd.concat([close_price.rename("close")\
                             ,mid_price.rename("mid"),wp_2_10.rename("wp_2_10"),wp_2_4.rename("wp_2_4"),wp_5_10.rename("wp_5_10")],axis=1)

merged_price = merged_price.dropna()

# l = merged_price.resample("D").apply(lambda x:len(x))
#     .apply(lambda x:x.index[-1])

merged_price['D'] = merged_price.index.date
merged_price['Time'] = merged_price.index.time

price_type = ['close', 'mid', 'wp_2_10', 'wp_2_4', 'wp_5_10']

tick_r = {c:merged_price.groupby("D")[c].apply(lambda x:x.pct_change()) for c in price_type}

tick_r = pd.DataFrame(tick_r).dropna()


_5T = {c:merged_price.groupby("D")[c].apply(lambda x:x.resample("5T").apply(lambda x:x.iloc[-1]/x.iloc[0]-1 if len(x)>0 else 0 ) ) for c in price_type}
_5T = pd.DataFrame(_5T)


_T = {c:merged_price.groupby("D")[c].apply(lambda x:x.resample("1T").apply(lambda x:x.iloc[-1]/x.iloc[0]-1 if len(x)>0 else 0 )) for c in price_type}

_T = pd.DataFrame(_T)


def resample_rets(merged_price,rule="5T"):
    price_type = ['close', 'mid', 'wp_2_10', 'wp_2_4', 'wp_5_10']
    _price = merged_price.resample(rule).last()
    _r = {c:_price.groupby("D")[c].apply(lambda x:x.pct_change()) for c in price_type}
    _r = pd.DataFrame(_r).dropna()
    return _r


am = [time(10,0),time(11,30)]
pm = [time(13,0),time(15,0)]

"""aics = {}
for _rule in ["1s","5s","10s","30s","60s"]:
    print(_rule)
    _r = resample_rets(merged_price,_rule)

    _r['time'] = _r.index.time
    s = _r
    mask = ((s['time']>=am[0])&(s['time']<=am[-1]))|((s['time']>=pm[0])&(s['time']<=pm[-1]))

    _r = _r[mask]

    mid_price = _r['mid']
    model2 = sm.tsa.AR(mid_price)
    # model2.select_order
    aics[_rule] = model2.fit(maxlag=5).aic
"""

"""
trades_dates = pd.read_pickle("High_Frequency/tradingDates.pkl")
trades_dates = trades_dates[(trades_dates.isOpen == 1)&(trades_dates.exchangeCD == "XSHG")].calendarDate.tolist()

trades_dates = pd.DatetimeIndex(trades_dates)

_r['Date'] = _r.index.date

trades_dates = trades_dates[(trades_dates>=pd.Timestamp(_r['Date'][0]))&(trades_dates<=pd.Timestamp(_r['Date'][-1]))]

np.setdiff1d(trades_dates.astype(str),_r['Date'].unique().astype(str))


"""
['2018-12-18', '2018-12-19', '2018-12-20', '2018-12-21',
       '2019-08-30', '2019-09-02', '2019-09-03', '2019-09-09',
       '2019-09-10', '2019-09-11', '2019-09-12', '2019-09-26',
       '2019-10-25'],
"""


"""


# AR(1)
length_imb = length_imbalance(data, 10)
height_imb = height_imbalance(data,10)
press = price_weighted_pressure(data,kws={})
relative_spread = spread(data)
abs_spread = spread(data,"abs")
voi = volume_order_imbalance(data, kws={})

# sns.distplot(press.dropna())

"""plt.subplots(figsize=(6,3))
sns.distplot(length_imb.dropna()[1])
plt.xlabel("")"""

rule = "5T"
_r = resample_rets(merged_price,rule)
mid_r = _r['mid'].to_frame()
mid_r['time'] = mid_r.index.time

s = mid_r
mask = ((s['time']>=am[0])&(s['time']<=am[-1]))|((s['time']>=pm[0])&(s['time']<=pm[-1]))
mid_r = mid_r[mask]['mid']

y = mid_r

n = 5
_ = {"Lag_%s"%i:_r['mid'].shift(i) for i in np.arange(1,n+1)}
x = pd.DataFrame(_).reindex(y.index)

# return innovation
model = sm.OLS(y,sm.add_constant(x))
res = model.fit()
res.summary()
# pd.Series(res).to_pickle("res1.pkl")


# e = mid_r
e = res.resid


limb_ = length_imb.resample(rule).last().shift().reindex(e.index).rename(columns=lambda x:"QR_%s"%x)
himb_ = height_imb.resample(rule).last().shift().reindex(e.index).rename(columns=lambda x:"HR_%s"%x)
press_ = press.resample(rule).last().shift().reindex(e.index).rename("press",inplace=True)
relative_spread_ = relative_spread.resample(rule).last().shift().reindex(e.index).rename("relative_spread")
abs_spread_ = abs_spread.resample(rule).last().shift().reindex(e.index).rename("abs_spread")
voi_ = voi.resample(rule).last().shift().reindex(e.index).rename("voi")

# limb_ = length_imb.resample(rule).mean().shift().reindex(e.index).rename(columns=lambda x:"QR_%s"%x)
# himb_ = height_imb.resample(rule).mean().shift().reindex(e.index).rename(columns=lambda x:"HR_%s"%x)
# press_ = press.resample(rule).mean().shift().reindex(e.index).rename("press",inplace=True)
# relative_spread_ = relative_spread.resample(rule).mean().shift().reindex(e.index).rename("relative_spread")
# abs_spread_ = abs_spread.resample(rule).mean().shift().reindex(e.index).rename("abs_spread")
# voi_ = voi.resample(rule).mean().shift().reindex(e.index).rename("voi")


x2 = pd.concat([limb_,himb_,press_,abs_spread_,relative_spread_,voi_],axis=1)

# x2_names = ['HR_3','HR_2']

# x2_names = x2.columns

x2_names = x2.columns.drop(["abs_spread","press"])
#
# x2_names = x2.columns.drop(['abs_spread','press','QR_10',\
#                             "QR_9","QR_8","QR_7","QR_6","QR_5",\
#                             "HR_10","HR_9","HR_8","HR_7","HR_6"
#                             ])
y = e

# y = _r['mid'].reindex(e.index)

model2 = sm.OLS(y*100,sm.add_constant(x2[x2_names]),missing="drop")
res2 = model2.fit()
res2.summary()


pd.Series(res2).to_pickle("e_5min_impact_reg.pkl")

sns.heatmap(x2.corr())

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
test_data = himb_.dropna()
pca.fit(test_data)
X_pca = pca.transform(test_data)
pca.explained_variance_ratio_

plt.matshow(pca.components_)



# 采样频率不同结果截然不同，因此对于高频交易来说，设定最长持有时间是有必要的，采用时间止损。


# 价格波动、流动性、历史收益

close = data['close']
last_vol = close.resample(rule).std().shift().reindex(e.index).rename("volatility")
last_r = close.resample(rule).apply(lambda x:x.iloc[-1]/x.iloc[0]-1 if len(x)>0 else 0).shift().reindex(e.index).rename("Lag_rtns")

liquidity = relative_spread.resample(rule).mean().shift().reindex(e.index).rename("liquidity")
# liquidity = abs_spread.resample(rule).mean().shift().reindex(e.index).rename("liquidity")

imbal_vol = voi.resample(rule).last().shift().reindex(e.index).rename("voi")
imbal_press = press.resample(rule).last().reindex(e.index).rename("press")
y = imbal_vol

x_reg = pd.concat([last_r,last_vol,liquidity],axis=1)
model_4 = sm.OLS(abs(y),sm.add_constant(x_reg),missing="drop")
res4 = model_4.fit()
pd.Series(res4).to_pickle("reg_absvoi2.pkl")

res4.summary()


from datetime import date,datetime

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import *
# press_ = press.resample("1T").mean()
# press_ = press_.reindex(e.index)
test_data = press
_ = test_data[test_data.index.date == date(2019,1,2)]

_minute = _.resample("1T").last()
acf_res = sm.tsa.stattools.acf(_.dropna()[:])

fig,ax = plt.subplots(2,1,figsize=(10,4))
_.reset_index(drop=True).plot(ax=ax[0])
ax[0].set_title("Press")
_minute = _.resample("1T").last()
# _minute2 = _.resample("10T").last()
plot_acf(_minute.dropna()[:],use_vlines=True,lags=30,ax=ax[1])
# plot_acf(_minute2.dropna()[:],use_vlines=True,lags=30,ax=ax[2])
plt.tight_layout()


fig,ax = plt.subplots(figsize=(6,3))
sns.distplot(_,ax=ax)
plt.xlabel("")


# plot

# sns.kdeplot(_.press,shade=True)
# sns.rugplot(_.press)


# _ = plt.ylim()
# plt.vlines(0,0,_[1])


# sns.distplot(_.voi)
"""

(a, bins=None, hist=True, kde=True, rug=False, fit=None,
             hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None,
             color=None, vertical=False, norm_hist=False, axlabel=None,
             label=None, ax=None)
"""
import pyecharts.options as opts
from pyecharts.charts import Tab,Surface3D,HeatMap,Timeline


def heat_plot(data):
    n, m = data.shape
    x, y = np.arange(0, n, 1), np.arange(0, m, 1)
    x, y = np.meshgrid(x, y, indexing='ij')
    z = data.values.flatten()
    x = m-1-x
    _data = list(zip(y.flatten(), x.flatten(), z))
    _data = [[int(d[0]), int(d[1]), np.round(d[2], 4)] for d in _data]
    h = (HeatMap(init_opts=opts.InitOpts(width="1000px", height="300px"))
        .add_xaxis(
        xaxis_data=data.columns.astype(str).tolist())
        .add_yaxis(
        series_name="",
        yaxis_data=data.index.astype(str).tolist()[::-1],
        value=_data,
        label_opts=opts.LabelOpts(position='inside', is_show=True,font_size=15),

    )
        .set_series_opts()
        .set_global_opts(
        # toolbox_opts=opts.ToolboxOpts(is_show=True),

        legend_opts=opts.LegendOpts(is_show=False),
        tooltip_opts=opts.TooltipOpts(
            is_show=True,
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        ),
        yaxis_opts=opts.AxisOpts(
            type_="category",
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        ),
        visualmap_opts=opts.VisualMapOpts(
            min_=z.min(), max_=z.max(), is_calculable=True, orient="vertical", pos_left="left"
        ),
        toolbox_opts=opts.ToolboxOpts(is_show=True,
                                      feature=opts.ToolBoxFeatureOpts(
                                          save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(
                                              background_color='white',
                                              connected_background_color="white",
                                              pixel_ratio=2,
                                              name="pic",
                                          ),
                                      )
                                      ),
    )
    )
    return h

_ = heat_plot(tick_r.corr())

_.render("11.html")
