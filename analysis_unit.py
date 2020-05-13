import pandas as pd
import numpy as np
from datetime import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
# from HFT.limit_order_book.plot import *
sns.set_style("whitegrid")


def volume_order_imbalance(data, kws):

    """

    Reference From <Order imbalance Based Strategy in High Frequency Trading>
    :param data:
    :param kws:
    :return:
    """
    drop_first = kws.setdefault("drop_first", True)

    current_bid_price = data['bidP1']

    bid_price_diff = current_bid_price - current_bid_price.shift()

    current_bid_vol = data['bidV1']

    nan_ = current_bid_vol[current_bid_vol == 0].index

    bvol_diff = current_bid_vol - current_bid_vol.shift()

    bid_increment = np.where(bid_price_diff > 0, current_bid_vol,
                             np.where(bid_price_diff < 0, 0, np.where(bid_price_diff == 0, bvol_diff, bid_price_diff)))

    current_ask_price = data['askP1']

    ask_price_diff = current_ask_price - current_ask_price.shift()

    current_ask_vol = data['askV1']

    avol_diff = current_ask_vol - current_ask_vol.shift()

    ask_increment = np.where(ask_price_diff < 0, current_ask_vol,
                             np.where(ask_price_diff > 0, 0, np.where(ask_price_diff == 0, avol_diff, ask_price_diff)))

    _ = pd.Series(bid_increment - ask_increment, index=data.index)

    if drop_first:
        _.loc[_.groupby(_.index.date).apply(lambda x: x.index[0])] = np.nan

    _.loc[nan_] = np.nan

    return _


def price_weighted_pressure(data, kws):
    n1 = kws.setdefault("n1", 1)
    n2 = kws.setdefault("n2", 10)

    bench = kws.setdefault("bench_type","MID")

    _ = np.arange(n1, n2 + 1)

    if bench == "MID":
        bench_prices = calc_mid_price(data)
    elif bench == "SPECIFIC":
        bench_prices = kws.get("bench_price")
    else:
        raise Exception("")

    def unit_calc(bench_price):
        """比结算价高的价单立马成交，权重=0"""

        bid_d = [bench_price / (bench_price - data["bidP%s" % s]) for s in _]
        # bid_d = [_.replace(np.inf,0) for _ in bid_d]
        bid_denominator = sum(bid_d)

        bid_weights = [(d / bid_denominator).replace(np.nan,1) for d in bid_d]

        press_buy = sum([data["bidV%s" % (i + 1)] * w for i, w in enumerate(bid_weights)])

        ask_d = [bench_price / (data['askP%s' % s] - bench_price) for s in _]
        # ask_d = [_.replace(np.inf,0) for _ in ask_d]
        ask_denominator = sum(ask_d)

        ask_weights = [d / ask_denominator for d in ask_d]

        press_sell = sum([data['askV%s' % (i + 1)] * w for i, w in enumerate(ask_weights)])

        return (np.log(press_buy) - np.log(press_sell)).replace([-np.inf, np.inf], np.nan)

    return unit_calc(bench_prices)



"""
2019-12-27 14:56:06         NaN
2019-12-27 14:56:09   -1.419429
2019-12-27 14:56:12   -2.546248
2019-12-27 14:56:15   -2.998164
2019-12-27 14:56:18   -3.316729
2019-12-27 14:56:21         NaN
2019-12-27 14:56:24         NaN
2019-12-27 14:56:27         NaN
2019-12-27 14:56:30         NaN
2019-12-27 14:56:34         NaN
2019-12-27 14:56:37         NaN
"""




def calc_mid_price(data):
    return (data['bidP1'] + data['askP1']) / 2


def get_mid_price_change(data, drop_first=True):
    _ = calc_mid_price(data).pct_change()
    if drop_first:
        _.loc[_.groupby(_.index.date).apply(lambda x: x.index[0])] = np.nan
    return _


def indicator_reg_analysis(data,indicator, k=20, m=5):
    r_mid_price = get_mid_price_change(data)

    def reg(x, y):
        _y = y.shift(-1).rolling(k).mean().shift(-k + 1).dropna()
        indep = sm.add_constant(pd.concat([x.shift(i) for i in range(m + 1)], axis=1).dropna())

        index = _y.index.intersection(indep.index)
        res = sm.OLS(_y.reindex(index), indep.reindex(index), missing="drop").fit()

        return res

    return r_mid_price.groupby(r_mid_price.index.date).apply(lambda y: reg(indicator.loc[y.index], y))


def weighted_price(data, n1, n2):
    _ = np.arange(n1, n2 + 1)
    numerator = sum([data['bidV%s' % i] * data['bidP%s' % i] + data['askV%s' % i] * data['askP%s' % i] for i in _])
    denominator = sum(data['bidV%s' % i] + data['askV%s' % i] for i in _)

    return numerator / denominator


def length_imbalance(data, n):
    _ = np.arange(1, n + 1)

    imb = {s: (data["bidV%s" % s] - data["askV%s" % s]) / (data["bidV%s" % s] + data["askV%s" % s]) for s in _}

    return pd.concat(imb.values(), keys=imb.keys()).unstack().T


def height_imbalance(data, n):
    _ = np.arange(2, n + 1)

    bid_height = [(data['bidP%s' % (i - 1)] - data['bidP%s' % i]) for i in _]
    ask_height = [(data['askP%s' % i] - data['askP%s' % (i - 1)]) for i in _]

    r = {i + 2: (b - ask_height[i]) / (b + ask_height[i]) for i, b in enumerate(bid_height)}

    r = pd.concat(r.values(), keys=r.keys()).unstack().T

    return r


# updates @12-06~12-07
# not normal distribution

# def signal_generator(data, scale=3, rolling_days=60, tymod='WP',rolling_ticks = 200):
#     #
#     # 为了节约时间 不遍历所有的tick数据 而是选择在阈值之上的入场点进行遍历
#     # 寻找下一个出场点，之后再遍历出场点之后的入场点进行遍历
#
#     # rolling_days = 100
#     # tymod = "press"
#
#     indicator = tick_rsa(data, rolling_days, tymod).dropna()
#
#     # if isinstance(scale,tuple):
#     #
#     #     indicator['up'] = indicator['std'] * scale[1] + indicator['avg']
#     #
#     #     indicator['down'] = indicator['std'] * scale[0] + indicator['avg']
#     #
#     #     enter_points = indicator.where(indicator['indicator'] > indicator['down']).where(
#     #         indicator['indicator'] < indicator['up']).dropna().index
#     #
#     # else:
#     indicator['up'] = indicator['std'] * scale + indicator['avg']
#
#     indicator['down'] = indicator['std']*scale*-1+indicator['avg']
#
#     signal_long = indicator.where(indicator['indicator'] > indicator['up']).dropna().index
#     signal_short = indicator.where(indicator['indicator']<indicator['down']).dropna().index
#
#     if isinstance(rolling_ticks,int):
#         _,ma = trend_to_ma(data['close'],rolling_ticks,"over",return_ma=True)
#         signal_long = np.intersect1d(signal_long,_)
#         _,ma = trend_to_ma(data['close'],rolling_ticks,"below",return_ma=True)
#         signal_short = np.intersect1d(signal_short,_)
#
#         return pd.DatetimeIndex(signal_long),pd.DatetimeIndex(signal_short),indicator,ma
#     else:
#         return pd.DatetimeIndex(signal_long),pd.DatetimeIndex(signal_short),indicator,None
#
#     # if not show:
#     #     return b,m,indicator
#     # else:
#     #     f_name = plot_options.setdefault("f_name", "enter_visual.html")
#     #     ds = np.unique(Financial.index.date)
#     #
#     #     periods = plot_options.setdefault("periods", (ds[0], ds[-1]))
#     #
#     #     with_indicator = plot_options.setdefault("with_indicator",True)
#     #
#     #     s = pd.Timestamp(periods[0])
#     #     e = pd.Timestamp(periods[1]).replace(hour=16)
#     #
#     #     signal_overview(b.loc[s:e],m.loc[s:e],f_name,indicator.loc[s:e],with_indicator)
#     #
#     #     return b,m,indicator

def intraday_moving_average(price,windows = 20):
    return price.groupby(price.index.date).apply(lambda x:x.rolling(windows).mean().shift())


def trend_to_ma(price,windows = 20,dir = "over",return_ma=False):
    ma = intraday_moving_average(price,windows)
    if dir == "over":
        _ = price[price>ma].index
    else:
        _ = price[price<ma].index
    return _,ma if return_ma else _


def sampling_data(data,rule):

    sample_data = data.resample(rule).last()

    return format_times(sample_data)


def format_times(data):
    times_am = (time(9, 30), time(11, 30))
    times_pm = (time(13, 0), time(14, 57))
    _ = data.index.time

    return data[((_>times_am[0])&(_<times_am[1]))|((_>times_pm[0])&(_<times_pm[1]))]


def spread(data,type="relative"):
    s = data['askP1'] - data['bidP1']
    return s/(0.5*(data['askP1']+data['bidP1'])) if type == "relative" else s
