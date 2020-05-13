from .analysis_unit import *

from .performance_analysis import *
from .data_util import *
# 为了节约时间 不遍历所有的tick数据 而是选择在阈值之上的入场点进行遍历
# 寻找下一个出场点，之后再遍历出场点之后的入场点进行遍历
# 更改数据接口之后实盘再更改，需要优化配置


target=0.004


def daily_strategy_(signal,price,target,max_holding_period,stop_loss):
    enter_ = signal[signal == 1].index

    tds = price.index

    en = []
    ex = []

    for i,e in enumerate(enter_):

        if (len(ex) == 0 or ex[-1]<e) or len(en) == 0:

            holding_period = tds[tds > e][:max_holding_period]
            if i<len(enter_)-1 and enter_[i+1] in holding_period :
                holding_period = tds[tds > enter_[i+1]][:max_holding_period]

            en.append(e)
            ex.append(holding_period[-1])

    # ln_price = np.log(price)['close']

    real_ex = []

    rs = []

    for i,e in enumerate(en):

        _ex = ex[i]

        # holding_r = ln_price[e:ex[i]] - ln_price[e]
        holding_r = price[e:_ex]/price[e]-1

        _stop = holding_r[holding_r <= stop_loss]

        if len(_stop)>0:
            _r = _stop[0]
            rs.append(_r)
            real_ex.append(_stop.index[0])
            continue

        _ = holding_r[holding_r >= target]

        _r = _[0] if len(_)>0 else price[_ex] / price[e] -1

        rs.append(_r)
        real_ex.append(_.index[0]) if len(_)>0 else real_ex.append(_ex)

    _ = pd.Series(rs,index=pd.MultiIndex.from_frame(pd.DataFrame(list(zip(en, real_ex)))))
    _.index.rename(['en', 'ex'], inplace=True)
    return _


def intraday_strategy(data,raw_enter_points,target_pnl,side="long",exit_time=time(14, 56, 59),stop_threshold=None,waiting_time=26000):

    e_fields = "ask" if side == "long" else "bid"
    ex_fields = "bid" if side == "long" else "ask"

    downward = stop_threshold

    exit_time = min(exit_time,time(14,56,59))

    enter_points = []
    exit_points = []

    # 设定两个约束列表，对于当时成交量为0的时刻延迟交易（寻找下一个入场点）；
    # 对于入场点可以直接过滤，最优买单交易量=0的时点，这一步骤直接在计算indicator直接体现了 在indicator=nan的时刻即是当前交易量=0

    # 一直持续到收益率>0.02才出场，设定一个等待时长，超过此时长将自动离场；
    trades = {}

    trade_vol = data[ex_fields + "V1"]
    avaliable_trades_times = trade_vol[trade_vol>0]

    for e in raw_enter_points:
        # 确保下一次开仓在 清仓之后
        # 每日收盘前清仓
        # 可以加入止盈、止损条件
        print(e)
        _exit_time = e.replace(hour=exit_time.hour, minute=exit_time.minute, second=exit_time.second)

        if (len(exit_points) > 0 and e > exit_points[-1]) or len(exit_points) == 0:

            available_time = avaliable_trades_times.loc[e:_exit_time].index

            if len(available_time) > 0:
                # 下一个可交易tick作为入场点
                e = available_time[1]
                """"""
                # e_price = data['%sP1'%e_fields].loc[e]

                e_price = data['close'].loc[e]

                # 指定收益/时间离场
                #     如果在指定期间内达到了指定收益，则在下一tick（可交易的）成交
                #     如果在指定期间内未达到指定收益，则在指定期间的下一个tick（可交易）成交（日内）
                #     如果该指定期间内未达到指定收益，并且触及盘尾，紧急清仓（日内）或者下一个交易日的可交易tick成交。
                _ex = e + np.timedelta64(waiting_time, "s")
                # 调整回日内交易
                _ex = min(_ex, _exit_time)
                # 如果涨跌停，股票价格=0,pnl=-1

                """"""
                # _chg = data['%sP1' % ex_fields].replace(0,np.nan).loc[e:_ex] / e_price

                _chg = data["close"].replace(0,np.nan).loc[e:_ex] / e_price


                _ex_marks = (_chg - 1) >= target_pnl if side == "long" else (1/_chg - 1)>=target_pnl

                _exs = _ex_marks.where(_ex_marks).dropna().index
                # 与avaliable_time取交集，
                # 如果存在涨停，但是存在对手方委托单，且达到了等待时长也依旧不卖出。
                _exs = available_time.intersection(_exs)

                if len(_exs) > 0:
                    # 当日指定期间内有成交条件，在达到目标收益的下一个tick成交
                    ex = next_available_time(available_time,_exs[0])
                    # 但是存在下一个tick停牌就卖不出的情况，需要等到次个交易日执行
                    label = 1 if ex is not None else -1
                    ex = next_available_time(avaliable_trades_times.index,_exit_time) if ex is None else ex

                else:
                    # 如果期间内不能成交
                    # 日内剩余可交易时间
                    ex = next_available_time(available_time,_ex,include=True)
                    label = 0 if ex is not None else -1
                    # ex = next_available_time(Financial[Financial['%sV1'%ex_fields]>0].index,_exit_time) if ex is None else ex
                    ex = next_available_time(avaliable_trades_times.index,_exit_time) if ex is None else ex

                    # print(_exit_time)

                # dur_chg = data['%sP1' % ex_fields].replace(0, np.nan).loc[e:ex] / e_price
                """"""
                dur_chg = data["close"].replace(0, np.nan).loc[e:ex] / e_price

                dur_r = dur_chg - 1 if side == "long" else 1/dur_chg-1

                if isinstance(downward, np.float):
                    sl = dur_r[dur_r < downward].index
                    if len(sl) > 0:
                        ex = sl[0]

                # 选择时间前的提前离场

                enter_points.append(e)
                exit_points.append(ex)

                # chg = data['%sP1'%ex_fields].loc[ex] / data['%sP1'%e_fields].loc[e]
                """"""
                chg = data['close'].loc[ex]/data['close'].loc[e]

                r = chg-1 if side == "long" else 1/chg-1

                trades[(e, ex)] = r,label

    return pd.DataFrame(trades).rename(index={0: "return", 1: "label"}).T


"""
def intraday_strategy_set_up(symbol,start_date,end_date,scale, rolling_days, tymod,exit_cond,kws={}):
    data = load_daily_LOB(symbol,start_date,end_date)
    return intraday_strategy_v2(data,scale,rolling_days,tymod,exit_cond,kws)


def intraday_strategy_v2(data,scale,tymod,exit_cond,kws={}):

    side = kws.setdefault("side","long")
    rolling_ticks = kws.get("rolling_ticks")
    rolling_days = kws.get("rolling_days")

    e_fields = "ask" if side == "long" else "bid"
    ex_fields = "bid" if side == "long" else "ask"

    stop_threshold = kws.setdefault("stop_threshold",(None,None))

    upward = stop_threshold[0]
    downward = stop_threshold[1]

    waiting_time = kws.setdefault("waiting_time",26000)
    waiting_time = 26000 if not waiting_time else int(waiting_time)

    limit_chg = kws.setdefault("limit_chg",None)

    target_pnl = kws.get("target_pnl")

    if exit_cond == "pnl" and (not isinstance(target_pnl,np.float)):
        raise Exception("please enter an correct target_pnl ")

    if exit_cond == "pnl" and (not isinstance(waiting_time,np.int)):
        raise Exception("please enter an correct waiting time")

    exit_time = kws.setdefault("exit_time", time(14, 56, 59))
    exit_time = min(exit_time,time(14,56,59))

    s_time = kws.setdefault("start_time",time(9,30,0))
    s_time = max(s_time,time(9,30,0))

    # Financial = load_data(symbol, start_date, end_date)
    # signal_generator(Financial, scale=3, rolling_days=60, tymod='WP', with_ma=True, rolling_ticks=200)

    signal_long, signal_short, indicator, ma = signal_generator(data, scale=scale, rolling_days=rolling_days, tymod=tymod,rolling_ticks=rolling_ticks)

    indicator = indicator['indicator'].dropna()

    raw_enter_points = signal_long if side == "long" else signal_short

    raw_enter_points = raw_enter_points[raw_enter_points.time < exit_time]
    raw_enter_points = raw_enter_points[raw_enter_points.time>s_time]

    # 如果接近涨跌停就不交易

    if isinstance(limit_chg,np.float):
        _limit = data[['last_close', 'close']].shift().loc[raw_enter_points]
        raw_enter_points = _limit[np.abs(_limit['close']/_limit['last_close']-1)<limit_chg].index

    enter_points = []
    exit_points = []

    # 设定两个约束列表，对于当时成交量为0的时刻延迟交易（寻找下一个入场点）；
    # 对于入场点可以直接过滤，最优买单交易量=0的时点，这一步骤直接在计算indicator直接体现了 在indicator=nan的时刻即是当前交易量=0

    # 一直持续到收益率>0.02才出场，设定一个等待时长，超过此时长将自动离场；
    trades = {}

    for e in raw_enter_points:
        # 确保下一次开仓在 清仓之后
        # 每日收盘前清仓
        # 可以加入止盈、止损条件
        print(e)
        _exit_time = e.replace(hour=exit_time.hour, minute=exit_time.minute, second=exit_time.second)

        if (len(exit_points) > 0 and e > exit_points[-1]) or len(exit_points) == 0:

            _ = indicator.loc[e:_exit_time]
            available_time = _.index

            if len(available_time) > 0:
                # 下一个可交易tick作为入场点
                e = available_time[1]

                e_price = data['%sP1'%e_fields].loc[e]

                # 这个是恢复零点之后的第一个时刻
                if exit_cond == "ind":

                    cross_zeros = _[_<=0] if side == "long" else _[_>=0]

                    ex = cross_zeros[0] if len(cross_zeros)>0 else next_available_time(available_time,_exit_time,include=True)

                    ex = next_available_time(data[data['%sV1'%ex_fields] > 0].index, _exit_time) if ex is None else ex

                else:
                    assert exit_cond == "pnl"
                    # 指定收益/时间离场
                    #     如果在指定期间内达到了指定收益，则在下一tick（可交易的）成交
                    #     如果在指定期间内未达到指定收益，则在指定期间的下一个tick（可交易）成交（日内）
                    #     如果该指定期间内未达到指定收益，并且触及盘尾，紧急清仓（日内）或者下一个交易日的可交易tick成交。
                    _ex = e + np.timedelta64(waiting_time, "s")
                    # 调整回日内交易
                    _ex = min(_ex, _exit_time)
                    # 如果涨跌停，股票价格=0,pnl=-1
                    _chg = data['%sP1' % ex_fields].replace(0,np.nan).loc[e:_ex] / e_price

                    _ex_marks = (_chg - 1) >= target_pnl if side == "long" else (1/_chg - 1)>=target_pnl

                    _exs = _ex_marks.where(_ex_marks).dropna().index
                    # 与avaliable_time取交集，
                    # 如果存在涨停，但是存在对手方委托单，且达到了等待时长也依旧不卖出。
                    _exs = available_time.intersection(_exs)

                    if len(_exs) > 0:
                        # 当日指定期间内有成交条件，在达到目标收益的下一个tick成交
                        ex = next_available_time(available_time,_exs[0])
                        # 但是存在下一个tick停牌就卖不出的情况，需要等到次个交易日执行
                        label = 1 if ex is not None else -1
                        ex = next_available_time(indicator.index,_exit_time) if ex is None else ex

                    else:
                        # 如果期间内不能成交
                        # 日内剩余可交易时间
                        ex = next_available_time(available_time,_ex,include=True)
                        label = 0 if ex is not None else -1
                        # ex = next_available_time(Financial[Financial['%sV1'%ex_fields]>0].index,_exit_time) if ex is None else ex
                        ex = next_available_time(indicator.index,_exit_time) if ex is None else ex

                        # print(_exit_time)

                dur_chg = data['%sP1' % ex_fields].replace(0, np.nan).loc[e:ex] / e_price

                dur_r = dur_chg - 1 if side == "long" else 1/dur_chg-1

                ex_dwn = ex
                ex_up = ex

                if isinstance(downward, np.float):
                    sl = dur_r[dur_r < downward].index
                    if len(sl) > 0:
                        ex_dwn = sl[0]

                if isinstance(upward, np.float):
                    sw = dur_r[dur_r > upward].index
                    if len(sw) > 0:
                        ex_up = sw[0]

                # 选择时间前的提前离场
                ex = min(ex_dwn, ex_up)

                enter_points.append(e)
                exit_points.append(ex)

                chg = data['%sP1'%ex_fields].loc[ex] / data['%sP1'%e_fields].loc[e]
                r = chg-1 if side == "long" else 1/chg-1

                trades[(e, ex)] = r,label

    return pd.DataFrame(trades).rename(index={0:"return",1:"label"}).T
#
# ret = intraday_strategy_v2(Financial, scale, tymod, exit_cond, kws)
#

"""
