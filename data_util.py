import pandas as pd
import numpy as np
from datetime import time


status_mapping = {"SZ": {"not_valid": "S0",
                         "call-auction": "O0",
                         "pre-market-call-auction": "B0",
                         "continuous-bidding": "T0",
                         "after-market-call-auction": "C0",
                         "closing": "E0"},
                  "SH": {"not_valid": "S 11",
                         "call-auction": "C111",
                         "continuous-bidding": "T111",
                         "after-market-call-auction": "U111",
                         "closing": "E111"
                         }
                  }


def load_daily_LOB(date, symbol, exchange, start_time, end_time):

    data = pd.read_csv("High_Frequency/L2测试数据/report_data/%s/%s%s.csv" % (date, exchange, symbol))

    data = data.set_index(pd.to_datetime(data['date']))

    data = data[(data.index.time < end_time) & (data.index.time > start_time)]

    data.drop('date', axis=1, inplace=True)

    return data


def load_LOB(symbol, start_dt, end_dt, start_time=time(9, 30, 0), end_time=time(14, 56, 59),
              deal_status="continuous-bidding"):
    dates = trading_dates(start_dt, end_dt)

    exchange = _match(symbol)

    flags = status_mapping.get(exchange).get(deal_status)

    #   数据统一
    if deal_status == "continuous-bidding":
        start_time = max(time(9, 30, 0), start_time)

    data = pd.concat([load_daily_LOB(d, symbol, exchange, start_time, end_time) for d in dates])

    # return raw_data_process(Financial[Financial['deal_status'] == flags])
    return data[data['deal_status'].str.strip() == flags]


def _match(symbol):
    return "SH" if symbol.startswith("60") else "SZ"


def generate_fields(start_level=1, end_level=10):
    _ = np.array([["askP%s" % i, "bidP%s" % i, "askV%s" % i, "bidV%s" % i] for i in
                  range(start_level, end_level + 1)]).flatten().tolist()
    return _ + ["date"]


def trading_dates(start, end=None):
    end = pd.Timestamp("2050-01-01") if end is None else end

    trading_dates = pd.read_pickle("High_Frequency/tradingDates.pkl")

    dates = trading_dates.where(trading_dates['isOpen'] == 1) \
        .where(trading_dates['exchangeCD'] == "XSHE") \
        .where(trading_dates['calendarDate'] >= start) \
        .where(trading_dates['calendarDate'] <= end).dropna()['calendarDate'].str.replace("-", "").tolist()

    return dates


def next_trading_dates(date, data=None):
    if data is not None:
        _ds = pd.Series(data.index.date).unique()
    else:
        _ds = pd.Series(pd.DatetimeIndex(trading_dates(date)).date)

    return pd.Timestamp(_ds[_ds > date.date()][0])


def next_available_time(time_lists,t,include=False):
    _ = time_lists[time_lists>t] if not include else time_lists[time_lists>=t]
    return _[0] if len(_)>0 else None




# def transaction_data(symbol,start_dt,end_dt,start_time = (9,30),end_time = time(14,56,59)):
#
#     dates = trading_dates(start_dt,end_dt)
#
#     exchange = _match(symbol)
#
#     if exchange == "SH":
#         return pd.concat([load_daily_transaction_data(d, symbol, exchange, start_time, end_time) for d in dates])
#     else:
#         res = {}
#         for d in dates:
#             trans_data = load_daily_transaction_data(d,symbol,exchange,start_time,end_time)
#
#             trans_data = trans_data[trans_data['deal_type'] == "F"]
#
#             trans_data['deal_price'] = trans_data['deal_price']/10000
#
#             order_data = load_daily_order_data(d,symbol,exchange)
#
#             order_data = order_data.set_index(pd.to_datetime(order_data['wtsj']))
#
#             merged_data = pd.concat([trans_data,order_data.reindex(trans_data.index)[['direction','xxjlh']].replace({2:"S",1:"B"})],axis=1).drop("deal_type",axis=1).rename(columns={"direction":"deal_type"})
#
#             deal_in_time = ((merged_data['xxjlh'] == merged_data['buy_syh']) + (merged_data['xxjlh'] == merged_data['sell_syh']))
#
#             _ = deal_in_time[~deal_in_time]
#
#             to_fix = trans_data.loc[_.index]
#
#             for i in to_fix.iterrows():
#                 t,cont = i
#                 buy_syh,sell_syh,price = cont['buy_syh'],cont['sell_syh'],cont['deal_price']
#                 _order = order_data.loc[:t]
#                 buy_q = _order.where((_order['xxjlh'] == buy_syh) &(_order['wtjg'] == price)).dropna()['wtsl'].sum()
#                 sell_q = _order.where((_order['xxjlh'] == sell_syh) &(_order['wtjg'] == price)).dropna()['wtsl'].sum()
#
#                 fix_sides = None
#                 if buy_q == sell_q == 0:
#                     fix_sides = np.nan
#                 elif buy_q == 0:
#                     fix_sides = "B"
#                 elif sell_q == 0:
#                     fix_sides = "S"
#                 else:
#                     raise Exception("")
#                 merged_data.loc[t]['deal_type'] = fix_sides
#
#             res[d] = merged_data
#
#         pd.concat(res.values(),keys=res.keys())
#
#
#             # order_type = order_data['direction'].replace({2:"sell_syh",1:"buy_syh"})
#             # pd.concat([order_type,order_data[['xxjlh']]],axis=1).pivot(columns='direction',values='xxjlh')
#
#
# def load_daily_transaction_data(date, symbol, exchange, start_time, end_time):
#
#     trans_data = pd.read_csv("E:/python-projects/HFT/limit_order_book/Financial/trans_data/%s/%s%s.csv" % (date, exchange, symbol))
#
#     trans_data = trans_data.set_index(pd.to_datetime(trans_data['deal_time']))
#
#     trans_data = trans_data[(trans_data.index.time < end_time) & (trans_data.index.time > start_time)]
#
#     trans_data.drop('deal_time', axis=1, inplace=True)
#
#     return trans_data
#

# def load_daily_order_data(date,symbol,exchange):
#     order_data = pd.read_csv("E:/python-projects/HFT/limit_order_book/Financial/order_data/%s/%s%s.csv" % (date, exchange, symbol))
#
#     return order_data