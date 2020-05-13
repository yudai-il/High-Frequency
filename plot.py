import pyecharts.options as opts
from pyecharts.charts import Line,Scatter,Grid
from pyecharts.render import make_snapshot
import numpy as np
from .performance_analysis import *

import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import matplotlib as mpl

colors = [mpl.colors.rgb2hex(i) for i in sns.color_palette()]

import matplotlib.pyplot as plt

from pyecharts.faker import Faker
from pyecharts.charts import Tab,Surface3D,HeatMap,Timeline




# supplements = {"指标":ind,
#                "移动平均":{"ma":input_ma,"price":input_price},
#                "marks":{"in":[],"out":[]}
#               }


def signal_overview(price, trades=None, supplements=None, plot_options={}, f_name="1213.html"):
    figs_counts = len(supplements)
    legends_heights = (figs_counts * 5)
    left = 90 - legends_heights
    height = left / figs_counts

    sub_plots = []

    fig_size = plot_options.setdefault("fig_size", ("1000px", "600px"))

    x_data = price.index.tolist()
    y_data = price.values.tolist()

    marks = supplements.get("marks")

    with_marks = True if marks is not None else False

    if with_marks:
        _ = pd.Series(index=price.index, data=np.arange(len(price)))

        marks_data = [opts.MarkPointItem(coord=[(x_data[int(l)]), (y_data[int(l)])],
                                         itemstyle_opts=opts.ItemStyleOpts(
                                             color=plot_options.setdefault(v[0], colors[i]), opacity=0.6),
                                         symbol_size=20, symbol="pin") for i, v in enumerate(marks.items()) for l in
                      _.reindex(v[1]).dropna().values.tolist()]
    else:
        marks_data = []
    with_trades = True if isinstance(trades, pd.Series) else False

    if with_trades:
        e_loc = [price.index.get_loc(i) for i in trades.index.get_level_values(0).unique()]
        ex_loc = [price.index.get_loc(i) for i in trades.index.get_level_values(1).unique()]
        pairs_loc = list(zip(e_loc, ex_loc))
        (pairs_loc)
        _b = [{"gt": p[0], "lte": p[1], "color": "#CD6839"} for p in pairs_loc]
        _c = [{"gt": pairs_loc[i][-1], "lte": p[0], "color": "#4A708B"} for i, p in enumerate(pairs_loc[1:])]
        _a = [{"lte": pairs_loc[0][0], "color": "#4A708B"}, {"gt": pairs_loc[-1][-1], "color": colors[0]}]
        pieces = _a + _b + _c
    else:
        pieces = []

    ds = np.unique(price.index.date)

    decor_ = price.groupby(price.index.date).apply(lambda x: [x.index[0], x.index[-1]]).loc[
        [l[1] for l in list(filter(lambda x: x[0] % 2 == 0, enumerate(ds)))]]
    data_zoom_axis = np.arange(figs_counts).astype(int).tolist()

    line = (Line(init_opts=opts.InitOpts(width=fig_size[0], height=fig_size[1]))
        .add_xaxis(xaxis_data=x_data)
        .add_yaxis(
        series_name="价格",
        y_axis=y_data,
        yaxis_index=0,
        is_smooth=True,
        is_symbol_show=False,
        is_hover_animation=False,
        linestyle_opts=opts.LineStyleOpts(width=2),
        markpoint_opts=opts.MarkPointOpts(
            data=marks_data,
        ),

    )
        .set_global_opts(
        title_opts=opts.TitleOpts(title="价格", pos_left='center', pos_top=str(figs_counts * 5) + "%"),
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        tooltip_opts=opts.TooltipOpts(
            trigger="axis",
            axis_pointer_type="cross",
            border_width=1,
        ),
        legend_opts=opts.LegendOpts(pos_left="left"),
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True, link=[{"xAxisIndex": "all"}]
        ),
        datazoom_opts=[
            opts.DataZoomOpts(
                is_show=True,
                is_realtime=True,
                xaxis_index=data_zoom_axis,
                pos_top="5%",
                range_start=0, range_end=100, orient="horizontal", type_="inside"),
            opts.DataZoomOpts(type_="slider", xaxis_index=data_zoom_axis, pos_bottom="bottom"
                              ),

        ],
        yaxis_opts=opts.AxisOpts(
            is_scale=True,
            axistick_opts=opts.AxisTickOpts(is_inside=False),
            axispointer_opts=opts.AxisPointerOpts(is_show=True),
            axisline_opts=opts.AxisLineOpts(symbol='none'),
            splitline_opts=opts.SplitLineOpts(is_show=True)
        ),
        xaxis_opts=opts.AxisOpts(
            is_show=False,
        ),
        visualmap_opts=opts.VisualMapOpts(
            is_show=False,
            is_piecewise=with_trades,
            dimension=0,
            pieces=pieces,
        ),
    )
        .set_series_opts(
        markarea_opts=opts.MarkAreaOpts(
            data=
            [opts.MarkAreaItem(name="",
                               x=(d[0], d[1]),
                               itemstyle_opts=opts.ItemStyleOpts(
                                   color="#A1A9AF",
                                   opacity=0.2),
                               ) for d in decor_]
        )
    )

    )

    sub_plots.append(line)

    if with_marks:
        _ = supplements.pop("marks")

    if len(supplements)>0:

        last_pic = list(supplements.keys())[-1]

        for i, s_1 in enumerate(supplements.items()):

            k, v = s_1[0], s_1[1]
            print(k)
            ls = Line(init_opts=opts.InitOpts(width=fig_size[0], height=fig_size[1]))
            ls = ls.add_xaxis(xaxis_data=price.index.tolist())
            for j, s_2 in enumerate(v.items()):
                n, l = s_2[0], s_2[1]
                print(n)
                num = i * len(s_1) + j
                ls.add_yaxis(
                    series_name=n,
                    y_axis=l.tolist(),
                    yaxis_index=len(sub_plots),
                    linestyle_opts=opts.LineStyleOpts(width=2, color=plot_options.setdefault(n, colors[num])),
                    label_opts=opts.LabelOpts(is_show=False),
                )
            s = legends_heights + height * len(sub_plots)
            ls = ls.set_global_opts(
                title_opts=opts.TitleOpts(title=k, pos_left='center', pos_top=str(s) + "%"),
                legend_opts=opts.LegendOpts(pos_left="left", pos_top=str(len(sub_plots) * 5) + "%"),
                yaxis_opts=opts.AxisOpts(
                    is_scale=True,
                    axistick_opts=opts.AxisTickOpts(is_inside=False),
                    axispointer_opts=opts.AxisPointerOpts(is_show=True),
                    axisline_opts=opts.AxisLineOpts(symbol='none'),
                    splitline_opts=opts.SplitLineOpts(is_show=True)
                ),
                xaxis_opts=opts.AxisOpts(
                    is_show=(k == last_pic),
                ),
            ).set_series_opts(
                markarea_opts=opts.MarkAreaOpts(
                    data=[opts.MarkAreaItem(name="", x=(d[0], d[1]),
                                            itemstyle_opts=opts.ItemStyleOpts(color="#A1A9AF", opacity=0.2), ) for d in
                          decor_],
                    is_silent=False,

                )
            )
            sub_plots.append(ls)

    pos = [i + "%" for i in (height * np.arange(len(sub_plots)) + legends_heights + 4.3).astype(str)]

    g = (Grid(init_opts=opts.InitOpts(width=fig_size[0], height=fig_size[1])))
    for i, v in enumerate(sub_plots):
        g.add(chart=v,
              grid_opts=opts.GridOpts(pos_left=50, pos_right=50, pos_top=pos[i], height=str(height - 5) + "%"),
              )
    g.render(f_name)


# make_snapshot(snapshot,c.render(),"pic.png")

"""
{'累计收益': 
0.030   -0.014    0.136710
        -0.013    0.137277
        -0.012    0.093535
        -0.011    0.091448
        -0.010    0.105431
        -0.009    0.074431
        -0.008    0.109376
        -0.007    0.128096
        -0.006    0.146159
        ...}
        
"""


def plot_performance_overview(dict_results,plot_tymod,p):

    if plot_tymod == "pic":

        figs = {}
        for k,v in dict_results.items():
            v = v.unstack()
            series = v.index.get_level_values(0).unique() if isinstance(v.index,pd.MultiIndex) else [k]
            c = 3
            r = np.ceil(len(series) / c)

            fig = plt.figure(figsize=(8,6))
            plt.subplots_adjust(hspace=0.3)

            for i,_s in enumerate(series):

                _data = v.loc[_s] if isinstance(v.index,pd.MultiIndex) else v

                X,Y = np.meshgrid(_data.index,_data.columns,indexing='ij')

                Z = _data.values
                pos = int(r * 100 + c * 10+(i+1))
                ax = fig.add_subplot(pos,projection='3d')

                ax.plot_surface(X, Y, Z, color='b')
                ax.set_title("%s"%_s,fontdict={"size":8})
            figs[k] = fig
            fig.savefig("%s_%s.png"%(p,k))
        return figs
    elif plot_tymod == 'dyn-3d':
        results = {}
        for k, perf in dict_results.items():
            perf = perf.unstack()
            c = (
                Surface3D(init_opts=opts.InitOpts(width="1000px", height="500px"))
                    .set_global_opts(
                    title_opts=opts.TitleOpts(title="绩效-曲面波图（%s）" % k),
                    visualmap_opts=opts.VisualMapOpts(max_=perf.values.max(), min_=perf.values.min(),
                                                      range_color=Faker.visual_color),
                    toolbox_opts=opts.ToolboxOpts(is_show=True),

            )
            )
            series = perf.index.get_level_values(0).unique() if isinstance(perf.index,pd.MultiIndex) else [k]
            for i, _s in enumerate(series):
                _data = perf.loc[_s] if isinstance(perf.index,pd.MultiIndex) else perf
                X, Y = np.meshgrid(_data.index, _data.columns)
                d = list(zip(X.flatten().astype(str), Y.flatten().astype(str), _data.values.T.flatten().astype(np.float)))
                c = c.add(
                    series_name=str(_s),
                    shading="color",
                    data=d,
                    xaxis3d_opts=opts.Axis3DOpts(type_="category"),
                    yaxis3d_opts=opts.Axis3DOpts(type_="category"),
                    grid3d_opts=opts.Grid3DOpts(width=100, height=50, depth=100),
                )
            results[k] = c

        tab = Tab()
        for k,v in results.items():
            tab.add(v,k)
        tab.render("%s.html"%p)
    elif plot_tymod == "dyn-hm":
        tab = Tab()
        for k, v in dict_results.items():
            v = v.unstack()
            _tl = _timeline(v,name=k)
            tab.add(_tl, k)
        tab.render("%s.html"%p)


def _timeline(perf,name = ""):
    tl = Timeline(init_opts=opts.InitOpts(width="1000px", height="400px"))
    series = perf.index.get_level_values(0).unique() if isinstance(perf.index,pd.MultiIndex) else [name]
    for i,_w in enumerate(series):
        _data=perf.loc[_w] if isinstance(perf.index,pd.MultiIndex) else perf
        n,m=_data.shape
        x,y = np.arange(0,n,1),np.arange(0,m,1)
        x,y = np.meshgrid(x,y,indexing='ij')
        z = _data.values.flatten()
        data = list(zip(y.flatten(),x.flatten(),z))
        data = [[int(d[0]), int(d[1]), np.round(d[2],4)] for d in data]
        h = (HeatMap(init_opts=opts.InitOpts(width="1000px", height="300px"))
        .add_xaxis(
            xaxis_data=_data.columns.astype(str).tolist())
        .add_yaxis(
            series_name="",
            yaxis_data=_data.index.astype(str).tolist(),
            value = data,
            label_opts=opts.LabelOpts(position='inside', is_show=True),

        )
        .set_series_opts()
        .set_global_opts(
            toolbox_opts=opts.ToolboxOpts(is_show=True),

            legend_opts=opts.LegendOpts(is_show=False),
            tooltip_opts = opts.TooltipOpts(
                is_show= True,
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
        )
        )
        tl.add(h, "{}".format(_w))
        if not isinstance(perf.index,pd.MultiIndex):
            tl.add_schema(is_timeline_show=False)
    return tl


def plot_daily_rets(dict_daily_perf,fields="累计收益",customized_index = {}):
    daily_rets = dict_daily_perf.get(fields).unstack(-1)

    cumsum_daily_rets = daily_rets.cumsum(axis=1)

    if not customized_index:
        s_rets = cumsum_daily_rets.iloc[:,-1].sort_values()

        customized_index = {"累计最小":s_rets.index[0],"累计最大":s_rets.index[-1]}

    _ = pd.MultiIndex.from_tuples(list(customized_index.values()))

    fig,ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    r = cumsum_daily_rets.loc[_].T
    r.plot(ax=ax)
    plt.legend(customized_index.keys())
    plt.xticks(rotation=45)
    plt.show()
    return r

