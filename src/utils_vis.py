import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go



def make_result_df(args, phase: str, datatype: str, dataset, y_pred):
    in_cols = args.dataset.args.in_columns
    out_cols = args.dataset.args.out_columns
    time_lag = args.dataset.args.time_lag
    input_days = args.dataset.args.input_days
    result_index = dataset.y_index 
    result_columns = [
        ('mode', 'phase'), ('mode', 'dataset'),
        *[('target', symbol) for symbol in out_cols],
        *[('pred', symbol) for symbol in out_cols],
        *[('input', f'{symbol}({time_lag + i}d before)') for i in range(input_days,0,-1) for symbol in in_cols if symbol not in out_cols],
        *[('input', f'{symbol}({i}d before)') for i in range(input_days,0,-1) for symbol in in_cols if symbol in out_cols],
        ]
    result_columns = pd.MultiIndex.from_tuples(result_columns)

    x = dataset.x.cpu().numpy()
    y_target = dataset.y.cpu().numpy()
    if dataset.scaler_x is not None:
        x = dataset.scaler_x.inverse_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    if dataset.scaler_y is not None:
        y_target = dataset.scaler_y.inverse_transform(y_target)
        y_pred = dataset.scaler_y.inverse_transform(y_pred)
    
    result = pd.DataFrame(columns=result_columns, index=result_index)
    result.index.name = 'date'
    result.loc[:, ('mode', 'phase')] = phase
    result.loc[:, ('mode', 'dataset')] = datatype
    lagged_indices = [i for i in range(len(in_cols)) if in_cols[i] not in out_cols]
    non_lagged_indices = [i for i in range(len(in_cols)) if in_cols[i] in out_cols]
    x = x[:, :, lagged_indices + non_lagged_indices]  # (batch, time, reordered feature)
    x = x.transpose(0, 2, 1)  # (batch, reordered feature, time)
    x = x.reshape(x.shape[0], -1)  # (batch, reordered feature * time)
    result['input'] = x
    result['target'] = y_target
    result['pred'] = y_pred

    return result





def visualize_result(df_raw, show=True):

    ### 데이터 전처리
    df_raw.index = pd.to_datetime(df_raw.index)
    df = pd.DataFrame(index=pd.date_range(df_raw.index[0], df_raw.index[-1], freq='D'),
                      columns=df_raw.columns, dtype=float)
    df.loc[df_raw.index, :] = df_raw
    df['mode'] = df['mode'].bfill()
    df.loc[:,['input', 'target', 'pred']] = df.loc[:,['input', 'target', 'pred']].interpolate()

    # multi-index to single index
    df.columns = [f'{col[1]} ({col[0]})' for col in df.columns]

    # Plot the data
    target_color = px.colors.qualitative.Dark2
    pred_color = px.colors.qualitative.Set2

    out_symbols = [col[:-7] for col in df.columns if '(pred)' in col]
    input_cols = [col for col in df.columns if '(input)' in col]
    in_symbols = set([col.split('(')[0] for col in input_cols if col.split('(')[0] not in out_symbols])
    input_cols = {
        symbol: [col for col in input_cols if col.split('(')[0] == symbol] for symbol in in_symbols
    }

    ## 그래프 레이아웃 설정
    # calcuate subplot position
    rangeslider_thickness, vspace = 0.05, 0.02
    subplot_height = (1.0 - rangeslider_thickness - 0.04 + vspace) / (len(out_symbols) + len(in_symbols)) - vspace
    offset = 1.0
    offsets = []
    for i in range(len(out_symbols)):
        offsets.append([round(offset - subplot_height, 2), round(offset, 2)])
        offset -= subplot_height + vspace
    # offset -= rangeslider_thickness + 0.04
    offset -= 0.04
    for i in range(len(in_symbols)):
        offsets.append([round(offset - subplot_height, 2), round(offset, 2)])
        offset -= subplot_height + vspace
    
    layout = dict(
        hovermode="x", hoversubplots="axis",
        margin=dict(l=20, r=20, t=20, b=20),
        grid=dict(rows=len(out_symbols) + len(in_symbols), columns=1),
        xaxis=dict( 
            type='date', tickformat="%Y-%m-%d", 
            showspikes=True, spikemode='across', spikethickness=1, spikedash='dot',
            range=[df.index[0], df.index[-1]],
            ),
        yaxis=dict( anchor='x',
                    domain=offsets[0]
                    ),
        **{f'yaxis{i+2}': dict(
            anchor=f'x', 
            domain=offsets[i+1]
            ) for i in range(len(out_symbols) + len(in_symbols) - 1)},

    ) 
    
    layout['xaxis']['rangeselector'] = dict( buttons=list([
        dict(count=5, label="5d", step="day", stepmode="backward"),
        dict(count=1, label="1m", step="month", stepmode="backward"),
        dict(count=3, label="3m", step="month", stepmode="backward"),
        dict(count=6, label="6m", step="month", stepmode="backward"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(step="all")])
    )
    layout['xaxis']['rangeslider'] = dict(visible=True, thickness=rangeslider_thickness, bgcolor='floralwhite',
                                           range=[df.index[0], df.index[-1]])

    ### 그래프 데이터 설정
    data = []
    for i, symbol in sorted(enumerate(out_symbols), key=lambda x: x[0], reverse=True):
        for name in ['target', 'pred']:
            data.append(go.Scatter(x=df.index, y=df[f'{symbol} ({name})'], 
                                    xaxis='x', yaxis=f'y{i+1}' if i > 0 else 'y',
                                    hovertemplate='%{y:.2f}',
                                    mode='lines+markers', 
                                    marker=dict(size=4),
                                    legendgroup=f'{symbol} ({name})',
                                    name=f'{symbol} ({name})'))
    for i, symbol in enumerate(in_symbols):
        for col in input_cols[symbol]:
            data.append(go.Scatter(x=df.index, y=df[col], 
                                 xaxis='x', yaxis=f'y{len(out_symbols)+i+1}',
                                 hovertemplate='%{y:.2f}',
                                 mode='lines+markers',
                                 marker=dict(size=3), 
                                 name=col)
            )
    
    # 그래프 생성
    fig = go.Figure(data=data, layout=layout)
    # print(fig.layout)

    ### 그래프 스타일 변경
    for line in fig.data:
        if '(target)' in line.name:
            line.line.dash = 'dot'
            line.line.color = target_color[out_symbols.index(line.name[:-9])]
        elif '(pred)' in line.name:
            line.line.dash = 'solid'
            line.line.color = pred_color[out_symbols.index(line.name[:-7])]
        else:
            line.line.dash = 'solid'
            line.line.color = 'rgba(0,0,0,0.3)'

    # input feature의 경우, 각 symbol의 첫번째와 마지막만 보이도록 설정
    show_legend = lambda t: '(input)' in t.name
    fig.update_traces(visible="legendonly", selector=show_legend)
    for symbol, cols in input_cols.items():
        show_symbol = lambda t: t.name in [cols[0], cols[-1]]
        fig.update_traces(visible=True, selector=show_symbol)


    ### 그래프 출력
    if show:
        fig.show()
        
        # idx = len(out_symbols) if len(out_symbols) > 1 else ''
        # # remove all overplots except
        # js = "const gd = document.getElementById('{plot_id}');"
        # js = f'''
        #     for (let i = 1; i <= {len(out_symbols)+len(in_symbols)}; i++) {{
        #         if (i == 1) i = '';
        #         if (i != {idx}) 
        #         document.querySelectorAll(`g.rangeslider-container > g.rangeslider-rangeplot > g.overplot > g.xy${{i+1}}`)
        #         .forEach((g) => g.remove());
        #     }}
        #     '''
        

        # fig.show(post_script=[js])


    return fig