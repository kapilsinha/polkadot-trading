from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, NumeralTickFormatter, HoverTool, Legend, Range1d, Step #, LinearAxis
from bokeh.palettes import Turbo256
from bokeh.plotting import figure, show

import numpy as np
import statsmodels.formula.api as smf
# from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

tools = ['pan', 'xwheel_zoom', 'ywheel_zoom', 'box_zoom', 'undo', 'redo', 'reset', 'save']
datetime_formatter = DatetimeTickFormatter(
    months="%m/%d %H:%M",
    days="%m/%d/ %H:%M",
    hours="%m/%d %H:%M",
    minutes="%m/%d %H:%M:%S",
    seconds="%m/%d %H:%M:%S"
)

class BokehHelper:
    def __init__(self, pair_d, token_d):
        self.pair_d = pair_d
        self.token_d = token_d
        
    def plot_combined_token_pairs(self, df, *, plot_normalized, plot_smoothed, plot_detrended):
        def format_token_pair_plot(p):
            p.xaxis[0].formatter = datetime_formatter

            p.add_tools(HoverTool(
                tooltips=[
                    ( 'block_timestamp', '@block_timestamp{%m/%d %H:%M:%S}' ),
                    ( 'block_number', '@block_number' ),
                ], formatters={
                    '@block_timestamp': 'datetime',
                },
                # display a tooltip whenever the cursor is vertically in line with a glyph
                mode='vline'
            ))
        
        def normalize(series):
            return (series - series.mean()) / series.std()
            
        p1 = figure(
            title=f'Token pair rates',
            width=1250,
            height=750,
            x_axis_label='Block timestamp',
            y_axis_label='Rate z-score',
            tools=tools
        )
        p1.y_range = Range1d(-3.5, 3.5)
        format_token_pair_plot(p1)
        legend_it = []

        num_elems = len(df.pair_address.unique()) * sum([plot_normalized, plot_smoothed, plot_detrended])
        colors = [Turbo256[i] for i in np.round(np.linspace(0, len(Turbo256) - 1, num_elems)).astype(int).tolist()]
        i = 0
        for pair_address in df.pair_address.unique():
            dd = df[df.pair_address == pair_address].reset_index(drop=True).dropna(axis=1, how='all').copy()
            
            if plot_normalized:
                dd['normalized_rate'] = normalize(dd['rate'])
                r = p1.line('block_timestamp', 'normalized_rate', source=ColumnDataSource(dd), line_color=colors[i], line_width=2)
                i += 1
                r.visible = False
                legend_it.append((f"{self.pair_d[pair_address]}", [r]))
            
            if plot_smoothed:
                dd['normalized_smoothed_rate'] = normalize(dd['smoothed_rate'])
                s = p1.line('block_timestamp', 'normalized_smoothed_rate', source=ColumnDataSource(dd), line_color=colors[i], line_width=4, line_alpha=0.5)
                i += 1
                s.visible = False
                legend_it.append((f"{self.pair_d[pair_address]} ema", [s]))
                
            if plot_detrended:
                dd['normalized_detrended_rate'] = normalize(dd['detrended_rate'])
                t = p1.line('block_timestamp', 'normalized_detrended_rate', source=ColumnDataSource(dd), line_color=colors[i], line_width=4, line_alpha=0.5)
                i += 1
                t.visible = False
                legend_it.append((f"{self.pair_d[pair_address]} detrended", [t]))

        # Legends do not scale with number of elements, so we just create multiple legends of size 8
        n = 25
        legends = [Legend(items=sublegend_it) for sublegend_it in [legend_it[i:i + n] for i in range(0, len(legend_it), n)]]
        for legend in legends:
            legend.click_policy='hide'
            p1.add_layout(legend, 'right')

        return p1
    
    def plot_token_pair(self, df_token_pair, jump_bps_thresh=50, rate_multiplier=1):
        def format_token_pair_plot(p, y_colname):
            p.xaxis[0].formatter = datetime_formatter

            p.add_tools(HoverTool(
                tooltips=[
                    ( 'timestamp', '@timestamp{%m/%d %H:%M:%S}' ),
                    ( 'block_timestamp', '@block_timestamp{%m/%d %H:%M:%S}' ),
                    ( 'rate',  f'@{y_colname}' + '{0.2f}' ),
                    ( 'block_number', '@block_number' ),
                    ( 'rate_bps_delta', '@rate_bps_delta{0.1f}' ),
                    ( 'revrate_bps_delta', '@revrate_bps_delta{0.1f}' ),
                ], formatters={
                    '@timestamp': 'datetime',
                    '@block_timestamp': 'datetime',
                },
                # display a tooltip whenever the cursor is vertically in line with a glyph
                # mode='vline'
                mode='mouse'
            ))

        dd = df_token_pair
        dd['rate'] *= rate_multiplier
        dd['smoothed_rate'] *= rate_multiplier
        dd['revrate'] /= rate_multiplier
        dd['smoothed_revrate'] /= rate_multiplier
        pair_addresses = list(dd.pair_address.unique())
        assert(len(pair_addresses) == 1)
        pair_address = pair_addresses[0]

        jumps = dd[(dd.rate_bps_delta.abs() >= jump_bps_thresh) | (dd.revrate_bps_delta.abs() >= jump_bps_thresh)]

        p1 = figure(
            title=f'{self.pair_d[pair_address]} forward',
            width=650,
            height=500,
            x_axis_label='Block timestamp',
            y_axis_label='Rate',
            tools=tools
        )
        format_token_pair_plot(p1, 'rate')
        p1.line('block_timestamp', 'rate', source=ColumnDataSource(dd), legend_label="Exchange rate", line_width=2)
        p1.circle('block_timestamp', 'rate', source=ColumnDataSource(jumps), legend_label="Big jumps", fill_color='red', line_color='red', fill_alpha=0.25)
        p1.line('block_timestamp', 'smoothed_rate', source=ColumnDataSource(dd), legend_label="Exp smooth rate", line_color='orange', line_width=2)

        p1.legend.location = 'top_right'
        p1.legend.click_policy='mute'
        
        p2 = figure(
            title=f'{self.pair_d[pair_address]} reverse',
            width=650,
            height=500,
            x_axis_label='Block timestamp',
            y_axis_label='Reverse rate (1 / rate)',
            tools=tools
        )
        format_token_pair_plot(p2, 'revrate')
        p2.line('block_timestamp', 'revrate', source=ColumnDataSource(dd), legend_label="Reverse rate", line_width=2)
        p2.circle('block_timestamp', 'revrate', source=ColumnDataSource(jumps), legend_label="Big jumps", fill_color='red', line_color='red', fill_alpha=0.25)
        p2.line('block_timestamp', 'smoothed_revrate', source=ColumnDataSource(dd), legend_label="Exp smooth reverse rate", line_color='orange', line_width=2)
        
        p2.legend.location = 'top_right'
        p2.legend.click_policy='mute'

        return row(p1, p2)
    
    def plot_combined_tokens(self, df):
        def format_token_plot(p):
            p.xaxis[0].formatter = datetime_formatter

            p.add_tools(HoverTool(
                tooltips=[
                    ( 'block_timestamp', '@block_timestamp{%m/%d %H:%M:%S}' ),
                    ( 'block_number', '@block_number' ),
                ], formatters={
                    '@block_timestamp': 'datetime',
                },
                # display a tooltip whenever the cursor is vertically in line with a glyph
                mode='vline'
            ))
        
        def normalize(series):
            return (series - series.mean()) / series.std()
            
        p1 = figure(
            title=f'Token values',
            width=1250,
            height=750,
            x_axis_label='Block timestamp',
            y_axis_label='Value in DAI, z-score',
            tools=tools
        )
        p1.y_range = Range1d(-3.5, 3.5)
        format_token_plot(p1)
        legend_it = []

        num_elems = len(df.token_address.unique()) * 2
        colors = [Turbo256[i] for i in np.round(np.linspace(0, len(Turbo256) - 1, num_elems)).astype(int).tolist()]
        for i, token_address in enumerate(df.token_address.unique()):
            dd = df[df.token_address == token_address].reset_index(drop=True).dropna(axis=1, how='all')
            dd['normalized_dai-multi_equiv_no_fees'] = normalize(dd['dai-multi_equiv_no_fees'])
            dd['normalized_smoothed_dai-multi_equiv_no_fees'] = normalize(dd['smoothed_dai-multi_equiv_no_fees'])
            
            r = p1.line('block_timestamp', 'normalized_dai-multi_equiv_no_fees', source=ColumnDataSource(dd), line_color=colors[2 * i], line_width=2)
            r.visible = False
            s = p1.line('block_timestamp', 'normalized_smoothed_dai-multi_equiv_no_fees', source=ColumnDataSource(dd), line_color=colors[2 * i + 1], line_width=4, line_alpha=0.5)
            s.visible = False

            legend_it += [(f"{self.token_d[token_address]}", [r]), (f"{self.token_d[token_address]} ema", [s])]

        # Legends do not scale with number of elements, so we just create multiple legends of size 8
        n = 25
        legends = [Legend(items=sublegend_it) for sublegend_it in [legend_it[i:i + n] for i in range(0, len(legend_it), n)]]
        for legend in legends:
            legend.click_policy='hide'
            p1.add_layout(legend, 'right')

        return p1
        
    def plot_token(self, df_token, jump_bps_thresh=50, rate_multiplier=1):
        def format_token_plot(p):
            p.xaxis[0].formatter = datetime_formatter

            p.add_tools(HoverTool(
                tooltips=[
                    ( 'timestamp', '@timestamp{%m/%d %H:%M:%S}' ),
                    ( 'block_timestamp', '@block_timestamp{%m/%d %H:%M:%S}' ),
                    ( 'DAI value',  '@{dai-multi_equiv_no_fees}{0.2f}' ),
                    # ( 'Smoothed DAI value',  '@{smoothed_dai-multi_equiv_no_fees}{0.2f}' ),
                    ( 'block_number', '@block_number' ),
                ], formatters={
                    '@timestamp': 'datetime',
                    '@block_timestamp': 'datetime',
                },
                # display a tooltip whenever the cursor is vertically in line with a glyph
                # mode='vline'
                mode='mouse'
            ))

        dd = df_token
        dd['dai-multi_equiv_no_fees'] *= rate_multiplier
        dd['smoothed_dai-multi_equiv_no_fees'] *= rate_multiplier
        token_addresses = list(dd.token_address.unique())
        assert(len(token_addresses) == 1)
        token_address = token_addresses[0]

        jumps = dd[(dd['dai-multi_equiv_no_fees_bps_delta'].abs() >= jump_bps_thresh)]
        
        p1 = figure(
            title=f'{self.token_d[token_address]} ({token_address})',
            width=750,
            height=600,
            x_axis_label='Block timestamp',
            y_axis_label='Value in DAI',
            tools=tools
        )
        format_token_plot(p1)
        p1.line('block_timestamp', 'dai-multi_equiv_no_fees', source=ColumnDataSource(dd), legend_label="Value in DAI", line_width=2)
        p1.circle('block_timestamp', 'dai-multi_equiv_no_fees', source=ColumnDataSource(jumps), legend_label="Big jumps", fill_color='red', line_color='red', fill_alpha=0.25)
        p1.line('block_timestamp', 'smoothed_dai-multi_equiv_no_fees', source=ColumnDataSource(dd), legend_label="Smoothed value in DAI", line_color='orange', line_width=2)
        
        p1.legend.location = 'top_right'
        p1.legend.click_policy='mute'

        return p1
