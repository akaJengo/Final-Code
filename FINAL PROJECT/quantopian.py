from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.data import builtin 
from quantopian.pipeline.filters import QTradableStocksUS
import quantopian.pipeline.factors as Factors
from quantopian.pipeline.data.builtin import USEquityPricing
import pandas as pd
import time
from quantopian.pipeline.factors import AverageDollarVolume
def initialize(context):
    context.leverage = 1
    schedule_function(rebalance, date_rules.month_start(), time_rules.market_close(hours=1))
    schedule_function(record_vars, date_rules.month_start(), time_rules.market_close())
    attach_pipeline(make_pipeline(), 'my_pipeline')
    pipe = Pipeline()
    dollar_volume = AverageDollarVolume(window_length=30)
    pipe.set_screen(QTradableStocksUS())
    pipe.add(dollar_volume,'ADV') 
    attach_pipeline(pipe, 'pipe')
    schedule_function(flush_portfolio, date_rules.every_day(), time_rules.market_close(minutes=10))
    
    
    schedule_function(limit_orders, date_rules.every_day(), time_rules.market_open(minutes=1))
    schedule_function(flush_orders, date_rules.every_day(), time_rules.market_close(minutes = 1))

def make_pipeline():
    base_universe = QTradableStocksUS()
    USEP = builtin.USEquityPricing
    base = USEP.volume.latest
    Base_Lower_Bound = 70
    Base_Upper_Bound = 100
    Filter_1 = morningstar.valuation_ratios.fcf_yield.latest
    Filter_1_Upper_Bound = 100
    Filter_1_Lower_Bound = 80
    Filter_1 = Filter_1.percentile_between(Filter_1_Lower_Bound, Filter_1_Upper_Bound, mask = base_universe)
    Filter_2 = morningstar.valuation.market_cap.latest
    Filter_2_Upper_Bound = 100
    Filter_2_Lower_Bound = 80
    Filter_2 = Filter_2.percentile_between(Filter_2_Lower_Bound, Filter_2_Upper_Bound, mask = base_universe)
    long_mask = Filter_1 & Filter_2
    Filter_3 = morningstar.valuation_ratios.fcf_yield.latest
    Filter_3_Upper_Bound = 5
    Filter_3_Lower_Bound = 0
    Filter_3 = Filter_3.percentile_between(Filter_3_Lower_Bound, Filter_3_Upper_Bound, mask = base_universe)
    Filter_4 = morningstar.valuation.market_cap.latest
    Filter_4_Upper_Bound = 5
    Filter_4_Lower_Bound = 0
    Filter_4 = Filter_4.percentile_between(Filter_4_Lower_Bound, Filter_4_Upper_Bound, mask = base_universe)
    short_mask = Filter_3 & Filter_4
    longs = base.percentile_between(Base_Lower_Bound, Base_Upper_Bound, mask=long_mask)
    shorts = base.percentile_between(Base_Lower_Bound, Base_Upper_Bound, mask=short_mask)
    return Pipeline(
        columns = {
            'SHORTS': shorts,
            'LONGS': longs,
        },
        screen = base_universe
    )
    return Pipeline()
def before_trading_start(context, data):
    context.output = pipeline_output('pipe').nsmallest(20,'ADV')
    context.outputs = pipeline_output('my_pipeline')
    context.longs = context.outputs[context.outputs['LONGS']].index
    context.long_weight = assign_weights_longs(context)
    context.shorts = context.outputs[context.outputs['SHORTS']].index
    context.short_weight = assign_weights_shorts(context)
def assign_weights_longs(context):
    context.weight = len(context.longs)
    if context.weight > 0:
        long_weight = context.leverage / (len(context.longs)*2)
    if context.weight > 0:
        return long_weight
def assign_weights_shorts(context):
    context.weight = len(context.shorts)
    if context.weight > 0:
        short_weight = context.leverage / (len(context.shorts)*2)
    if context.weight > 0:
        return short_weight
def record_vars(context, data):
    longs = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1
    shorts = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount < 0:
            shorts += 1
    record(leverage=context.account.leverage, long_count=longs, short_count=shorts)
def rebalance(context,data):
    for security in context.portfolio.positions:
        if security not in context.longs and data.can_trade(security) and security not in context.shorts:
            order_target_percent(security, 0)
    for security in context.longs:
        if context.long_weight > 0:
            if data.can_trade(security):
                order_target_percent(security, context.long_weight)
    for security in context.shorts:
        if context.short_weight > 0:
            if data.can_trade(security):
                order_target_percent(security, -context.short_weight)
def limit_orders(context,data):
    currprice = data.current(context.output.index,'price')
    nstocks = max(10,len(context.output))
    for stock in currprice.iteritems():
        try:
            order_target_percent(stock[0],1./nstocks,style = LimitOrder(stock[1]-0.01))
            order_target_percent(stock[0],-1./nstocks,style = LimitOrder(stock[1]+0.01))
        except:
            pass
def flush_orders(context,data):
    for stock, orders in get_open_orders().iteritems(): 
        for order in orders:
            if order not in context.longs and order not in context.shorts:
                cancel_order(order)
def flush_portfolio(context,data):
    for stock in context.portfolio.positions:
            if stock not in context.longs and stock not in context.shorts:
                order_target_percent(stock,0)