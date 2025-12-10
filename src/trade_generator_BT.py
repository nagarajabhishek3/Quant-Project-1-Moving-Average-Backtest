

import sys
import numpy as np


def generate_trades_v1(data,
                       closecol='Close',
                       signalcol='signal',
                       init_capital=100000,
                       fixedcapital=True,
                       slippage_per_trade=0.0,
                       verbose=False):
    """
    Robust trade generator/backtest skeleton.
    - data: pandas DataFrame indexed by date (or numeric index). Must contain closecol and signalcol.
    - closecol: name of price column (default 'Close')
    - signalcol: integer signal column: 1 (long), -1 (short), 0 (flat) or similar
    - init_capital: starting capital (used only if fixedcapital True)
    - fixedcapital: if True, returns final capital = init_capital + sum(trade_pnl)
    - slippage_per_trade: flat slippage applied at entry/exit as absolute price (optional)
    Returns:
      final_capital, trade_pnl_array, mtm_pnl_array
    """
    import numpy as np

    # ----- basic input checks -----
    if closecol not in data.columns:
        raise ValueError(f"close column '{closecol}' not found in data")
    if signalcol not in data.columns:
        raise ValueError(f"signal column '{signalcol}' not found in data")

    n = len(data)
    if n == 0:
        return init_capital, np.array([]), np.array([])

    # Use positional indexing for safety
    close = data[closecol].reset_index(drop=True)
    sig = data[signalcol].reset_index(drop=True)

    # Running state
    pos = 0                  # +1 for long, -1 for short, 0 for flat
    qty = 0.0                # number of units currently held (can be 0)
    entry_price = None       # price at which current position was entered
    trade_pnl_list = []      # final realized PnL for each closed trade
    mtm_pnl_list = []        # period-by-period mark-to-market pnl (same length as data)
    current_capital = float(init_capital)

    # If you want sizing logic, modify the sizing below. For now simple qty = 1 for a position.
    def compute_qty_for_position():
        # Adjust this to your sizing (fixed 1 unit here)
        return 1

    for i in range(n):
        price_i = float(close.iloc[i])
        price_prev = float(close.iloc[i-1]) if i > 0 else price_i
        signal_i = int(sig.iloc[i]) if not np.isnan(sig.iloc[i]) else 0

        # mtm for the bar: unrealized PnL contribution from previous position
        if pos != 0 and entry_price is not None:
            mtm_period_pnl_pos = qty * (price_i - price_prev) * pos  # pos is sign, qty positive
        else:
            mtm_period_pnl_pos = 0.0
        mtm_pnl_list.append(mtm_period_pnl_pos)

        # Entry logic: go long if signal is 1 and currently flat; go short if -1 and flat.
        if pos == 0:
            if signal_i == 1:
                # enter long
                pos = 1
                qty = compute_qty_for_position()
                entry_price = price_i + slippage_per_trade  # account for slippage
                if verbose:
                    print(f"Enter LONG at {entry_price} index {i}")
            elif signal_i == -1:
                # enter short
                pos = -1
                qty = compute_qty_for_position()
                entry_price = price_i - slippage_per_trade
                if verbose:
                    print(f"Enter SHORT at {entry_price} index {i}")

        else:
            # If in a position, check for exit (signal flips to 0 or opposite sign)
            if signal_i == 0 or (signal_i * pos) < 0:
                # close existing position at current price (apply slippage)
                exit_price = price_i - slippage_per_trade if pos == 1 else price_i + slippage_per_trade
                # Realized pnl: (exit - entry) * qty * pos
                realized_pnl = (exit_price - entry_price) * qty * pos
                trade_pnl_list.append(realized_pnl)

                if verbose:
                    print(f"Exit {'LONG' if pos==1 else 'SHORT'} at {exit_price} idx {i} pnl {realized_pnl}")

                # reset pos
                pos = 0
                qty = 0.0
                entry_price = None

                # If the new signal calls for immediate reverse entry (signal_i == opposite),
                # we can optionally open a new position in the same bar.
                if signal_i == 1:
                    pos = 1
                    qty = compute_qty_for_position()
                    entry_price = price_i + slippage_per_trade
                    if verbose:
                        print(f"Re-enter LONG at {entry_price} idx {i}")
                elif signal_i == -1:
                    pos = -1
                    qty = compute_qty_for_position()
                    entry_price = price_i - slippage_per_trade
                    if verbose:
                        print(f"Re-enter SHORT at {entry_price} idx {i}")

        # end loop

    # Convert lists to arrays
    trade_pnl = np.array(trade_pnl_list) if len(trade_pnl_list) > 0 else np.array([])
    mtm_pnl = np.array(mtm_pnl_list) if len(mtm_pnl_list) > 0 else np.array([])

    # Final capital accounting
    if fixedcapital:
        final_capital = float(init_capital + trade_pnl.sum())
    else:
        # if not fixed, treat mtm series as cumulative additions (example)
        final_capital = float(init_capital + trade_pnl.sum() + mtm_pnl.sum())

    return final_capital, trade_pnl, mtm_pnl


  


def generate_trades_v2(data, init_capital, max_capital_deploy, buy_margin, sell_margin, pnl_target, pnl_stoploss, 
                       fixedcapital = False, datecol = 'Date', closecol = 'Close', slippage = 0.002, buy_transcost = 0.0001,
                       sell_transcost = 0.0005):
    
    try:
        
        # no trade till signal change if exiting on TGT or SL (may or may not be part of strategy)
        # todo
        # check the PNL logics
        
        if (init_capital <= 0 or max_capital_deploy <= 0 or buy_margin <= 0 or sell_margin <= 0 or pnl_target <=0 or pnl_stoploss <= 0 ):
            return 0, [], []
        
        # simulate trading
        
        capital = init_capital
        qty = 0
        entryprice = 0.0
        pos = 0         # 0 - hold /no pos , +1: long , -1 : short
        margin_blocked = 0.0
        prev_pos = 0
        exit_reason ='';
        
        trade_pnl = []
        mtm_pnl = [] 
        
        for i in range(len(data)):
            
            if (capital <= 0):
                break
            
            if (pos == 0):
                # if there is no existing open position
                
                # check for short signal
                if (data['signal'][i] == -1.0 ):
                    if not ((prev_pos == -1.0) & (exit_reason == 'SLTGT')):
                        # take a short pos
                        
                        pos = -1
                        
                        entryprice = data[closecol][i] * (1 - slippage - sell_transcost)
                        
                        margin_blocked = capital * max_capital_deploy
                        
                        qty = -(margin_blocked // (entryprice * sell_margin))
                        
                        if (-qty < 1):
                            pos = 0
                            qty = 0
                            # break
                    
                # check for long signal
                elif (data['signal'][i] == 1.0 ):
                    if not ((prev_pos == 1.0) & (exit_reason == 'SLTGT')):
                        # take a long pos
                        
                        pos = 1
                        
                        entryprice = data[closecol][i] * (1 + slippage + buy_transcost)
                        
                        margin_blocked = capital * max_capital_deploy
                        
                        qty = margin_blocked // (entryprice * buy_margin)
                        
                        if (qty < 1):
                            pos = 0
                            qty = 0
                            # break
                    
                # else if there is no signal
                # else:
                    # do nothing
            
            elif (pos != 0 ):
                # if there is an existing position then check for exit conditions
                
                if (pos > 0):
                    mtm_pnl_from_entry = (data[closecol][i] * (1 - slippage - sell_transcost) - entryprice) * pos
                    mtm_pnl_pos_from_entry = (data[closecol][i] * (1 - slippage - sell_transcost) - entryprice) * qty
                else:
                    mtm_pnl_from_entry = (data[closecol][i] * (1 + slippage + buy_transcost) - entryprice) * pos
                    mtm_pnl_pos_from_entry = (data[closecol][i] * (1 + slippage + buy_transcost) - entryprice) * qty
                
                #mtm_ret_from_entry = mtm_pnl_from_entry / margin_blocked
                mtm_ret_from_entry =  mtm_pnl_from_entry / entryprice
                
                mtm_period_pnl_pos = qty * (data[closecol][i] - data[closecol][i-1])
                
                if (mtm_ret_from_entry > pnl_target or mtm_ret_from_entry < -pnl_stoploss):
                    # target is hit or stoploss is hit, exit the position
                    
                    trade_pnl = np.append(trade_pnl, mtm_pnl_pos_from_entry )
                    
                    mtm_pnl = np.append(mtm_pnl, mtm_period_pnl_pos)
                    
                    if not (fixedcapital):
                        capital = capital + mtm_pnl_pos_from_entry
                    
                    # release the margins and all others
                    margin_blocked = 0
                    qty = 0
                    entryprice = 0
                    exit_reason = 'SLTGT'
                    prev_pos = pos
                    pos = 0
                    
                    # todo - dont take pos on tgt/sl exit till signal reversal
                    
                    
                elif (data['signal'][i] != data['signal'][i-1]):
    
                    trade_pnl = np.append(trade_pnl, mtm_pnl_pos_from_entry )
                    
                    mtm_pnl = np.append(mtm_pnl, mtm_period_pnl_pos)
                    
                    if not (fixedcapital):
                        capital = capital + mtm_pnl_pos_from_entry
    
                    # release the margins and all others
                    margin_blocked = 0
                    qty = 0
                    entryprice = 0
                    exit_reason = 'SC'
                    prev_pos = pos
                    pos = 0
                    
                    # enter a reverse trade
                    # check for short signal
                    if ((data['signal'][i] == -1.0) & (prev_pos != -1.0)):
                        # take a short pos
                        
                        pos = -1
                        
                        entryprice = data[closecol][i] * (1 - slippage - sell_transcost)
                        
                        margin_blocked = capital * max_capital_deploy
                        
                        qty = -(margin_blocked // (entryprice * sell_margin))
                        
                        if (-qty < 1):
                            pos = 0
                            qty = 0
                            # break
                        
                    # check for long signal
                    elif ((data['signal'][i] == 1.0) & (prev_pos != 1.0)):
                        # take a long pos
                        
                        pos = 1
                        
                        entryprice = data[closecol][i] * (1 + slippage + buy_transcost)
                        
                        margin_blocked = capital * max_capital_deploy
                        
                        qty = margin_blocked // (entryprice * buy_margin)
                        
                        if (qty < 1):
                            pos = 0
                            qty = 0
                            # break
                        
                    # else if there is no signal
                    # else:
                        # do nothing
                else:
                    # neither target nor stoploss is hit, hold position
                    
                    mtm_pnl = np.append(mtm_pnl, mtm_period_pnl_pos)

        if fixedcapital:
            capital = init_capital + trade_pnl.sum()
    
        return capital, trade_pnl, mtm_pnl

    except Exception as ex:
        print(sys._getframe().f_code.co_name, ex)
        return None, [], []
