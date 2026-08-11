# Measured results

## RTH  (43 sessions, 16,770 bars, 95 exhaustion fires = 2.2/session)

### Headline at a realistic 5 bps slippage
```
--- rth stop 2% / target 2% / 5bps ---
trades           42
total P&L        $-2,337  (-2.34%)
win rate         59.5%
mean trade       +0.011%   95% CI [-0.367%, +0.374%]
expectancy       $-55.65/trade
profit factor    0.77
Sharpe (daily)   -1.03
max drawdown     -4.74%
avg hold         33 min
gap-through fills 0.0%  (stops that fired through the level)
buy & hold TQQQ  -3.50%
```

### Slippage sensitivity
```
 slippage_bps  trades  total_return_pct  mean_trade_bps  ci_lo_bps  ci_hi_bps  profit_factor   sharpe
         0.00      42             -0.70            9.69     -28.54      46.40           0.93    -0.08
         2.00      42             -1.36            6.25     -31.72      42.82           0.86    -0.46
         5.00      42             -2.34            1.10     -36.66      37.43           0.77    -1.03
        10.00      42             -3.98           -7.54     -44.79      28.40           0.63    -1.97
        20.00      42             -7.28          -24.93     -61.22      10.07           0.41    -3.78
        30.00      42            -10.57          -42.29     -77.60      -7.95           0.26    -5.45
```

### vs random entry of identical duration
```
{
  "strategy_mean_bps": 1.1005023934643066,
  "random_mean_bps": -0.030434330590745833,
  "random_ci_bps": [
    -44.037567127010036,
    43.90129872323732
  ],
  "beats_random": false,
  "hold_minutes": 33
}
```

### Split half
```
       half  trades  mean_trade_bps  ci_lo_bps  ci_hi_bps  win_rate  total_pnl
 first_half      21          -32.82     -89.77      22.80      0.57   -5251.93
second_half      21           35.02      -9.46      77.45      0.62    2914.50
```

### Exit reasons
```
       reason  n  mean_ret_pct    total_pnl
         stop  9     -2.049000 -8879.000000
  session_end  2      0.081579   -26.354830
       target  3      1.949000  1949.000000
score_neutral 28      0.460466  4618.933301
```

### By tranche depth
```
     side  tranches  n  win_rate  mean_ret_pct    total_pnl
long_sqqq         1 10  0.600000     -0.074152  -247.174577
long_sqqq         2  3  0.000000     -0.764191 -1528.381058
long_sqqq         3  2  0.000000     -0.423546  -847.091663
long_tqqq         1 20  0.750000      0.289106  1927.372889
long_tqqq         2  6  0.666667      0.101713   406.852879
long_tqqq         3  1  0.000000     -2.049000 -2049.000000
```

## ETH  (43 sessions, 38,796 bars, 258 exhaustion fires = 6.0/session)

### Headline at a realistic 15 bps slippage
```
--- eth stop 2% / target 2% / 15bps ---
trades           117
total P&L        $-10,926  (-10.93%)
win rate         47.0%
mean trade       -0.179%   95% CI [-0.318%, -0.052%]
expectancy       $-93.38/trade
profit factor    0.43
Sharpe (daily)   -6.54
max drawdown     -11.51%
avg hold         41 min
gap-through fills 0.0%  (stops that fired through the level)
buy & hold TQQQ  -0.19%
```

### Slippage sensitivity
```
 slippage_bps  trades  total_return_pct  mean_trade_bps  ci_lo_bps  ci_hi_bps  profit_factor   sharpe
         0.00     117              7.05           13.91       0.08      26.47           1.61     3.76
         2.00     117              4.36            8.50      -5.81      21.47           1.35     2.26
         5.00     117              1.11            2.75     -11.41      15.58           1.08     0.49
        10.00     117             -5.54           -8.41     -22.54       4.52           0.67    -3.40
        20.00     117            -15.58          -27.48     -41.17     -14.95           0.28    -9.32
        30.00     117            -25.51          -46.47     -59.61     -34.34           0.10   -14.42
```

### vs random entry of identical duration
```
{
  "strategy_mean_bps": -17.939812005202846,
  "random_mean_bps": 0.39592222129731713,
  "random_ci_bps": [
    -18.661497437585286,
    20.13089917909531
  ],
  "beats_random": false,
  "hold_minutes": 41
}
```

### Split half
```
       half  trades  mean_trade_bps  ci_lo_bps  ci_hi_bps  win_rate  total_pnl
 first_half      58          -18.04     -38.96       0.89      0.48   -5569.41
second_half      59          -17.84     -36.60      -1.33      0.46   -5356.41
```

### Exit reasons
```
       reason   n  mean_ret_pct     total_pnl
         stop  11     -2.147000 -12166.333333
  session_end   1     -0.403622   -134.540708
score_neutral 105      0.028867   1375.054038
```

### By tranche depth
```
     side  tranches  n  win_rate  mean_ret_pct    total_pnl
long_sqqq         1 39  0.358974     -0.293495 -3815.430601
long_sqqq         2  7  0.142857     -0.187612  -875.524246
long_sqqq         3  6  0.166667     -0.114011  -684.066678
long_tqqq         1 38  0.657895      0.013282   168.232460
long_tqqq         2 20  0.500000     -0.349740 -4663.203279
long_tqqq         3  7  0.571429     -0.150833 -1055.827660
```
