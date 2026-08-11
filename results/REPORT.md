# Measured results — 8 years of 1-minute bars

## RTH — 1,984 sessions, 771,536 bars, 2018-08-08 to 2026-08-07
4,422 exhaustion fires (2.2/session)   [score+signals in 50s]

### Headline @ 5 bps
```
--- rth stop 2% / target 2% / 5bps ---
trades           2037
total P&L        $-178,108  (-178.11%)
win rate         56.7%
mean trade       -0.115%   95% CI [-0.160%, -0.070%]
expectancy       $-87.44/trade
profit factor    0.60
Sharpe (daily)   0.31
max drawdown     -180.89%
avg hold         43 min
gap-through fills 0.5%  (stops that fired through the level)
buy & hold TQQQ  +1678.7%
```

### Slippage sensitivity
```
 slippage_bps  trades  total_return_pct  mean_trade_bps  ci_lo_bps  ci_hi_bps  profit_factor    sharpe
         0.00    2036            -84.67           -1.20      -5.64       3.28           0.79     -0.88
         2.00    2036           -123.39           -5.51      -9.94      -1.03           0.71      0.37
         5.00    2037           -178.11          -11.52     -16.01      -7.05           0.60      0.31
        10.00    2037           -269.57          -21.18     -25.64     -16.77           0.45     -0.08
        20.00    2037           -448.99          -40.33     -44.72     -36.00           0.25      0.38
        30.00    2038           -632.47          -60.72     -64.94     -56.51           0.13      0.32
```

### Year by year
```
 year  trades  win_rate  mean_trade_bps  ci_lo_bps  ci_hi_bps  total_pnl
 2018      97      0.49          -27.66     -53.60      -2.14  -12706.03
 2019     261      0.59           -6.27     -16.08       3.09  -17599.44
 2020     252      0.53          -20.26     -34.63      -5.62  -26204.82
 2021     229      0.55          -14.39     -27.24      -2.46  -21528.24
 2022     271      0.58          -14.25     -29.27       0.77  -22801.61
 2023     263      0.54           -8.35     -19.90       2.69  -24098.23
 2024     255      0.61           -8.95     -20.59       2.19  -25578.40
 2025     256      0.59           -4.55     -16.54       6.78  -12203.30
 2026     153      0.56           -8.10     -24.89       8.03  -15388.07
```

### vs random entry
```
{
  "strategy_mean_bps": -11.51959719863112,
  "random_mean_bps": 2.647878632583771,
  "random_ci_bps": [
    -3.332945184727724,
    8.887946806918015
  ],
  "beats_random": false,
  "hold_minutes": 42
}
```

### Walk-forward (grid fit on first half, scored on second)
```
{
  "split_at": "2022-08-25 11:56:00-04:00",
  "best_in_sample": {
    "stop_pct": 0.5,
    "target_pct": 0.5
  },
  "in_sample_trades": 1038,
  "in_sample_pnl": -45732.24230196624,
  "out_of_sample_trades": 1030,
  "out_of_sample_pnl": -39719.135625027106,
  "out_of_sample_mean_bps": -8.99833444504145,
  "out_of_sample_ci_bps": [
    -11.766830492533286,
    -6.216029400289635
  ]
}
```

### Tier / amplifier
```
 min_tier  amplifier_required  trades  total_return_pct  win_rate  mean_trade_bps  ci_lo_bps  ci_hi_bps
        1               False    2037           -178.11      0.57          -11.52     -16.01      -7.05
        1                True     236            -16.57      0.53          -18.79     -31.23      -6.64
        2               False    1277           -129.90      0.53          -19.95     -26.11     -13.87
        2                True     144            -13.39      0.51          -25.31     -42.23      -8.88
        3               False    1019            -84.91      0.51          -25.00     -31.87     -18.06
        3                True      94             -8.14      0.53          -25.97     -47.04      -5.54
```

### Exit reasons
```
       reason    n  mean_ret_pct      total_pnl
         stop  324     -2.049728 -321783.982207
  session_end  111     -0.553025  -36150.797217
       target   44      1.949000   33133.000000
score_neutral 1558      0.260005  146693.645147
```

### By tranche depth
```
     side  tranches   n  win_rate  mean_ret_pct     total_pnl
long_sqqq         1 787  0.608640     -0.016275  -4269.391685
long_sqqq         2 268  0.451493     -0.233885 -41787.526081
long_sqqq         3 114  0.350877     -0.334266 -38106.268773
long_tqqq         1 621  0.650564     -0.026516  -5488.793179
long_tqqq         2 176  0.534091     -0.275044 -32271.855783
long_tqqq         3  71  0.225352     -0.791328 -56184.298776
```

_[rth complete in 423s]_

## ETH — 1,984 sessions, 1,595,533 bars, 2018-08-08 to 2026-08-07
10,021 exhaustion fires (5.1/session)   [score+signals in 115s]

### Headline @ 15 bps
```
--- eth stop 2% / target 2% / 15bps ---
trades           4523
total P&L        $-766,448  (-766.45%)
win rate         34.6%
mean trade       -0.297%   95% CI [-0.321%, -0.274%]
expectancy       $-169.46/trade
profit factor    0.25
Sharpe (daily)   0.06
max drawdown     -769.15%
avg hold         43 min
gap-through fills 0.7%  (stops that fired through the level)
buy & hold TQQQ  +786.5%
```

### Slippage sensitivity
```
 slippage_bps  trades  total_return_pct  mean_trade_bps  ci_lo_bps  ci_hi_bps  profit_factor    sharpe
         0.00    4523           -122.29            0.46      -1.96       2.86           0.82      0.31
         2.00    4523           -208.86           -3.55      -5.97      -1.15           0.71      0.50
         5.00    4523           -338.66           -9.54     -11.92      -7.18           0.56     -0.36
        10.00    4523           -552.04          -19.52     -21.91     -17.15           0.37      0.53
        20.00    4523           -971.87          -39.53     -41.87     -37.19           0.17      0.38
        30.00    4524          -1398.25          -59.82     -62.19     -57.51           0.08      0.42
```

### Year by year
```
 year  trades  win_rate  mean_trade_bps  ci_lo_bps  ci_hi_bps  total_pnl
 2018     184      0.39          -35.03     -50.18     -20.33  -33067.97
 2019     461      0.26          -32.97     -38.87     -27.06  -82141.08
 2020     570      0.35          -35.41     -43.74     -27.36 -103350.30
 2021     525      0.31          -28.72     -34.85     -22.84  -88760.53
 2022     629      0.47          -29.77     -37.98     -21.74 -102253.43
 2023     596      0.33          -28.87     -34.71     -23.06 -110140.37
 2024     595      0.30          -30.23     -35.68     -24.73 -112298.25
 2025     586      0.32          -25.89     -31.80     -19.95  -79094.94
 2026     377      0.41          -22.41     -29.81     -15.21  -55340.68
```

### vs random entry
```
{
  "strategy_mean_bps": -29.72450003867177,
  "random_mean_bps": 1.1413849893786565,
  "random_ci_bps": [
    -1.891324175794642,
    4.333691331068397
  ],
  "beats_random": false,
  "hold_minutes": 43
}
```

### Walk-forward (grid fit on first half, scored on second)
```
{
  "split_at": "2022-12-30 18:08:00-05:00",
  "best_in_sample": {
    "stop_pct": 0.5,
    "target_pct": 0.5
  },
  "in_sample_trades": 2405,
  "in_sample_pnl": -308506.32743128826,
  "out_of_sample_trades": 2184,
  "out_of_sample_pnl": -285871.11697270937,
  "out_of_sample_mean_bps": -31.236990148932804,
  "out_of_sample_ci_bps": [
    -32.87130181758578,
    -29.637302147674582
  ]
}
```

### Tier / amplifier
```
 min_tier  amplifier_required  trades  total_return_pct  win_rate  mean_trade_bps  ci_lo_bps  ci_hi_bps
        1               False    4523           -766.45      0.35          -29.72     -32.10     -27.39
        1                True     612            -62.08      0.33          -27.78     -33.88     -21.67
        2               False    2904           -466.86      0.35          -34.77     -38.00     -31.67
        2                True     393            -39.66      0.35          -29.03     -37.43     -20.68
        3               False    2354           -293.17      0.34          -37.36     -40.96     -33.78
        3                True     264            -28.24      0.34          -32.10     -43.03     -21.46
```

### Exit reasons
```
       reason    n  mean_ret_pct      total_pnl
         stop  497     -2.170612 -558669.410331
score_neutral 3895     -0.076949 -214983.701857
  session_end   88     -0.524229  -24454.112257
       target   43      1.865186   31659.662443
```

### By tranche depth
```
     side  tranches    n  win_rate  mean_ret_pct      total_pnl
long_sqqq         1 1707  0.308143     -0.230428 -131113.621626
long_sqqq         2  601  0.209651     -0.412894 -165432.964368
long_sqqq         3  269  0.144981     -0.550961 -148208.524710
long_tqqq         1 1324  0.503021     -0.206584  -91172.240143
long_tqqq         2  418  0.370813     -0.363900 -101406.738052
long_tqqq         3  204  0.254902     -0.632909 -129113.473104
```

_[eth complete in 1017s]_
