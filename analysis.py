"""
Market Efficiency & Size Premium Analysis
Reads raw data from Results.xlsx, reruns the full statistical analysis,
and cross-checks against stored results in part_c_results.xlsx.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

RESULTS_FILE   = '/mnt/user-data/uploads/Results.xlsx'
PART_C_FILE    = '/mnt/user-data/uploads/part_c_results.xlsx'

# ──────────────────────────────────────────────
# 1. STATS ENGINE
# ──────────────────────────────────────────────
def get_stats(series: pd.Series, freq: str = 'monthly') -> pd.Series:
    """
    Compute: Mean, Std, AC(1..5), Ljung-Box Q-stat, Variance Ratios.
    freq: 'monthly' → VR horizons [2,4,8,12]; 'daily' → [2,5,10,20]
    """
    series = series.replace([-99.99, -999], np.nan).dropna()
    if len(series) == 0:
        return pd.Series(dtype=float)

    stats = {}
    stats['Mean'] = series.mean()
    stats['Std']  = series.std(ddof=1)

    # Autocorrelations lags 1–5
    n = len(series)
    ac_sum_sq = 0.0
    for lag in range(1, 6):
        ac = series.autocorr(lag=lag)
        stats[f'AC({lag})'] = ac
        ac_sum_sq += ac ** 2

    # Ljung-Box Q-statistic (5 lags)
    stats['Q-Stat'] = n * ac_sum_sq
    stats['Q-p']    = 1 - chi2.cdf(stats['Q-Stat'], df=5)

    # Variance Ratios using log returns
    log_ret = np.log(1 + series / 100)
    var_1   = log_ret.var(ddof=1)
    qs = [2, 4, 8, 12] if freq == 'monthly' else [2, 5, 10, 20]

    for q in qs:
        ret_q = log_ret.rolling(q).sum()
        var_q = ret_q.var(ddof=q)
        stats[f'VR({q})'] = (var_q / (q * var_1)) if var_1 != 0 else np.nan

    return pd.Series(stats)

# ──────────────────────────────────────────────
# 2. LOAD RAW DATA FROM RESULTS.XLSX
# ──────────────────────────────────────────────
print("Loading raw data from Results.xlsx ...")

xl = pd.ExcelFile(RESULTS_FILE)
print(f"  Sheets found: {xl.sheet_names}")

mkt_m_raw  = pd.read_excel(RESULTS_FILE, sheet_name='Market_Data')
mkt_d_raw  = pd.read_excel(RESULTS_FILE, sheet_name='Market_Data_Daily')
port_m_raw = pd.read_excel(RESULTS_FILE, sheet_name='Portfolio_Data')
port_d_raw = pd.read_excel(RESULTS_FILE, sheet_name='Portfolio_Data_Daily')

# Parse dates
def parse_dates(df, freq):
    fmt = '%Y%m' if freq == 'monthly' else '%Y%m%d'
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'].astype(str).str.split('.').str[0], format=fmt, errors='coerce')
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)
    return df

mkt_m  = parse_dates(mkt_m_raw[['Date','Mkt-RF','RF']],  'monthly')
mkt_d  = parse_dates(mkt_d_raw[['Date','Mkt-RF','RF']],  'daily')
port_m = parse_dates(port_m_raw[['Date','Lo 20','Qnt 2','Qnt 3','Qnt 4','Hi 20']], 'monthly')
port_d = parse_dates(port_d_raw[['Date','Lo 20','Qnt 2','Qnt 3','Qnt 4','Hi 20']], 'daily')

mkt_m['Market'] = mkt_m['Mkt-RF'] + mkt_m['RF']
mkt_d['Market'] = mkt_d['Mkt-RF'] + mkt_d['RF']

PORT_COLS = ['Lo 20', 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20']

print(f"  Monthly market obs : {len(mkt_m)} ({mkt_m.index[0].date()} → {mkt_m.index[-1].date()})")
print(f"  Daily   market obs : {len(mkt_d)} ({mkt_d.index[0].date()} → {mkt_d.index[-1].date()})")

# ──────────────────────────────────────────────
# 3. RUN FULL-SAMPLE ANALYSIS
# ──────────────────────────────────────────────
print("\nRunning full-sample analysis ...")
results_list = []

# Monthly
res = get_stats(mkt_m['Market'], 'monthly'); res.name = 'Market (Monthly)'; results_list.append(res)
for col in PORT_COLS:
    if col in port_m.columns:
        res = get_stats(port_m[col], 'monthly'); res.name = f'{col} (Monthly)'; results_list.append(res)

# Daily
res = get_stats(mkt_d['Market'], 'daily'); res.name = 'Market (Daily)'; results_list.append(res)
for col in PORT_COLS:
    if col in port_d.columns:
        res = get_stats(port_d[col], 'daily'); res.name = f'{col} (Daily)'; results_list.append(res)

full_results = pd.DataFrame(results_list)

# ──────────────────────────────────────────────
# 4. SUBPERIOD ANALYSIS
# ──────────────────────────────────────────────
print("Running subperiod analysis ...")

MONTHLY_BREAK = '1976-04'   # Sub1: Jul 1926–Mar 1976 / Sub2: Apr 1976–Dec 2025
DAILY_BREAK   = '1974-02-21'

def run_subperiod(mkt_series, port_df, freq, sub1_end, sub2_start, label_suffix):
    rows = []
    mkt  = mkt_series
    mkt1 = mkt[mkt.index <= sub1_end];  mkt2 = mkt[mkt.index >= sub2_start]

    for s, tag in [(mkt1, f'Market (Sub1){label_suffix}'),
                   (mkt2, f'Market (Sub2){label_suffix}')]:
        r = get_stats(s, freq); r.name = tag; rows.append(r)

    for col in PORT_COLS:
        if col not in port_df.columns: continue
        p1 = port_df[port_df.index <= sub1_end][col]
        p2 = port_df[port_df.index >= sub2_start][col]
        for s, tag in [(p1, f'{col} (Sub1){label_suffix}'),
                       (p2, f'{col} (Sub2){label_suffix}')]:
            r = get_stats(s, freq); r.name = tag; rows.append(r)
    return rows

sub_rows_m = run_subperiod(mkt_m['Market'], port_m, 'monthly', MONTHLY_BREAK, MONTHLY_BREAK, ' [Monthly]')
sub_rows_d = run_subperiod(mkt_d['Market'], port_d, 'daily',   DAILY_BREAK,   DAILY_BREAK,   ' [Daily]')
sub_results = pd.DataFrame(sub_rows_m + sub_rows_d)

# ──────────────────────────────────────────────
# 5. CROSS-CHECK AGAINST STORED RESULTS
# ──────────────────────────────────────────────
print("\nCross-checking against stored results ...")

stored_ref = pd.read_excel(RESULTS_FILE, sheet_name='Results_Final')
stored_ref = stored_ref.set_index(stored_ref.columns[0])

tol = 1e-3
mismatches = []
for name in full_results.index:
    if name not in stored_ref.index: continue
    for col in ['Mean', 'Std', 'AC(1)', 'Q-Stat', 'Q-p']:
        if col not in full_results.columns or col not in stored_ref.columns: continue
        v_new = full_results.loc[name, col]
        v_old = stored_ref.loc[name, col]
        if pd.notna(v_new) and pd.notna(v_old):
            if abs(v_new - v_old) > tol:
                mismatches.append({'Series': name, 'Stat': col,
                                   'Recomputed': round(v_new, 6),
                                   'Stored':     round(v_old, 6),
                                   'Diff':        round(v_new - v_old, 6)})

if mismatches:
    print(f"  ⚠  {len(mismatches)} mismatch(es) found:")
    for m in mismatches[:10]:
        print(f"     {m['Series']} | {m['Stat']}: recomputed={m['Recomputed']}, stored={m['Stored']}, diff={m['Diff']}")
else:
    print("  ✓  All recomputed statistics match stored results within tolerance.")

# ──────────────────────────────────────────────
# 6. SAVE RESULTS
# ──────────────────────────────────────────────
out_path = '/home/claude/recomputed_results.xlsx'
with pd.ExcelWriter(out_path) as writer:
    full_results.round(6).to_excel(writer, sheet_name='Full_Sample')
    sub_results.round(6).to_excel(writer, sheet_name='Subperiods')
    if mismatches:
        pd.DataFrame(mismatches).to_excel(writer, sheet_name='Mismatches', index=False)

print(f"\nResults saved → {out_path}")
print("\nFull-sample results preview:")
print(full_results[['Mean','Std','AC(1)','Q-Stat','Q-p','VR(2)']].round(4).to_string())
