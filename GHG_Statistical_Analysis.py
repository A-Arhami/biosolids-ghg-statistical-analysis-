"""Statistical methods for greenhouse gas emissions from wastewater solids thermal processes."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score

DATA_DIR = Path(".")
MHF_FILE = DATA_DIR / "XXXX.xlsx"
FBI_REGRESSION_FILE = DATA_DIR / "XXXX.xlsx"
FBI_PIECEWISE_FILE = DATA_DIR / "XXXX.xlsx"
FBI_CORR_FILE = DATA_DIR / "XXXX.xlsx"
DRYER_FILE = DATA_DIR / "XXXX.xlsx"
SHEET_NAME = "Sheet1"
IQR_MULTIPLIER = 1.5

def clean_numeric_series(series, remove_zero=True, remove_outliers=True,
                         iqr_multiplier=IQR_MULTIPLIER):
    """Screen a single variable using the specified IQR multiplier."""
    values = pd.to_numeric(series, errors='coerce').replace(
        [np.inf, -np.inf], np.nan).dropna()
    if remove_zero:
        values = values[values != 0]
    if remove_outliers and len(values) > 0:
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if pd.notna(iqr) and iqr > 0:
            lower_limit = q1 - iqr_multiplier * iqr
            upper_limit = q3 + iqr_multiplier * iqr
            values = values[values.between(lower_limit, upper_limit, inclusive='both')]
    return values


def pairwise_clean_df(df, x_col, y_col,
                      iqr_multiplier=IQR_MULTIPLIER,
                      remove_x_zero=True, remove_y_zero=True):
    """Screen x, then screen y among x-retained rows, as in FBI/dryer source."""
    if x_col == y_col:
        raise ValueError('Use two distinct variable names for a correlation pair')
    temp = df[[x_col, y_col]].copy()
    for col in (x_col, y_col):
        temp[col] = pd.to_numeric(temp[col], errors='coerce')
    temp = temp.replace([np.inf, -np.inf], np.nan)
    clean_x = clean_numeric_series(temp[x_col], remove_zero=remove_x_zero,
                                   iqr_multiplier=iqr_multiplier)
    temp = temp.loc[temp.index.intersection(clean_x.index)].copy()
    clean_y = clean_numeric_series(temp[y_col], remove_zero=remove_y_zero,
                                   iqr_multiplier=iqr_multiplier)
    temp = temp.loc[temp.index.intersection(clean_y.index)].copy()
    return temp.dropna(subset=[x_col, y_col])


def valid_pair(df, x_col, y_col, remove_x_zero=True, remove_y_zero=True):
    """Select paired numeric observations without additional outlier screening."""
    pair = df[[x_col, y_col]].copy()
    pair = pair.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if remove_x_zero:
        pair.loc[pair[x_col] == 0, x_col] = np.nan
    if remove_y_zero:
        pair.loc[pair[y_col] == 0, y_col] = np.nan
    return pair.dropna()


def correlation_statistics(df, columns, method='pearson', iqr_multiplier=None):
    """Return pairwise coefficients, significance levels, and sample sizes."""
    cols = list(dict.fromkeys(c for c in columns if c in df.columns))
    corr = pd.DataFrame(np.nan, index=cols, columns=cols)
    pvals = corr.copy()
    ns = corr.copy()
    details = []
    for col in cols:
        if iqr_multiplier is None:
            vals = pd.to_numeric(df[col], errors='coerce').replace([0, np.inf, -np.inf], np.nan).dropna()
        else:
            vals = clean_numeric_series(df[col], iqr_multiplier=iqr_multiplier)
        if len(vals) >= 2 and vals.nunique() > 1:
            corr.loc[col, col] = 1.0
            ns.loc[col, col] = len(vals)
    for i, x_col in enumerate(cols):
        for y_col in cols[i+1:]:
            if iqr_multiplier is None:
                paired = valid_pair(df, x_col, y_col)
            else:
                paired = pairwise_clean_df(df, x_col, y_col, iqr_multiplier=iqr_multiplier)
            n = len(paired)
            if n < 3 or paired[x_col].nunique() < 2 or paired[y_col].nunique() < 2:
                continue
            if method == 'pearson':
                coefficient, p = stats.pearsonr(paired[x_col], paired[y_col])
            elif method == 'spearman':
                coefficient, p = stats.spearmanr(paired[x_col], paired[y_col])
            else:
                raise ValueError('method must be pearson or spearman')
            corr.loc[x_col, y_col] = corr.loc[y_col, x_col] = coefficient
            pvals.loc[x_col, y_col] = pvals.loc[y_col, x_col] = p
            ns.loc[x_col, y_col] = ns.loc[y_col, x_col] = n
            details.append({'Variable 1': x_col, 'Variable 2': y_col,
                            'r_or_rho': coefficient, 'p': p, 'n': n})
    return {'r': corr, 'p': pvals, 'n': ns, 'pairs': pd.DataFrame(details)}

def exp_decay_plateau(x, a, b, c):
    return a * np.exp(-b * np.asarray(x)) + c


def piecewise_linear(x, x0, y0, k1, k2):
    return np.where(np.asarray(x) < x0,
                    y0 + k1 * (np.asarray(x) - x0),
                    y0 + k2 * (np.asarray(x) - x0))


def mhf_regression_metrics(y_true, y_pred, n_params, model_name):
    """MHF regression fbi_regression_metrics."""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    if n - n_params - 1 > 0:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_params - 1)
    else:
        adj_r2 = np.nan
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    if n > 0 and sse > 0:
        aic = n * np.log(sse / n) + 2 * n_params
        bic = n * np.log(sse / n) + np.log(n) * n_params
    else:
        aic = np.nan
        bic = np.nan
    return {'Model': model_name, 'R2': r2, 'Adj_R2': adj_r2,
            'RMSE': rmse, 'MAE': mae, 'AIC': aic, 'BIC': bic}


def fbi_regression_metrics(y_true, y_pred, n_params):
    """FBI polynomial and exponential regression fbi_regression_metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_params - 1, 1)
    aic = n * np.log(rss / n) + 2 * n_params if rss > 0 else np.nan
    bic = n * np.log(rss / n) + np.log(n) * n_params if rss > 0 else np.nan
    return {'R2': r2, 'Adj_R2': adj_r2, 'RMSE': rmse,
            'MAE': mae, 'AIC': aic, 'BIC': bic}


def polynomial_equation(coef, intercept, degree, variable='T_K'):
    """Format polynomial coefficients in descending powers of temperature."""
    if degree == 1:
        return f'y = {coef[0]:.12g} * {variable} + {intercept:.12g}'
    if degree == 2:
        return (f'y = {coef[1]:.12g} * {variable}^2 + '
                f'{coef[0]:.12g} * {variable} + {intercept:.12g}')
    if degree == 3:
        return (f'y = {coef[2]:.12g} * {variable}^3 + '
                f'{coef[1]:.12g} * {variable}^2 + '
                f'{coef[0]:.12g} * {variable} + {intercept:.12g}')
    raise ValueError('Only degrees 1, 2, 3 are supported')


def calculate_aic(y_true, y_pred, num_params):
    """Return the residual-sum-of-squares AIC value."""
    residuals = y_true - y_pred
    rss = np.sum(residuals ** 2)
    n = len(y_true)
    return n * np.log(rss / n) + 2 * num_params

def prepare_mhf(filename=MHF_FILE):
    df = pd.read_excel(filename, sheet_name=SHEET_NAME)
    df.columns = df.columns.astype(str).str.strip()
    for col in df.columns:
        if col != 'Time':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    temp_cols_original = [c for c in df.columns if
        'Temp' in c and '(C)' not in c and '(F)' not in c]
    for col in temp_cols_original:
        temp_f = df[col].where(df[col] != 0, np.nan)
        df[f'{col} (C)'] = (temp_f - 32) * 5 / 9
    df = df.rename(columns={c: f'{c} (F)' for c in temp_cols_original})
    primary_cols = [f'Hearth {i} Temp (C)' for i in range(1, 6)
                    if f'Hearth {i} Temp (C)' in df]
    if not primary_cols:
        raise KeyError('No primary-hearth temperature columns found in EDA_MHF.xlsx')
    df['Primary Hearth Avg Temp (C)'] = df[primary_cols].replace(0, np.nan).mean(axis=1)
    return df


def fit_mhf_co_models(df):
    xcol = 'Primary Hearth Avg Temp (C)'
    ycol = 'CO (g/dry mega gram sludge)'
    screened = valid_pair(df, xcol, ycol)
    x = screened[xcol].to_numpy(dtype=float)
    y = screened[ycol].to_numpy(dtype=float)
    if len(x) < 6:
        raise ValueError('Too few valid observations for MHF CO model')
    rows = []
    equations = []
    for degree in (1, 2, 3):
        coef = np.polyfit(x, y, degree)
        pred = np.poly1d(coef)(x)
        rows.append(mhf_regression_metrics(y, pred, n_params=degree+1,
                                            model_name=f'Polynomial degree {degree}'))
        equations.append(f"Polynomial degree {degree}: " +
                         ' + '.join(f'{c:.12g}*x^{degree-i}' for i, c in enumerate(coef)))
    # Exponential model
    b0 = 0.005
    c0 = min(y)
    a0 = (max(y) - min(y)) * np.exp(b0 * x.min())
    try:
        popt_exp, pcov_exp = curve_fit(exp_decay_plateau, x, y,
                                       p0=[a0, b0, c0], maxfev=20000)
        ypred_exp = exp_decay_plateau(x, *popt_exp)
        rows.append(mhf_regression_metrics(y, ypred_exp, n_params=3,
                                            model_name='Exponential decay with plateau'))
        equations.append('Exponential decay with plateau: '
                         f'y = {popt_exp[0]:.12g} * exp(-{popt_exp[1]:.12g}*x) + {popt_exp[2]:.12g}')
    except (RuntimeError, ValueError, OverflowError) as exc:
        equations.append('Exponential fit failed: ' + str(exc))
    return pd.DataFrame(rows), len(x), equations

def prepare_fbi_regression(filename=FBI_REGRESSION_FILE):
    df = pd.read_excel(filename)
    df = df.rename(columns={'FB.avg (K)': 'T_K', '% N sludge to N2O': 'N_to_N2O'})
    for col in ('T_K', 'N_to_N2O'):
        if col not in df.columns:
            raise KeyError(f'Missing {col} in FBI_EDA.xlsx')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[['T_K', 'N_to_N2O']].replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df['T_K'] > 0) & (df['N_to_N2O'] >= 0)].copy()
    return df


def fit_fbi_temperature_models(data, label):
    X = data[['T_K']].values
    T = data['T_K'].values
    y = data['N_to_N2O'].values
    if len(y) < 6:
        raise ValueError('Too few observations for regression')
    rows, equations = [], []
    for deg in (1, 2, 3):
        model = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False),
                              LinearRegression())
        model.fit(X, y)
        yhat = model.predict(X)
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        cv_rmse = np.sqrt(-cross_val_score(
            model, X, y, scoring='neg_mean_squared_error', cv=kf).mean())
        m = fbi_regression_metrics(y, yhat, n_params=deg + 1)
        m.update({'Dataset': label, 'Model': f'Polynomial degree {deg}',
                  'CV_RMSE': cv_rmse, 'n': len(y)})
        rows.append(m)
        coefs = model.named_steps['linearregression'].coef_
        intercept = model.named_steps['linearregression'].intercept_
        equations.append(f'Polynomial degree {deg}: ' +
                         polynomial_equation(coefs, intercept, deg, variable='T_K'))
    try:
        # Exponential model
        p0 = [max(y) - min(y), 0.001, min(y)]
        bounds = ([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
        popt, pcov = curve_fit(exp_decay_plateau, T, y,
                               p0=p0, bounds=bounds, maxfev=20000)
        yhat = exp_decay_plateau(T, *popt)
        m = fbi_regression_metrics(y, yhat, n_params=3)
        m.update({'Dataset': label, 'Model': 'Exponential decay with plateau',
                  'CV_RMSE': np.nan, 'n': len(y),
                  'a': popt[0], 'b': popt[1], 'c_plateau': popt[2]})
        rows.append(m)
        equations.append('Exponential decay with plateau: '
                         f'y = {popt[0]:.12g} * exp(-{popt[1]:.12g}*T_K) + {popt[2]:.12g}')
    except (RuntimeError, ValueError, OverflowError) as exc:
        equations.append('Exponential fit failed: ' + str(exc))
    return pd.DataFrame(rows), equations

def fit_fbi_piecewise_models(filename=FBI_PIECEWISE_FILE):
    df = pd.read_excel(filename)
    xcol, ycol = 'fb.avg (C)', '% N sludge to N2O'
    for col in (xcol, ycol):
        if col not in df:
            raise KeyError(f'Missing {col} in FBI_Model.xlsx')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df[ycol] > 0].copy()  # Use positive nitrogen-conversion observations
    screened = df.sort_values(xcol)
    x = screened[xcol].values + 273.15  # Celsius to kelvin
    y = screened[ycol].values
    if len(x) < 7:
        raise ValueError('Too few observations for piecewise regression')

    # Starting parameters and bounds
    initial_guess_piecewise = [np.median(x), np.median(y), -0.05, 0.00]
    bounds_piecewise = ([x.min(), y.min(), -np.inf, -np.inf],
                        [x.max(), y.max(), np.inf, np.inf])
    popt_piecewise, pcov_piecewise = curve_fit(
        piecewise_linear, x, y, p0=initial_guess_piecewise,
        bounds=bounds_piecewise, maxfev=20000)
    y_pred_piecewise = piecewise_linear(x, *popt_piecewise)

    # Nonnegative coefficients for this exponential comparison
    initial_guess_exp = [max(y) - min(y), 0.005, min(y)]
    bounds_exp = ([0, 0, 0], [np.inf, np.inf, np.inf])
    popt_exp, pcov_exp = curve_fit(
        exp_decay_plateau, x, y, p0=initial_guess_exp,
        bounds=bounds_exp, maxfev=20000)
    y_pred_exp = exp_decay_plateau(x, *popt_exp)

    comparison = pd.DataFrame({
        'Model': ['Piecewise linear', 'Exponential decay with plateau'],
        'n': [len(x), len(x)],
        'R2': [r2_score(y, y_pred_piecewise), r2_score(y, y_pred_exp)],
        'RMSE': [np.sqrt(mean_squared_error(y, y_pred_piecewise)),
                 np.sqrt(mean_squared_error(y, y_pred_exp))],
        'MAE': [mean_absolute_error(y, y_pred_piecewise),
                mean_absolute_error(y, y_pred_exp)],
        'AIC': [calculate_aic(y, y_pred_piecewise, 4),
                calculate_aic(y, y_pred_exp, 3)]
    })
    return comparison, popt_piecewise, popt_exp

def parse_excel_or_text_datetime(series):
    raw = series.copy()
    numeric = pd.to_numeric(raw, errors='coerce')
    parsed = pd.Series(pd.NaT, index=raw.index, dtype='datetime64[ns]')
    excel_mask = numeric.between(20000, 80000)
    if excel_mask.any():
        parsed.loc[excel_mask] = pd.to_datetime(
            numeric.loc[excel_mask], unit='D', origin='1899-12-30', errors='coerce')
    remaining = ~excel_mask
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(raw.loc[remaining], errors='coerce')
    return parsed.dt.round('s')


def prepare_fbi_correlations(filename=FBI_CORR_FILE):
    df = pd.read_excel(filename, sheet_name=SHEET_NAME)
    df.columns = df.columns.astype(str).str.strip()
    date_col = 'date.time' if 'date.time' in df else 'DateTime'
    if date_col not in df:
        raise KeyError('Expected date.time or DateTime in EDA_FBI.xlsx')
    df['DateTime'] = parse_excel_or_text_datetime(df[date_col])
    df = df.dropna(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)
    for c in df.columns:
        if c not in ('DateTime', 'date.time'):
            converted = pd.to_numeric(df[c], errors='coerce')
            if converted.notna().any(): df[c] = converted
    periods = {
        'Period 1': ('2022-10-20 09:00:00', '2022-10-20 12:20:00'),
        'Period 2': ('2022-10-20 16:51:00', '2022-10-20 17:04:00'),
        'Period 3': ('2022-10-21 07:41:00', '2022-10-21 13:30:00'),
    }
    df['Period'] = pd.Series(index=df.index, dtype='object')
    for label, (start, end) in periods.items():
        df.loc[df['DateTime'].between(pd.Timestamp(start), pd.Timestamp(end)), 'Period'] = label
    return df[df['Period'].notna()].copy()


def prepare_dryer_correlations(filename=DRYER_FILE):
    df = pd.read_excel(filename, sheet_name=SHEET_NAME)
    df.columns = df.columns.astype(str).str.strip()
    protected = {'Time', 'DateTime', 'date.time'}
    for col in df.columns:
        if col not in protected:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    convert_map = {
        'Train #1 RTO Chamber A Temp F': 'RTO Chamber A Temp (C)',
        'Train #1 RTO Chamber B Temp F': 'RTO Chamber B Temp (C)',
        'Train #1 RTO Average Chamber Temp F': 'RTO Average Temp (C)'
    }
    for temp_f_col, temp_c_col in convert_map.items():
        if temp_f_col in df:
            df[temp_c_col] = (df[temp_f_col].replace(0, np.nan) - 32) * 5 / 9
    aliases = {
        'Train #1 RTO Chamber A Temp (C)': 'RTO Chamber A Temp (C)',
        'Train #1 RTO Chamber B Temp (C)': 'RTO Chamber B Temp (C)',
        'Train #1 RTO Average Chamber Temp (C)': 'RTO Average Temp (C)'
    }
    for original, standard in aliases.items():
        if original in df and standard not in df:
            df[standard] = pd.to_numeric(df[original], errors='coerce')
    if ('RTO Average Temp (C)' not in df and
            all(c in df for c in ('RTO Chamber A Temp (C)', 'RTO Chamber B Temp (C)'))):
        df['RTO Average Temp (C)'] = df[
            ['RTO Chamber A Temp (C)', 'RTO Chamber B Temp (C)']
        ].replace(0, np.nan).mean(axis=1)
    return df

# Facility-specific variable selection and analysis

MHF_CORRELATION_COLUMNS = [
    "Primary Hearth Avg Temp (C)", "Hearth 1 Temp (C)",
    "Dry Tons Per Hour", "N Feed Rate (g/hr)", "C Feed Rate (tons/hr)",
    "% Solid", "Stack Feed Rate (dscfm)",
    "CO2 (g/dry mega gram sludge)", "N2O (g/dry mega gram sludge)",
    "% N sludge to N2O", "NOx (g/dry mega gram sludge)",
    "NO (g/dry mega gram sludge)", "NO2 (g/dry mega gram sludge)",
    "CH4 (g/dry mega gram sludge)", "CO (g/dry mega gram sludge)",
    "O3 (g/dry mega gram sludge)",
]

FBI_CORRELATION_COLUMNS = [
    "fb.avg (C)", "bed.a (C)", "Wet Tons Per Hour (sludge feed rate)",
    "% Solid", "Dry Tons Per Hour", "% Nitrogen", "N Feed Rate (g/hr)",
    "Stack Feed Rate (dscfm)", "CO2 (g/dry mega gram sludge)",
    "CO (g/dry mega gram sludge)", "NOx (g/dry mega gram sludge)",
    "CH4 (g/dry mega gram sludge)", "N2O (g/dry mega gram sludge)",
    "O3 (g/dry mega gram sludge)", "NO (g/dry mega gram sludge)",
    "NO2 (g/dry mega gram sludge)", "% N sludge to N2O",
    "% N to NO", "% N to NO2",
]

DRYER_CORRELATION_COLUMNS = [
    "RTO Average Temp (C)", "RTO Chamber A Temp (C)", "RTO Chamber B Temp (C)",
    "% Solid", "Dry Tons Per Hour", "% Nitrogen", "N Feed Rate (g/hr)",
    "Stack Feed Rate (dscfm)", "Gas Usage (cf/hr)", "Actual MMBTU/hr",
    "Theoretical MMBTU/hr", "Wet Tons/hr In", "Wet Tons/hr Out",
    "H2O Evaporated (tons/hr)", "H2O Evaporated (mega gram/hr)",
    "% Carbon", "C Feed Rate (g/hr)",
    "CO2 (g/dry mega gram sludge)", "N2O (g/dry mega gram sludge)",
    "% N sludge to N2O", "NOx (g/dry mega gram sludge)",
    "NO (g/dry mega gram sludge)", "NO2 (g/dry mega gram sludge)",
    "CH4 (g/dry mega gram sludge)", "CO (g/dry mega gram sludge)",
    "O3 (g/dry mega gram sludge)", "CO2 (g/Mg Water Evaporated)",
    "N2O (g/Mg Water Evaporated)", "N2O (g/mega gram water evaporated)",
    "NOx (g/Mg Water Evaporated)", "NO (g/Mg Water Evaporated)",
    "NO2 (g/Mg Water Evaporated)", "CH4 (g/Mg Water Evaporated)",
    "CO (g/Mg Water Evaporated)", "O3 (g/Mg Water Evaporated)",
    "% C sludge to CO2",
]


def available_columns(df, names):
    """Select available variables and accept calculated stack-flow alternatives."""
    selected = list(names)
    if "Stack Feed Rate (dscfm)" not in df and "Calculated Stack Feed Rate (dscfm)" in df:
        selected = ["Calculated Stack Feed Rate (dscfm)" if col == "Stack Feed Rate (dscfm)" else col
                    for col in selected]
    return list(dict.fromkeys(col for col in selected if col in df))


def analyze_mhf(filename=MHF_FILE):
    data = prepare_mhf(filename)
    hearth_columns = [f"Hearth {i} Temp (C)" for i in range(1, 13)
                      if f"Hearth {i} Temp (C)" in data]
    hearth_correlations = correlation_statistics(data, hearth_columns, method="pearson")
    correlations = correlation_statistics(data,
        available_columns(data, MHF_CORRELATION_COLUMNS), method="pearson")

    x_col = "Primary Hearth Avg Temp (C)"
    y_col = "% N sludge to N2O"
    sample = valid_pair(data, x_col, y_col)
    linear = stats.linregress(sample[x_col], sample[y_col])
    rho, rho_p = stats.spearmanr(sample[x_col], sample[y_col])
    conversion_regression = pd.DataFrame([{
        "n": len(sample), "slope": linear.slope, "intercept": linear.intercept,
        "Pearson_r": linear.rvalue, "Pearson_p": linear.pvalue,
        "R2": linear.rvalue ** 2, "Spearman_rho": rho, "Spearman_p": rho_p,
    }])
    co_models, co_n, co_equations = fit_mhf_co_models(data)
    return {
        "hearth_correlations": hearth_correlations,
        "correlations": correlations,
        "conversion_regression": conversion_regression,
        "co_model_comparison": co_models,
        "co_model_equations": co_equations,
        "co_sample_size": co_n,
    }


def analyze_fbi_regressions(filename=FBI_REGRESSION_FILE):
    data = prepare_fbi_regression(filename)
    all_data, all_equations = fit_fbi_temperature_models(data, "Zeros retained")
    nonzero = data[data["N_to_N2O"] > 0].copy()
    nonzero_data, nonzero_equations = fit_fbi_temperature_models(nonzero, "Zeros excluded")
    return {
        "model_comparison": pd.concat([all_data, nonzero_data], ignore_index=True),
        "equations": {"zeros_retained": all_equations, "zeros_excluded": nonzero_equations},
    }


def analyze_fbi_gam(filename=FBI_REGRESSION_FILE):
    from pygam import LinearGAM, s

    data = prepare_fbi_regression(filename)
    data = data.loc[data["N_to_N2O"] > 0]
    X = data[["T_K"]].to_numpy()
    y = data["N_to_N2O"].to_numpy()
    model = LinearGAM(s(0, n_splines=6)).gridsearch(X, y, progress=False)
    fitted = model.predict(X)
    temperature_grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 300).reshape(-1, 1)
    confidence_limits = model.confidence_intervals(temperature_grid, width=0.95)
    statistics = pd.DataFrame([{
        "n": len(y), "R2": r2_score(y, fitted),
        "RMSE": np.sqrt(mean_squared_error(y, fitted)),
        "AIC": model.statistics_["AIC"],
        "pseudo_R2": str(model.statistics_["pseudo_r2"]),
        "edof": model.statistics_["edof"], "lambda": str(model.lam),
    }])
    return {"statistics": statistics, "model": model,
            "temperature_grid": temperature_grid, "confidence_intervals": confidence_limits}


def analyze_fbi_piecewise(filename=FBI_PIECEWISE_FILE):
    model_comparison, piecewise_parameters, exponential_parameters = fit_fbi_piecewise_models(filename)
    return {
        "model_comparison": model_comparison,
        "piecewise_parameters": piecewise_parameters,
        "exponential_parameters": exponential_parameters,
    }


def analyze_fbi_correlations(filename=FBI_CORR_FILE):
    data = prepare_fbi_correlations(filename)
    selected = available_columns(data, FBI_CORRELATION_COLUMNS)
    return {
        "pearson": correlation_statistics(data, selected, "pearson", iqr_multiplier=3.0),
        "spearman": correlation_statistics(data, selected, "spearman", iqr_multiplier=3.0),
    }


def analyze_dryer_correlations(filename=DRYER_FILE):
    data = prepare_dryer_correlations(filename)
    selected = available_columns(data, DRYER_CORRELATION_COLUMNS)
    return {
        "pearson": correlation_statistics(data, selected, "pearson", iqr_multiplier=3.0),
        "spearman": correlation_statistics(data, selected, "spearman", iqr_multiplier=3.0),
    }
