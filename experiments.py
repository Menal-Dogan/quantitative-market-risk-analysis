from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from data import fetch_prices, handle_missing_values, validate_data
from portfolio import (
    compute_simple_returns,
    compute_log_returns,
    aggregate_portfolio_returns,
)
from risk_measures import (
    compute_losses,
    historical_var,
    historical_es,
    parametric_gaussian_var,
    parametric_gaussian_es,
    compare_risk_measures,
)
from backtesting import (
    compute_rolling_volatility,
    rolling_var_backtest,
    kupiec_pof_test,
)


def run_empirical_study() -> None:
    """
    Executes the full empirical market risk analysis pipeline:
    1. Data ingestion, cleaning, and validation (2018-2025).
    2. Portfolio return construction (60% SPY / 30% TLT / 10% GLD).
    3. Distributional diagnostics (Skewness, Excess Kurtosis, Jarque-Bera, Shapiro-Wilk).
    4. Static risk comparison (Historical vs. Gaussian VaR & ES at 95% and 99%).
    5. Out-of-sample one-day-ahead VaR backtesting (W = 250) and Kupiec POF tests.
    6. Sensitivity analysis across estimation windows (W = 125, 250, 500).
    7. Generation of the 5 diagnostic figures in ./figures/.
    """
    # --- Configuration & Documented Assumptions ---
    tickers = ["SPY", "TLT", "GLD"]
    weights = np.array([0.60, 0.30, 0.10])  # Daily constant-mix rebalancing assumption
    start_date = "2018-01-01"
    end_date = "2025-01-01"
    default_window = 250  # ~1 trading year estimation window
    confidence_levels = (0.95, 0.99)

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Data Ingestion & Validation ---
    raw_prices = fetch_prices(tickers, start_date=start_date, end_date=end_date)
    prices = handle_missing_values(raw_prices)[tickers]
    validate_data(prices)

    # --- 2. Returns & Portfolio Aggregation ---
    simple_returns = compute_simple_returns(prices)
    log_returns = compute_log_returns(prices)
    port_returns = aggregate_portfolio_returns(simple_returns, weights)
    port_log_returns = np.log1p(port_returns)
    losses = compute_losses(port_returns)

    print("=" * 72)
    print("1. DATASET & RETURN CONVENTIONS SUMMARY")
    print("=" * 72)
    print(f"Assets & Weights      : {dict(zip(tickers, weights))}")
    print(f"Sample Period         : {port_returns.index.min().date()} to {port_returns.index.max().date()}")
    print(f"Total Return Obs (T)  : {len(port_returns)}")
    print(f"Annualized Mean (Simp): {port_returns.mean() * 252:.2%}")
    print(f"Annualized Vol (Simp) : {port_returns.std(ddof=1) * np.sqrt(252):.2%}")
    print(f"Max |R_simple - r_log|: {np.max(np.abs(port_returns - port_log_returns)):.6f}")

    # --- 3. Distributional Diagnostics (Non-Normality Audit) ---
    loss_arr = losses.to_numpy()
    skew_val = float(stats.skew(loss_arr))
    kurt_ex_val = float(stats.kurtosis(loss_arr, fisher=True))
    jb_stat, jb_pval = stats.jarque_bera(loss_arr)
    sw_stat, sw_pval = stats.shapiro(loss_arr)

    print("\n" + "=" * 72)
    print("2. DISTRIBUTIONAL DIAGNOSTICS OF PORTFOLIO LOSSES (L_t = -R_{P,t})")
    print("=" * 72)
    print(f"Mean Daily Loss       : {np.mean(loss_arr):.4%}")
    print(f"Daily Loss Volatility : {np.std(loss_arr, ddof=1):.4%}")
    print(f"Skewness (S)          : {skew_val:.4f}  (Positive => heavy right loss tail)")
    print(f"Excess Kurtosis (K_F) : {kurt_ex_val:.4f}  (Gaussian benchmark = 0.0)")
    print(f"Jarque-Bera Stat      : {jb_stat:.2f} (p-value = {jb_pval:.4e})")
    print(f"Shapiro-Wilk Stat     : {sw_stat:.4f} (p-value = {sw_pval:.4e})")

    # --- 4. Full-Sample Risk Measures Comparison (95% vs 99%) ---
    risk_table = compare_risk_measures(losses, confidence_levels=confidence_levels)
    print("\n" + "=" * 72)
    print("3. IN-SAMPLE RISK MEASURES (DAILY RELATIVE LOSS)")
    print("=" * 72)
    print((risk_table * 100).round(3).to_string())

    # --- 5. One-Day-Ahead Out-of-Sample Backtesting (W = 250) ---
    print("\n" + "=" * 72)
    print(f"4. ONE-DAY-AHEAD BACKTEST & KUPIEC POF TEST (WINDOW W = {default_window})")
    print("=" * 72)
    backtest_results: dict[tuple[float, str], pd.DataFrame] = {}
    kupiec_rows = []

    for cl in confidence_levels:
        for method in ("historical", "gaussian"):
            bt_df = rolling_var_backtest(
                losses, window=default_window, confidence_level=cl, method=method
            )
            backtest_results[(cl, method)] = bt_df
            k_res = kupiec_pof_test(bt_df["Violation"], confidence_level=cl)
            kupiec_rows.append(
                {
                    "Confidence": f"{cl:.0%}",
                    "Method": method.capitalize(),
                    "OOS_Days": k_res["n_observations"],
                    "Observed_Viol": k_res["observed_violations"],
                    "Expected_Viol": round(k_res["expected_violations"], 2),
                    "Obs_Rate (%)": round(k_res["observed_rate"] * 100, 2),
                    "Exp_Rate (%)": round(k_res["expected_rate"] * 100, 2),
                    "LR_POF": round(k_res["lr_stat"], 4),
                    "p_value": round(k_res["p_value"], 4),
                    "Reject_H0 (5%)": k_res["reject_h0"],
                }
            )

    kupiec_df = pd.DataFrame(kupiec_rows)
    print(kupiec_df.to_string(index=False))

    # --- 6. Sensitivity to Estimation Window Length ---
    print("\n" + "=" * 72)
    print("5. ESTIMATION WINDOW SENSITIVITY (99% VaR BACKTEST)")
    print("=" * 72)
    sens_rows = []
    for w in (125, 250, 500):
        for method in ("historical", "gaussian"):
            bt_w = rolling_var_backtest(losses, window=w, confidence_level=0.99, method=method)
            k_w = kupiec_pof_test(bt_w["Violation"], confidence_level=0.99)
            sens_rows.append(
                {
                    "Window": w,
                    "Method": method.capitalize(),
                    "OOS_Days": k_w["n_observations"],
                    "Obs_Viol": k_w["observed_violations"],
                    "Exp_Viol": round(k_w["expected_violations"], 2),
                    "Obs_Rate (%)": round(k_w["observed_rate"] * 100, 2),
                    "p_value": round(k_w["p_value"], 4),
                    "Reject_H0": k_w["reject_h0"],
                }
            )
    print(pd.DataFrame(sens_rows).to_string(index=False))

    # --- 7. Generate Diagnostic Figures ---
    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: Portfolio Cumulative Returns
    cum_assets = (1.0 + simple_returns).cumprod()
    cum_port = (1.0 + port_returns).cumprod()

    fig, ax = plt.subplots(figsize=(11, 5))
    for col in tickers:
        ax.plot(cum_assets.index, cum_assets[col], alpha=0.55, linewidth=1.2, label=f"{col}")
    ax.plot(
        cum_port.index,
        cum_port,
        color="black",
        linewidth=2.2,
        label="Portfolio (60% SPY / 30% TLT / 10% GLD)",
    )
    ax.set_title("Cumulative Wealth Index (Base = 1.0)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth Factor")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "01_cumulative_returns.png", dpi=150)
    plt.close(fig)

    # Figure 2: Distribution of Portfolio Losses with VaR and ES Thresholds
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(loss_arr * 100, bins=80, density=True, alpha=0.55, color="steelblue", label="Empirical Losses")
    x_grid = np.linspace(loss_arr.min(), loss_arr.max(), 400)
    gauss_pdf = stats.norm.pdf(x_grid, loc=np.mean(loss_arr), scale=np.std(loss_arr, ddof=1))
    ax.plot(x_grid * 100, gauss_pdf / 100, color="black", linestyle="--", linewidth=1.6, label="Fitted Gaussian PDF")

    thresholds = [
        (parametric_gaussian_var(losses, 0.95) * 100, "darkorange", "--", "Gaussian VaR 95%"),
        (historical_var(losses, 0.95) * 100, "darkorange", "-", "Historical VaR 95%"),
        (parametric_gaussian_var(losses, 0.99) * 100, "crimson", "--", "Gaussian VaR 99%"),
        (historical_var(losses, 0.99) * 100, "crimson", "-", "Historical VaR 99%"),
        (historical_es(losses, 0.99) * 100, "purple", ":", "Historical ES 99%"),
    ]
    for val, color, ls, label in thresholds:
        ax.axvline(val, color=color, linestyle=ls, linewidth=1.6, label=f"{label} ({val:.2f}%)")

    ax.set_title("Empirical Portfolio Daily Loss Distribution vs. VaR & ES Thresholds")
    ax.set_xlabel("Daily Loss L_t = -R_{P,t} (%)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "02_loss_distribution_var_es.png", dpi=150)
    plt.close(fig)

    # Figure 3: Rolling Volatility and Rolling VaR (W = 250)
    roll_vol = compute_rolling_volatility(port_returns, window=default_window, annualize=True)
    bt_hist_95 = backtest_results[(0.95, "historical")]
    bt_hist_99 = backtest_results[(0.99, "historical")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(roll_vol.index, roll_vol * 100, color="navy", linewidth=1.4)
    ax1.set_title(f"One-Day-Ahead Annualized Rolling Volatility (W = {default_window} Days)")
    ax1.set_ylabel("Annualized Volatility (%)")

    ax2.plot(bt_hist_95.index, bt_hist_95["VaR_Forecast"] * 100, color="darkorange", linewidth=1.4, label="Rolling Historical VaR 95%")
    ax2.plot(bt_hist_99.index, bt_hist_99["VaR_Forecast"] * 100, color="crimson", linewidth=1.4, label="Rolling Historical VaR 99%")
    ax2.set_title(f"One-Day-Ahead Rolling Historical VaR Forecasts (W = {default_window} Days)")
    ax2.set_xlabel("Evaluation Date")
    ax2.set_ylabel("Daily VaR Forecast (%)")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "03_rolling_volatility_and_var.png", dpi=150)
    plt.close(fig)

    # Figure 4: Realized Losses vs One-Day-Ahead VaR Forecasts (95% Historical)
    viol_df = bt_hist_95[bt_hist_95["Violation"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(bt_hist_95.index, bt_hist_95["Realized_Loss"] * 100, color="slategray", alpha=0.65, linewidth=0.9, label="Realized Daily Loss L_t")
    ax.plot(bt_hist_95.index, bt_hist_95["VaR_Forecast"] * 100, color="navy", linewidth=1.5, label="1-Day-Ahead Historical VaR 95%")
    ax.scatter(
        viol_df.index,
        viol_df["Realized_Loss"] * 100,
        color="crimson",
        s=28,
        zorder=5,
        label=f"VaR Violations (n = {len(viol_df)})",
    )
    ax.set_title("Out-of-Sample Backtest: Realized Daily Losses vs. 1-Day-Ahead 95% Historical VaR")
    ax.set_xlabel("Evaluation Date")
    ax.set_ylabel("Daily Loss / VaR (%)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "04_backtest_realized_vs_var.png", dpi=150)
    plt.close(fig)

    # Figure 5: Comparison of Rolling Historical vs Parametric Gaussian VaR (95% and 99%)
    bt_gauss_95 = backtest_results[(0.95, "gaussian")]
    bt_gauss_99 = backtest_results[(0.99, "gaussian")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(bt_hist_95.index, bt_hist_95["VaR_Forecast"] * 100, color="navy", linewidth=1.4, label="Historical VaR 95%")
    ax1.plot(bt_gauss_95.index, bt_gauss_95["VaR_Forecast"] * 100, color="teal", linestyle="--", linewidth=1.4, label="Gaussian VaR 95%")
    ax1.set_title("Rolling 95% VaR Comparison: Historical vs. Parametric Gaussian")
    ax1.set_ylabel("Daily VaR (%)")
    ax1.legend(loc="upper left")

    ax2.plot(bt_hist_99.index, bt_hist_99["VaR_Forecast"] * 100, color="crimson", linewidth=1.4, label="Historical VaR 99%")
    ax2.plot(bt_gauss_99.index, bt_gauss_99["VaR_Forecast"] * 100, color="darkorange", linestyle="--", linewidth=1.4, label="Gaussian VaR 99%")
    ax2.set_title("Rolling 99% VaR Comparison: Historical vs. Parametric Gaussian")
    ax2.set_xlabel("Evaluation Date")
    ax2.set_ylabel("Daily VaR (%)")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "05_historical_vs_parametric_var.png", dpi=150)
    plt.close(fig)

    print("\nAll 5 figures successfully saved to ./figures/")


if __name__ == "__main__":
    run_empirical_study()