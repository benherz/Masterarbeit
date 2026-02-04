import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import shap
from matplotlib.colors import ListedColormap
import seaborn as sns

# Set color scheme for plots
SIGNAL_COLORS = { # convention to use all caps, because this remains constant
    "Sell": "#C62828",
    "Hold": "#78909C",
    "Buy": "#2E7D32"
}


##### Function to plot signal shares lineplot over time (quarterly) #####
def signal_shares_lineplot_quarters(recommendations_df: pd.DataFrame, llm_indicator=True):

    # Copy and ensure datetime
    df = recommendations_df.copy()
    df["temp_date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df["year"] = df["temp_date"].dt.year

    # Filter for years of interest
    df = df[(df["year"] >= 2000) & (df["year"] <= 2025)]
    
    # Only keep March, June, September, December
    df = df[df["temp_date"].dt.month.isin([3, 6, 9, 12])]

    # Count signals per quarter
    signal_counts = df.groupby(["temp_date", "action"]).size().unstack(fill_value=0)

    # Convert counts to shares (per quarter)
    signal_shares = signal_counts.div(signal_counts.sum(axis=1), axis=0)

    # Capitalize action names for legend, this also orders them alphabetically, enabling consistent color mapping
    signal_shares.columns = [col.capitalize() for col in signal_shares.columns]
    
    # Professional colors (consistent with your stackplot)
    colors = [SIGNAL_COLORS["Buy"], SIGNAL_COLORS["Hold"], SIGNAL_COLORS["Sell"]]

    # Plot yearly averages as lineplot
    ax = signal_shares.plot(
        figsize=(12, 6), 
        color=colors, 
        linewidth=2.5,
    )

    # Legend
    ax.legend(loc="upper right", title="Action", fontsize=12, title_fontsize=12, markerscale=1.5, borderaxespad=0.1)

    # Titles and labels
    ax.set_title("Share of LLM Signals Over Time (Yearly average)" if llm_indicator else "Share of Analyst Signals Over Time (Yearly average)",
                 fontsize=14)
    ax.set_ylabel("Share (%)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)

    # X-axis ticks: use actual datetime values for first of each year
    years = sorted(df["temp_date"].dt.year.unique())
    tick_dates = [signal_shares.index[signal_shares.index.year == y][0] for y in years]
    ax.set_xticks(tick_dates)
    ax.set_xticklabels(years, rotation=45)

    # Axis limits
    ax.set_xlim(signal_shares.index.min(), signal_shares.index.max())
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # Grid lines
    ax.grid(axis="y", linestyle="--", alpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.9)

    plt.tight_layout()
    plt.show()





##### Function to plot signal shares lineplot over time (yearly mean) #####
def signal_shares_lineplot_yearlymean(recommendations_df: pd.DataFrame, llm_indicator=True):

    # Copy and ensure datetime
    df = recommendations_df.copy()
    df["temp_date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df["year"] = df["temp_date"].dt.year

    # Filter for years of interest
    df = df[(df["year"] >= 2000) & (df["year"] <= 2025)]
    
    # Only keep March, June, September, December
    df = df[df["temp_date"].dt.month.isin([3, 6, 9, 12])]

    # Count signals per quarter
    signal_counts = df.groupby(["temp_date", "action"]).size().unstack(fill_value=0)

    # Convert counts to shares (per quarter)
    signal_shares = signal_counts.div(signal_counts.sum(axis=1), axis=0)

    # Take yearly mean across quarters
    signal_shares["year"] = signal_shares.index.year
    signal_shares_yearly = signal_shares.groupby("year").mean()
    
    # Capitalize action names for legend
    signal_shares_yearly.columns = [col.capitalize() for col in signal_shares_yearly.columns]
    
    # Professional colors (consistent with your stackplot)
    colors = [SIGNAL_COLORS["Buy"], SIGNAL_COLORS["Hold"], SIGNAL_COLORS["Sell"]]

    # Plot yearly averages as lineplot
    ax = signal_shares_yearly.plot(
        figsize=(12, 6), 
        color=colors, 
        linewidth=2.5,
    )

    # Legend
    ax.legend(loc="upper right", title="Action", fontsize=12, title_fontsize=12, markerscale=1.5, borderaxespad=0.1)

    # Titles and labels
    ax.set_title("Share of LLM Signals Over Time (Yearly average)" if llm_indicator else "Share of Analyst Signals Over Time (Yearly average)",
                 fontsize=14)
    ax.set_ylabel("Share (%)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)

    # Year ticks
    ax.set_xticks(signal_shares_yearly.index)
    ax.set_xticklabels(signal_shares_yearly.index, rotation=45)

    # Axis limits
    ax.set_xlim(signal_shares_yearly.index.min(), signal_shares_yearly.index.max())
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # Grid lines
    ax.grid(axis="y", linestyle="--", alpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.9)

    plt.tight_layout()
    plt.show()





##### Function to plot signal shares (over time) lineplot per economic sector with yearly mean #####
def signal_shares_lineplot_per_sector(recommendations_df: pd.DataFrame, llm_indicator=True):
    # Copy input df
    df = recommendations_df.copy()
    df["temp_date"] = pd.to_datetime(df["date"], format="%Y-%m")
    df["year"] = df["temp_date"].dt.year

    # Filter years and quarters
    df = df[(df["year"] >= 2000) & (df["year"] <= 2025)]
    df = df[df["temp_date"].dt.month.isin([3, 6, 9, 12])]

    # Determine unique sectors
    sectors = df["sector"].unique()
    n_sectors = df["sector"].nunique()

    # Setup grid of subplots
    ncols = 3
    nrows = int(np.ceil(n_sectors / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(18, 5 * nrows), sharex=False, sharey=True
    )
    axes = axes.flatten()

    for i, sector in enumerate(sectors):
        ax = axes[i]
        sector_df = df[df["sector"] == sector]

        # Count signals per quarter per sector
        signal_counts = sector_df.groupby(["temp_date", "action"]).size().unstack(fill_value=0)

        # Skip empty sectors
        if signal_counts.empty:
            continue

        # Convert counts to shares (per quarter)
        signal_shares = signal_counts.div(signal_counts.sum(axis=1), axis=0)

        # Take yearly mean across quarters
        signal_shares["year"] = signal_shares.index.year
        signal_shares_yearly = signal_shares.groupby("year").mean()

        # Capitalize + enforce consistent order (Buy, Hold, Sell)
        signal_shares_yearly = signal_shares_yearly.rename(columns=str.capitalize)
        signal_shares_yearly = signal_shares_yearly.reindex(columns=["Buy", "Hold", "Sell"], fill_value=0)

        # Plot yearly averages
        colors = [SIGNAL_COLORS["Buy"], SIGNAL_COLORS["Hold"], SIGNAL_COLORS["Sell"]]
        signal_shares_yearly.plot(ax=ax, color=colors, linewidth=2.5)

        # Titles & labels
        ax.set_title(sector, fontsize=12)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # X-axis ticks/labels
        ax.set_xticks(signal_shares_yearly.index[::2])
        ax.set_xticklabels(signal_shares_yearly.index[::2], rotation=45)
        ax.set_xlabel("Year", fontsize=12)

        # Legend
        ax.legend(loc="upper right", title="Action", fontsize=12, title_fontsize=12, markerscale=1.5, borderaxespad=0.1)

    # Remove last axis (plot frame) if not needed
    for j in range(n_sectors, len(axes)):
        if axes[j] in fig.axes:
            fig.delaxes(axes[j])

    # Global labels
    plt.suptitle(
        "Share of LLM Signals Over Time by Sector (Yearly Mean)"
        if llm_indicator
        else "Share of Analyst Signals Over Time by Sector (Yearly Mean)",
        fontsize=20,
        y=0.9,
        x = 0.5,
    )
   # fig.text(0.5, 0.04, "Year", ha="center", fontsize=18)
    fig.text(0.04, 0.5, "Share (%)", va="center", rotation="vertical", fontsize=18)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.9])
    plt.show()




##### Function to plot signal shares stackplot over time (quarterly) #####
def signal_shares_stackplot_quarters(recommendations_df: pd.DataFrame, llm_indicator=True):
    # Copy and convert date to datetime
    df = recommendations_df.copy()
    df["temp_date"] = pd.to_datetime(df["date"], format="%Y-%m")
    
    # Filter for years of interest
    df = df[(df["temp_date"].dt.year >= 2000) & (df["temp_date"].dt.year <= 2025)]
    
    # Filter for March, June, September, December
    df = df[df["temp_date"].dt.month.isin([3, 6, 9, 12])]
    
    # Count signals per unique date
    signal_counts = df.groupby(["temp_date", "action"]).size().unstack(fill_value=0)
    
    # Convert counts to shares
    signal_shares = signal_counts.div(signal_counts.sum(axis=1), axis=0)
    
    # Custom professional colors
    colors = [SIGNAL_COLORS["Buy"], SIGNAL_COLORS["Hold"], SIGNAL_COLORS["Sell"]]
    
    # Plot
    ax = signal_shares.plot.area(figsize=(12, 6), color=colors)
    ax.legend(loc='upper right')
    
    # Title and labels
    ax.set_title("Share of LLM Signals Over Time (quarterly)" if llm_indicator else "Share of Analyst Signals Over Time (quarterly)")
    ax.set_ylabel("Share (%)")
    ax.set_xlabel("Year")
    
    # X-axis ticks: use actual datetime values for first of each year
    years = sorted(df["temp_date"].dt.year.unique())
    tick_dates = [signal_shares.index[signal_shares.index.year == y][0] for y in years]
    ax.set_xticks(tick_dates)
    ax.set_xticklabels(years, rotation=45)
    
    # Y-axis limits
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()





##### Function to plot signal shares stackplot over time (yearly mean) #####
def signal_shares_stackplot_yearlymean(recommendations_df: pd.DataFrame, llm_indicator=True):

    # Filter for March, June, September, December
    df = recommendations_df.copy()
    # Create temp date as datetime variable for filtering etc.
    df["temp_date"] = pd.to_datetime(df["date"], format="%Y-%m")
    # Create year only column
    df["year"] = df["temp_date"].dt.year 

    # Filter for years of interest
    df = df[(df["year"] >= 2000) & (df["year"] <= 2025)]

    # Count signals per date
    signal_counts = df.groupby(["year", "action"]).size().unstack(fill_value=0)

    # Convert counts to shares
    signal_shares = signal_counts.div(signal_counts.sum(axis=1), axis=0)
    
    # Custom color palette
    colors = [SIGNAL_COLORS["Buy"], SIGNAL_COLORS["Hold"], SIGNAL_COLORS["Sell"]]

    # Actual plot
    ax = signal_shares.plot.area(figsize=(12, 6), color=colors)
    ax.legend(loc='upper right')

    # Plot title and axis labels
    ax.set_title("Share of LLM Signals Over Time" if llm_indicator else "Share of Analyst Signals Over Time")
    ax.set_ylabel("Share (%)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    
    # X-axis ticks for every year
    ax.set_xticks(signal_shares.index)  
    ax.set_xticklabels(signal_shares.index, rotation=45)

    # Axis limits
    ax.set_xlim(signal_shares.index.min(), signal_shares.index.max())
    ax.set_ylim(0, 1)  # since shares are between 0 and 1

    plt.xticks(rotation=45)  # rotate the automatically placed year labels
    plt.tight_layout()
    plt.show()





##### Function to plot signal shares per economic sector #####
def plot_signal_shares_per_sector(recommendations_df: pd.DataFrame, llm_indicator=True):
    # Copy input df to avoid modifying original
    plot_df = recommendations_df.copy()
    
    # Determine unique sectors
    sectors = plot_df["sector"].unique()
    n_sectors = plot_df["sector"].nunique()

    # Set up subplots
    ncols = 3
    nrows = int(np.ceil(n_sectors / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    axes = axes.flatten() # axes.flatten converts 2D array to 1D array for easy indexing in iteration

    for i, sector in enumerate(sectors):
        
        # Select current axis of loop
        ax = axes[i]

        # Filter df for current sector
        sector_df = plot_df[plot_df["sector"] == sector]
        # Determine number of recommendations and companies in sector
        n_recommendations = sector_df.shape[0]
        n_companies = sector_df["cik"].nunique()

        # Compute signal shares
        shares = sector_df["action"].value_counts(normalize=True)
        # Force consistent order: Buy, Sell, Hold
        shares = shares.reindex(["buy", "sell", "hold"], fill_value=0)
        
        # Bar plot
        ax.bar(shares.index.str.capitalize(), shares.values, color=[SIGNAL_COLORS[s.capitalize()] for s in shares.index])
        # Add grid lines
        ax.grid(True, axis='y', linestyle='--', color='gray', alpha=0.5)
        ax.set_title(f"Sector: {sector}", fontsize=14)
        ax.set_ylabel("Share (%)", fontsize=12)
        ax.set_ylim(0,1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        # Create custom legend without line and add line break
        legend_label = f"No Signals: {n_recommendations}\nNo Companies: {n_companies}"
        ax.legend([legend_label], fontsize=10, loc="upper right", frameon=False, handlelength=0, labelspacing=1.2) # handlelength 0 removes the line in legend

    # Remove last axis (plot frame) if not needed
    if n_sectors < len(axes):
        for j in range(n_sectors, len(axes)):
            fig.delaxes(axes[j])       

    plt.tight_layout()
    plt.suptitle(
        "Shares of LLM Recommendations by Sector" if llm_indicator else "Shares of Analyst Recommendations by Sector",
        fontsize=16,
        y=1.025
    )
    plt.show()    





##### Function to plot multiclass global SHAP values #####,
def plot_multiclass_shap(shap_values, X_data, model_classes, input_title):

    # Prepare Data
    X_cleaned = X_data.copy()
    X_cleaned.columns = [c.replace('_', ' ').title() for c in X_cleaned.columns]
    
    # Capitalize and format class labels
    class_labels = [str(cls).capitalize() for cls in model_classes]
    if input_title == "Feature Attribution Gemini Decisions": 
        color_map =  ListedColormap(sns.color_palette(["#C62828", "#2E7D32", "#78909C"]).as_hex())
    else: 
        color_map = ListedColormap(sns.color_palette(["#2E7D32", "#78909C", "#C62828"]).as_hex())
    
    # 4. Create Figure
    fig, ax = plt.subplots(figsize=(11, 7))
    
    # SHAP Plot
    shap.summary_plot(
        shap_values, 
        X_cleaned, 
        plot_type="bar", 
        class_names=class_labels,
        show=False,
        color = color_map
    )

    # 6. Refined Styling
    # Center the title and increase padding
    plt.title(input_title, 
              fontsize=14, 
              pad=20, 
              x = 0.2)

    # Add a light vertical grid to help compare bar lengths
    ax.xaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    
    # Clean up labels
    plt.xlabel("Mean |SHAP Value| (Average impact on model output magnitude)", 
               fontsize=12, labelpad=15, x = 0.2)
    plt.ylabel("Financial Metrics", fontsize=12, labelpad=15)
    plt.yticks(fontsize=10)

    # 7. Final Polish and Save
    plt.tight_layout()
    
    # Save with high resolution for print
    filename = f"SHAP_Global_{input_title.replace(' ', '_')}.png"
    plt.savefig(f"../figures/{filename}", dpi=400, bbox_inches='tight')
    plt.show()




##### Function to plot SHAP waterfall for individual samples #####,
def plot_waterfall(idx, 
                   shap_values_llm, shap_values_analyst,
                   input_df, 
                   explainer_llm, explainer_analyst, 
                   model_classes):
    # Needs to be redefined here to use in script
    financial_metric_cols = [
    # "price_to_earnings",    
    # "book_to_market",
    "interest_coverage",    
    # "market_capitalization",
    # "cash_flow_to_price",
    "return_on_equity",
    "working_capital_to_total_assets",
    "retained_earnings_to_total_assets",
    "ebit_to_total_assets",
    # "market_cap_to_total_liabilities",
    "sales_to_assets",
    "operating_margin",
    "debt_to_equity",
    "debt_to_assets"
]
    import matplotlib as mpl
    mpl.rcParams.update({'ytick.labelsize': 7})
    # Identify the predicted class for the sample
    prediction_llm = input_df.iloc[idx]["pred_llm"] 
    prediction_analyst = input_df.iloc[idx]["pred_analyst"]
    class_idx_llm = list(model_classes).index(prediction_llm)
    class_idx_analyst = list(model_classes).index(prediction_analyst)

    # Further information to identify the sample
    date = pd.to_datetime(input_df.iloc[idx]['date']).strftime('%m-%Y')
    cik = input_df.iloc[idx]['cik']

    # 2. Clean Feature Names & Round Data
    features = input_df[financial_metric_cols].apply(pd.to_numeric, errors='coerce')
    processed_features = [c.replace('_', ' ').title() for c in features.columns]
    formatted_data = [round(val, 2) for val in features.iloc[idx].values]

    # Two explanations: one for LLM, one for Analyst
    exp_llm = shap.Explanation(
        values=shap_values_llm[idx, :, class_idx_llm],
        base_values=explainer_llm.expected_value[class_idx_llm],
        data=formatted_data,
        feature_names=processed_features
    )
    exp_analyst = shap.Explanation(
        values=shap_values_analyst[idx, :, class_idx_analyst], 
        base_values=explainer_analyst.expected_value[class_idx_analyst],
        data=formatted_data,
        feature_names=processed_features
    )
   
    # 4. Plotting
    fig = plt.figure(figsize=(10, 10), dpi=300)
    
    # Subplot 1: LLM
    ax1 = fig.add_subplot(2, 1, 1)
    shap.plots.waterfall(exp_llm, max_display=10, show=False) # show = False so that the plot can further be adjusted
    # Surgical Font Size Adjustment, what a pain to find out
    ax1.tick_params(axis='y', labelsize=7) # shrink names
    plt.title(f"LLM Reasoning: {str(prediction_llm).capitalize()} for CIK {cik} in {date}", 
              x=0.0, loc='left', fontsize=10, fontweight='bold', pad=15)

    
    # Second Waterfall for Analyst
    ax2 = fig.add_subplot(2, 1, 2)
    shap.plots.waterfall(exp_analyst, show=False) 
    ax2.tick_params(axis='y', labelsize=7) 
    plt.title(f"Analyst Reasoning: {str(prediction_analyst).capitalize()} for CIK {cik} in {date}", 
              x=0.0, loc='left', fontsize=10, fontweight='bold', pad=15)
  
    
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=10, labelcolor='#555555') 
    plt.xlabel("Contribution to Prediction Confidence", fontsize=10, color='#555555')
    plt.tight_layout()
    plt.savefig(f"../figures/SHAP_Waterfall_llm_{prediction_llm}_analyst_{prediction_analyst}.png", dpi=400, bbox_inches='tight')
    plt.show()
