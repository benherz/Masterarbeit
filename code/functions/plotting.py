import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Set color scheme for plots
SIGNAL_COLORS = { # convention to use all caps, because this remains constant
    "Sell": "#d1495b",
    "Hold": "#6c757d",
    "Buy": "#1f4e79"
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




