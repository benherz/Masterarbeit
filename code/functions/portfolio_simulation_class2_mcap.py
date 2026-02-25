import pandas as pd
from tqdm import tqdm
import numpy as np

# Portfolio simulation that incorporates market capitalization for buy signal allocation

class PortfolioSimulation2_mcap:

    def __init__(self, initial_capital=100000, transaction_cost_rate=0.01):
        """Initialize the portfolio simulation with a given initial capital."""
        # Initialize the portfolio simulation with a given capital
        self.initial_capital = initial_capital
        self.transaction_cost_rate = transaction_cost_rate
        # Initialize cash position to track over time (starts out as initial capital ofc)
        self.cash = initial_capital
        # Portfolio value at t = 0 is just the initial capital
        self.portfolio_value = initial_capital
        # Inititialize an empty dictionary to store positions over time
        self.positions = {}
        # Initialize an empty list to store transactions
        self.transactions = []
        # Initialize empty DataFrames for stock prices and recommendations
        # Both dfs will be input by the user
        self.stock_prices = pd.DataFrame()  
        self.recommendations = pd.DataFrame()  
        self.mcap_df = pd.DataFrame()
        self.risk_free_rate_df = pd.DataFrame()
        # To keep track of the number of transactions
        self.no_transactions = 0
        # To keep track of skipped transactions
        self.skipped_transactions = []
        self.no_skipped_transactions = 0
        self.no_skipped_buys = 0
        self.no_skipped_sells = 0
        # To keep track of buys and sells
        self.no_buys = 0
        self.no_sells = 0
        self.no_holds = 0
        # To keep track of transaction costs
        self.transaction_costs = []

    ##### Function to load all input dataframes
    def load_dataframes(self, stock_prices_df: pd.DataFrame, recommendations_df: pd.DataFrame, 
                        mcap_df: pd.DataFrame, risk_free_rate_df: pd.DataFrame):
        """
        Load and preprocess all input DataFrames.

        Converts the 'date' column to monthly periods (Period[M]) to ensure alignment
        with recommendation dates. Stores the result in `self.stock_prices`.

        Args:
            stock_prices_df (pd.DataFrame): DataFrame with stock prices.
            recommendations_df (pd.DataFrame): DataFrame with recommendations.
            mcap_df (pd.DataFrame): DataFrame with market capitalization data.
            risk_free_rate_df (pd.DataFrame): DataFrame with risk-free rates.
        """
        # Stock price df
        stock_price_df = stock_prices_df.copy()
        stock_price_df['date'] = pd.to_datetime(stock_price_df['date']).dt.to_period('M')
        self.stock_prices = stock_price_df

        # DF containing buy/sell/hold signals
        recommendation_df = recommendations_df.copy()
        recommendation_df['date'] = pd.to_datetime(recommendation_df['date']).dt.to_period('M')
        self.recommendations = recommendation_df

        # DF containing mcap of every company at every month
        mcap_df = mcap_df.copy()
        mcap_df['date'] = pd.to_datetime(mcap_df['date']).dt.to_period('M')
        self.mcap_df = mcap_df

        # DF containing risk-free rate (US 3 month treasury bill)
        risk_free_rate_df = risk_free_rate_df.copy()
        risk_free_rate_df['date'] = pd.to_datetime(risk_free_rate_df['date']).dt.to_period('M')
        risk_free_rate_df.rename(columns = {'monthly_yield': 'rate'}, inplace=True)
        self.risk_free_rate_df = risk_free_rate_df


    ##### LLM, as well as analyst recommendations do not necessarily align with the monthly stock prices on a day-level.
    # Therefore, a function that returns the stock price of the same month as the recommendation is implemented.
    # If no price is available for the given month, the trade is skipped.
    def get_nearest_price(self, cik, date):
        """
        Retrieve the stock price for the specified stock (cik) at the given monthly period.

        Converts the input date to a monthly period if necessary, then looks up the price
        for that stock in the stored stock prices DataFrame. If the price for the specified
        month is not available, raises a ValueError.

        Args:
            cik (str): Stock identifier.
            date (pd.Period or datetime-like): Date (or convertible) for which to get the price.

        Returns:
            float: The stock price for the given month.

        Raises:
            ValueError: If no price data is available for the specified stock and date.
        """        
        # Convert input date to Period (monthly)
        if not isinstance(date, pd.Period):
            date = pd.to_datetime(date).to_period('M')
        stock_on_date = self.stock_prices[self.stock_prices['cik'] == cik].copy()
        # Simply return price of that month
        if date in stock_on_date['date'].values:
            return stock_on_date[stock_on_date['date'] == date]['price'].values[0]
        else:
            raise ValueError("No price data available.")

    
    ##### Function to grab nearest market cap for a given CIK and date
    def get_nearest_mcap(self, cik, mcap_df, date):
        """
        Retrieve the closest market cap date-wise for the specified stock (cik) at the given monthly period.

        Converts the input date to a timestamp if necessary, then looks up the market cap
        for that stock in the stored market cap DataFrame. If the market cap for the specified
        month is not available, it finds the closest available date and returns that market cap.

        Args:
            cik (str): Stock identifier.
            date (pd.Period or datetime-like): Date (or convertible) for which to get the market cap.

        Returns:
            float: The market cap for the given month.

        Raises:
            ValueError: If no market cap data is available for the specified stock and date.
        """
        # Convert input date to timestamp, if it isnt already
        # If date is a Period object, convert it to a Timestamp at the month's end
        if isinstance(date, pd.Period):
            date = date.end_time  # This gives the last day of the month

        # Convert input date to Timestamp (if not already)
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date) + pd.offsets.MonthEnd(0)

        # Normalize the date to ignore the time part (only compare day)
        date = date.normalize() 
        
        # Filter to target CIK rows
        cik_mcap_df = mcap_df[mcap_df['cik'] == cik].copy()
        if cik_mcap_df.empty:
            return None
        
        # Convert the 'date' column in cik_mcap_df to the timestamp at the end of the month
        cik_mcap_df['date'] = pd.to_datetime(cik_mcap_df['date'].dt.to_timestamp()).apply(lambda x: x + pd.offsets.MonthEnd(0))
        cik_mcap_df['date'] = cik_mcap_df['date'].dt.normalize()

        # Check if there's an exact match for the date
        exact_match = cik_mcap_df[cik_mcap_df['date'] == date]
        if not exact_match.empty:
           # print(f"Returning exact match for {cik} on date {date}")
            # Return the exact match if found
            closest_row = exact_match.iloc[0]
            return closest_row['market_cap']
        
        # If no exact match, calculate the closest match by date difference
        cik_mcap_df['date_diff'] = (cik_mcap_df['date'] - date).abs()

        # Find the row with the minimum date difference
        closest_row = cik_mcap_df.loc[cik_mcap_df['date_diff'].idxmin()]
        #print(f"Returning approximate match for {cik} on date {closest_row['date']} instead of {date}")

        return closest_row['market_cap']
            

        ##### Function to simulate a stock purchase
    def buy(self, cik, price, date, allocation):
        """
        Execute a buy transaction for one or more shares of the specified stock if sufficient allocated cash is available.

        Checks if the allocated amount is enough to buy at least one share (including transaction fees).
        If so, decreases cash by the amount spent, increases the stock position,
        records the transaction, and increments the transaction count.
        If the allocation is insufficient (even for one share including fees), the method exits without action.

        For simplicity, it assumes that the stock is bought at the end of the month.
        Currently, it buys as many whole shares as possible given the allocated cash and fees.

        Args:
            cik (str): Stock identifier.
            price (float): Price at which the stock is bought.
            date (pd.Period or datetime-like): Date of the transaction.
            allocation (float): The amount of cash allocated to this particular stock.

        Returns:
            None
        """

        transaction_cost_rate = self.transaction_cost_rate  # Assume this is defined elsewhere in the class

        # Compute effective cost per share including transaction fee
        effective_price_per_share = price * (1 + transaction_cost_rate)

        # Only buy if allocation is sufficient to buy at least one share including fees
        if effective_price_per_share > allocation:
            self.skipped_transactions.append((cik, date, "Not enough allocation for one share incl. fees"))
            self.no_skipped_transactions += 1
            self.no_skipped_buys += 1
            return

        # Compute how many shares can be bought with allocated cash
        qty = int(allocation // effective_price_per_share)  # // rounds DOWN to nearest integer -> only buy whole shares, 

        # Calculate total cost including fees (consistent with effective_price_per_share)
        final_cost = qty * effective_price_per_share

        # Calculate fee as the difference between final cost and pure stock price
        fee = final_cost - (qty * price)
        self.transaction_costs.append(fee)  # Track transaction fees

        # New cash balance is simply old one, minus the price of the stock including fees
        self.cash -= final_cost

        # To keep track of the number of positions, the dictionary is updated
        self.positions[cik] = self.positions.get(cik, 0) + qty

        # Append the transaction to the list to later on check positions over time
        self.transactions.append(('buy', cik, price, date, qty, final_cost))

        # Increment the number of transactions and buys
        self.no_buys += 1
        self.no_transactions += 1

  ##### Function to simulate a stock sale
    def sell(self, cik, price, date):
        """
        Execute a sell transaction for one share of the specified stock.

        Checks if the stock position exists and is greater than zero before selling.
        Updates cash balance, sells all stocks, records the transaction,
        and increments the transaction count. If no shares are held, the method exits without action.
        For simplicity, it assumes that the stock is sold at the end of the month.

        Args:
            cik (str): Stock identifier.
            price (float): Price at which the stock is sold.
            date (pd.Period or datetime-like): Date of the transaction.

        Returns:
            None
        """
        transaction_cost_rate = self.transaction_cost_rate  # Assume this is defined elsewhere in the class

        if self.positions.get(cik, 0) == 0: # 0 inside the brackets is simply the "default" to return, if cik is not in positions
            # If no shares are held, the transaction is skipped
            self.skipped_transactions.append((cik, date, "No shares held"))
            self.no_skipped_transactions += 1
            self.no_skipped_sells += 1
            return
        
        qty = self.positions.get(cik, 0)
        gross_proceeds = price * qty # Total proceeds from the sale
        fee = gross_proceeds * transaction_cost_rate # Transaction costs that occur for this sale
        self.transaction_costs.append(fee)  # Append fee to transaction costs
        net_proceeds = gross_proceeds - fee # Actual earnings, after fees are deducted
        self.cash += net_proceeds # Increase cash by the price of the stock times the quantity held
        #print(f"Selling {qty} shares of {cik} at {price} on {date} for a total of {price * qty}.")
        self.positions[cik] = 0 # Set the position to 0, since we sold all shares
        self.transactions.append(('sell', cik, price, date, qty, net_proceeds))
        # Increment the number of transactions and sells
        self.no_sells += 1
        self.no_transactions += 1
       


    ##### Function that puts it all together
    def simulate_trading(self):
        """
        Simulate monthly (quarterly) portfolio rebalancing based on 'buy', 'sell', and 'hold' signals.

        For each unique month in the recommendations:
            1. Execute all 'sell' signals first, liquidating full positions at the given month's price.
            2. Evenly distribute the available cash across all 'buy' signals for that month.
            3. Buy as many whole shares as possible for each recommended stock using the allocated amount.

        Notes:
            - 'Hold' signals are ignored.
            - If price data for a stock is unavailable for the given month, the transaction is skipped.
            - Transactions and skipped actions are logged for analysis.

        Returns:
            None
        """
        unique_dates = self.recommendations['date'].unique()

        for date in tqdm(unique_dates, desc="Simulating Trades"):
            recs_on_date = self.recommendations[self.recommendations['date'] == date]

            # Step 1: Process hold signals
            hold_recs = recs_on_date[recs_on_date['action'] == 'hold']
            for _, rec in hold_recs.iterrows():
                self.skipped_transactions.append((rec['cik'], date, "Hold signal - no action taken"))
                self.no_skipped_transactions += 1
                self.no_holds += 1
                continue


            # Step 2: Process all SELL signals first 
            sell_recs = recs_on_date[recs_on_date['action'] == 'sell']
            # Use iterrows, because we need more than 1 column: cik, price, and date
            for _, rec in sell_recs.iterrows():
                cik = rec['cik']
                try:
                    price = self.get_nearest_price(cik, date)
                    self.sell(cik, price, date)
                except ValueError as e:
                    self.skipped_transactions.append((cik, date, str(e)))
                    self.no_skipped_transactions += 1
                    self.no_skipped_sells += 1
                    continue

            #  Step 3: Process BUY signals with equal cash allocation 
            buy_recs = recs_on_date[recs_on_date['action'] == 'buy']

            if len(buy_recs) == 0:
                continue  # Nothing to buy this month
            
            # Money to spend on each stock is based on share of total mcap among all buy signals
            buy_ciks = buy_recs['cik'].values
            # Get market caps for all buy signals
            nearest_caps = []
            for cik in buy_ciks:
                mcap = self.get_nearest_mcap(cik, self.mcap_df, date)
                if mcap is not None:
                    nearest_caps.append((cik, mcap))
                else:
                    self.skipped_transactions.append((cik, date, "No market cap data found"))
                    self.no_skipped_transactions += 1
                    continue

            # Compute total mcap of all buy signals
            buy_signals_total_mcap = sum(mcap for _, mcap in nearest_caps)
            mcaps_store = []
            # Step 3: Allocate and buy
            for cik, market_cap in nearest_caps:
                try:
                    price = self.get_nearest_price(cik, date)


                    # Allocate money based on share of total market cap
                    allocation = (market_cap / buy_signals_total_mcap) * self.cash

                    self.buy(cik, price, date, allocation)
                    mcaps_store.append(market_cap)

                except (ValueError, IndexError) as e:
                    self.skipped_transactions.append((cik, date, str(e)))
                    self.no_skipped_transactions += 1
                    self.no_skipped_buys += 1
                    continue

                


    ###### Function to get positions (i.e. cash and stocks) at a specific date
    # This function uses the transactions list, which contains action, cik, price and date of any transaction
    # First loop is used to obtain all positions at a given date, including cash
    # Second loop is used to obtain the value of each position at a given date
    def get_positions_at_date(self, date):
        """
        Returns a snapshot of portfolio positions (stocks and cash) at a given date.

        Reconstructs historical positions by replaying all transactions up to the input date.
        Each stock position is valued using the nearest available monthly price.
        Cash is included as its current balance. Only positive stock positions are retained.

        Args:
            date (str, datetime-like, or pd.Period): The date to inspect positions at. Converted to monthly Period.

        Returns:
            pd.DataFrame: A DataFrame indexed by date with columns ['cik', 'quantity', 'total_value'].
                        Includes both stock positions and cash.

        Notes:
            - The first loop replays all transactions up to the given date to reconstruct the portfolio state (cash + share counts).
            - The second loop assigns a monetary value to each position (using current price for each stock at the input date),
            and filters out stocks with zero or negative holdings. Cash is always included.                        
        """        
        # Convert input date to Period (monthly)
        if not isinstance(date, pd.Period):
            date = pd.to_datetime(date).to_period('M')

        # Initialize positions as dictionary with cash only. Cash starts at initial capital
        # I name this positions_snapshot to avoid confusion with self.positions
        positions_snapshot = {'cash': self.initial_capital} 

        # Loop through transactions and update positions as transactions were made
        for action, cik, price, tx_date, qty, cost in self.transactions:
            # Only consider transactions up to the specified date
            if tx_date <= date:
                if action == 'buy':
                    positions_snapshot[cik] = positions_snapshot.get(cik, 0) + qty
                    positions_snapshot['cash'] -= cost
                elif action == 'sell':
                    positions_snapshot[cik] = positions_snapshot.get(cik, 0) - qty
                    positions_snapshot['cash'] += cost

        # Second loop is needed, because the price of a stock at the input date is not the same as in the previous loop
        # Therefore, we have to loop over all positions and fetch the nearest price for each stock
        positions_snapshot = {k: v for k, v in positions_snapshot.items() if (k == 'cash' or v > 0)} 
        # Empty list to store positions
        positions_snapshot_list = []
        for cik, qty in positions_snapshot.items():
            if cik == 'cash':
                val = qty   # cash value is simply the cash amount
            else:
                price = self.get_nearest_price(cik, date)
                val = price * qty
            # I decided to append the integer of quantity to make it easier to read (minor rounding issues in cash positions)
            positions_snapshot_list.append((date, cik, int(qty), val))
        # Convert to df
        positions_snapshot_df = pd.DataFrame(
            positions_snapshot_list,
            columns=['date', 'cik', 'quantity', 'total_value']
        )
        return positions_snapshot_df.set_index('date')
    
    
    ##### Function to get the monetary value of the portfolio at a specific date
    def get_portfolio_value(self, date=None):
        """
        Calculate the total monetary value of the portfolio at a specific date.

        Args:
            date (str, datetime, or pd.Period, optional): The date to evaluate the portfolio.
                If None, the current date is used.

        Returns:
            float: The total portfolio value rounded to 4 decimals.

        Notes:
            - Uses get_positions_at_date to compute the market value of all holdings.
            - Assumes positions already reflect all transactions up to the given date.
        """    
        # Convert input date to Period (monthly)
        if not isinstance(date, pd.Period):
            date = pd.to_datetime(date).to_period('M')
        positions_df = self.get_positions_at_date(date)
        
        # Total value is simply the sum of all positions' values, 
        # since get_positions_at_date already calculates the total value of each position
        total_value = positions_df["total_value"].sum()
        
        return np.round(total_value, 4)



    ##### Function to get positions over time
    # This function returns a DataFrame of positions at each unique transaction date
    """
    Generate a time series of portfolio snapshots based on transaction history.

    For each unique month in which a transaction occurred, this function computes 
    the portfolio's composition (quantities and total values) by calling 
    `get_positions_at_date(date)`. It returns a concatenated DataFrame of all such 
    monthly snapshots, indexed by date.

    Returns:
        pd.DataFrame: A DataFrame containing positions (quantity and total_value) 
        for each CIK at each relevant month, with missing values filled as 0.
    """    
    def get_positions_over_time(self):
        """Returns a DataFrame of positions at each unique transaction date (as Periods)."""
        # Set returns a list of unique dates from transactions
        # tx[3] is the date of the transaction
        # Extract Periods from the transaction dates
        #dates = [pd.Period(tx[3], freq='M') for tx in self.recommendations]
        dates = self.recommendations['date'].unique()
        # Sort dates to ensure chronological order
        dates = sorted(dates)

        # Generate full monthly range from min to max date
        unique_dates = pd.period_range(start=min(dates), end=max(dates), freq='M')
        all_positions = []

        for date in tqdm(unique_dates, desc="Getting Positions Over Time"):
            positions_df = self.get_positions_at_date(date)
            all_positions.append(positions_df)

        combined_df = pd.concat(all_positions)
        combined_df = combined_df.fillna(0)
        return combined_df.reset_index()


    ##### Function to calculate monthly returns
    # This function calculates the monthly returns of the portfolio based on the portfolio value 
    # at the end of the previous month and the portfolio value at the end of the current month (common practice)
    """
    Calculate the monthly returns of the portfolio over the simulation period.

    For each month between the first and last recommendation date, this function:
    - Computes the portfolio value at the beginning and end of the month.
    - Calculates the return as the percentage change from start to end.
    - Appends the raw and normalized start/end values and return to a results list.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'month': monthly period,
            - 'start_value' / 'end_value': raw portfolio values,
            - 'return': monthly return (as decimal),
            - 'normalized_start_value' / 'normalized_end_value': relative to initial capital.
    """    
    def calculate_monthly_returns(self):
        
        # Work with copy of recommendations just to be sure
        recs = self.recommendations.copy()
        # Determine range of months to loop over
        all_months = pd.period_range(
            start=recs['date'].min(),
            end=recs['date'].max()
        )

        # Empty list to append monthly returns to 
        monthly_returns = []

        for i in tqdm(range(1, len(all_months)), desc="Calculating Monthly Returns"):

            month = all_months[i]
            prev_month = all_months[i - 1]

            value_start = self.get_portfolio_value(prev_month)
            value_end = self.get_portfolio_value(month)

            ret = (value_end - value_start) / value_start if value_start > 0 else 0

            monthly_returns.append({
                'month': month,
                'start_value': value_start,
                'end_value': value_end,
                'return': np.round(ret, 6)
            })
       
        monthly_returns = pd.DataFrame(monthly_returns)
        monthly_returns["normalized_start_value"] = monthly_returns["start_value"] / self.initial_capital
        monthly_returns["normalized_end_value"] = monthly_returns["end_value"] / self.initial_capital
        
        # Compute excess return
        monthly_returns = pd.merge(
            monthly_returns,
            self.risk_free_rate_df,
            left_on='month',
            right_on='date',
            how='left'
        )
        monthly_returns.drop(columns=['date', 'yearly_yield'], inplace=True)  # Drop the 'date' column after merging
        monthly_returns["excess_return"] = monthly_returns["return"] - monthly_returns["rate"]

        return monthly_returns
    
    ##### Function to calculate portfolio statistics
    def portfolio_statistics(self, monthly_returns=None):
        """
        Calculate key statistics of the portfolio based on monthly returns.

        Args:
        monthly_returns (DataFrame, optional): Precomputed monthly returns.

        Computes:
        - Mean return
        - Geometric mean return
        - Standard deviation of returns
        - Annualized return
        - Annualized standard deviation

        Returns:
            dict: A dictionary containing the calculated statistics.
        """
        if monthly_returns is None:
            monthly_returns = self.calculate_monthly_returns()
        
        # Calculate mean return
        mean_return = monthly_returns['return'].mean()
        mean_excess_return = monthly_returns['excess_return'].mean()

        # Calculate geometric mean return (better for compounding effects)
        geometric_mean_return = (1 + monthly_returns['return']).prod() ** (1 / len(monthly_returns)) - 1
        
        # Calculate standard deviation of returns
        std_return = monthly_returns['return'].std()
        std_excess_return = monthly_returns['excess_return'].std()

        # Calculate annualized return and standard deviation
        annualized_return = (1 + mean_return) ** 12 - 1
        annualized_std = std_return * np.sqrt(12)

        # Calculate Sharpe ratio
        sharpe_ratio = (mean_excess_return) / std_excess_return if std_excess_return > 0 else 0

        pf_statistics = {
            "Mean return (monthly)": np.round(mean_return,6),
            "Annualized mean return": np.round(annualized_return,6),
            "Geometric mean return (monthly)": np.round(geometric_mean_return,6),
            "Excess mean return (monthly)": np.round(mean_excess_return,6),
            "Standard deviation (monthly)": np.round(std_return,6),
            "Annualized standard deviation": np.round(annualized_std,6),
            "Annualized Sharpe Ratio": np.round(sharpe_ratio*np.sqrt(12),6),
            "Number of buys": self.no_buys,
            "Number of sells": self.no_sells,
            "Number of holds": self.no_holds,
            "Total number of transactions": self.no_transactions,
            "Number of skipped buys": self.no_skipped_buys,
            "Number of skipped sells": self.no_skipped_sells,
            "Total number of skipped transactions": self.no_skipped_transactions,
            "Total amount of transaction costs": np.round(np.sum(self.transaction_costs), 6)
        }

        return pf_statistics
