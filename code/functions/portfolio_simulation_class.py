import pandas as pd
from tqdm import tqdm
import numpy as np

# Throughout this class, I generally only consider MONTHLY data. (Recommendations and stock prices)
# LLM, as well as sell-side analyst recommendations for a given month will be processed along with the
# monthly closing price of the respective stock. 
# In the case of LLM recommendations, this isn't even a big issue, since the recommendations were generated
# with financial statements published at the end of a given month.
# In the case of sell-side analyst recommendations, this approach assumes, that the recommendation is valid for at leastthe entire month.

class PortfolioSimulation:
   
    def __init__(self, initial_capital=100):
        """Initialize the portfolio simulation with a given initial capital."""
        # Initialize the portfolio simulation with a given capital
        self.initial_capital = initial_capital
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

    ##### Function to load input stock prices
    def load_stock_prices(self, stock_prices_df):
        """
        Load and preprocess the stock prices DataFrame.

        Converts the 'date' column to monthly periods (Period[M]) to ensure alignment
        with recommendation dates. Stores the result in `self.stock_prices`.

        Args:
            stock_prices_df (pd.DataFrame): DataFrame with at least 'cik', 'price', and 'date' columns.
        """
        df = stock_prices_df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.to_period('M')
        self.stock_prices = df


    ##### Function to load input recommendations
    def load_recommendations(self, recommendations_df):
        """
        Load and preprocess the recommendations DataFrame.

        Converts the 'date' column to monthly periods (Period[M]) for consistency
        with other components of the simulation. Stores the processed DataFrame
        in the `self.recommendations` attribute.

        Args:
            recommendations_df (pd.DataFrame): DataFrame containing at least 'cik', 'action', and 'date' columns.
    """
        df = recommendations_df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.to_period('M')
        self.recommendations = df


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


    ##### Function to simulate a stock purchase
    def buy(self, cik, price, date):
        """
        Execute a buy transaction for one share of the specified stock if sufficient cash is available.

        Checks if there is enough cash to buy the stock at the given price.
        If so, decreases cash by the price, increases the stock position by one share,
        records the transaction, and increments the transaction count.
        If cash is insufficient, the method exits without action.
        For simplicity, it assumes that the stock is bought at the end of the month.
        Further, I always just buy or sell 1 stock at a given point in time (for now).

        Args:
            cik (str): Stock identifier.
            price (float): Price at which the stock is bought.
            date (pd.Period or datetime-like): Date of the transaction.

        Returns:
            None
        """        
        # Only buy if capital is sufficient
        if price > self.cash:
            #print(f"Skipped BUY on {date}: Not enough capital.")
            self.skipped_transactions.append((cik, date, "Not enough capital"))
            self.no_skipped_transactions += 1
            self.no_skipped_buys += 1
            return
        # New cash balance is simply old one, minus the price of the stock
        self.cash -= price
        # To keep track of the number of positions, the dictionary is updated
        self.positions[cik] = self.positions.get(cik, 0) + 1 #. get() returns 0 if cik is not in positions, otherwise returns the current value and then we add 1
        # Append the transaction to the list to later on check positions over time
        self.transactions.append(('buy', cik, price, date))
        # Increment the number of transactions and buys
        self.no_buys += 1
        self.no_transactions += 1


    ##### Function to simulate a stock sale
    def sell(self, cik, price, date):
        """
        Execute a sell transaction for one share of the specified stock.

        Checks if the stock position exists and is greater than zero before selling.
        Updates cash balance, reduces the stock position by one share, records the transaction,
        and increments the transaction count. If no shares are held, the method exits without action.
        For simplicity, it assumes that the stock is sold at the end of the month.
        Further, I always just buy or sell 1 stock at a given point in time (for now)

        Args:
            cik (str): Stock identifier.
            price (float): Price at which the stock is sold.
            date (pd.Period or datetime-like): Date of the transaction.

        Returns:
            None
        """        
        if self.positions.get(cik, 0) == 0: # 0 inside the brackets is simply the "default" to return, if cik is not in positions
            # If no shares are held, the transaction is skipped
            self.skipped_transactions.append((cik, date, "No shares held"))
            self.no_skipped_transactions += 1
            self.no_skipped_sells += 1
            return
        self.cash += price
        self.positions[cik] -= 1
        self.transactions.append(('sell', cik, price, date))
        # Increment the number of transactions and sells
        self.no_sells += 1
        self.no_transactions += 1
        
        


    ##### Function that puts it all together
    def simulate_trading(self):
        """
        Simulate trading actions based on recommendations.

        Iterates over the recommendations DataFrame and executes buy or sell
        transactions according to each recommendation's action. Uses the nearest
        available stock price for the given month. If price data is unavailable,
        the transaction is skipped and recorded.

        Converts 'strong buy' and 'strong sell' signals to 'buy' and 'sell', respectively.
        'Hold' actions are ignored.

        Skipped transactions due to missing price data are appended to
        `self.skipped_transactions`.

        Returns:
            None
        """
        for _, rec in tqdm(self.recommendations.iterrows(), total=len(self.recommendations), desc="Simulating Trades"):
            cik, action, date = rec['cik'], rec['action'], rec['date'] # 1 rec(ommendation) is 1 row in the recommendations df 
            try:
                price = self.get_nearest_price(cik, date)  
                # Value error is "grabbed" from get_nearest_price function
            except ValueError as e:
                #print(f"Error fetching price for {cik} on {date}: {e}")
                self.skipped_transactions.append((cik, date, str(e)))
                continue # continue to next recommendation if no price is available, no action will be taken
            # Simple if statement to determine action
            # For now, I ignore "Strong" signals and 
            if action == "strong buy":
                action = 'buy'
            elif action == "strong sell": # for now, this simply elif is used to convert varying signals to "buy" or "sell"
                action = 'sell'
            if action == 'buy':
                self.buy(cik, price, date)
                #print(f"Bought {cik} at {price} on {date}")
            elif action == 'sell':
                self.sell(cik, price, date)
            # Hold can simply be ignored


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
        for action, cik, price, tx_date in self.transactions:
            # Only consider transactions up to the specified date
            if tx_date <= date:
                if action == 'buy':
                    positions_snapshot[cik] = positions_snapshot.get(cik, 0) + 1
                    positions_snapshot['cash'] -= price
                elif action == 'sell':
                    positions_snapshot[cik] = positions_snapshot.get(cik, 0) - 1
                    positions_snapshot['cash'] += price

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
    def get_portfolio_value(self, date=None):
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
        
        # Calculate geometric mean return (better for compounding effects)
        geometric_mean_return = (1 + monthly_returns['return']).prod() ** (1 / len(monthly_returns)) - 1
        
        # Calculate standard deviation of returns
        std_return = monthly_returns['return'].std()
        
        # Calculate annualized return and standard deviation
        annualized_return = (1 + mean_return) ** 12 - 1
        annualized_std = std_return * np.sqrt(12)
        
        pf_statistics = {
            "Mean return (monthly)": np.round(mean_return,6),
            "Geometric mean return (monthly)": np.round(geometric_mean_return,6),
            "Standard deviation (monthly)": np.round(std_return,6),
            "Annualized mean return": np.round(annualized_return,6),
            "Annualized standard deviation": np.round(annualized_std,6),
            "Number of buys": self.no_buys,
            "Number of sells": self.no_sells,
            "Total number of transactions": self.no_transactions,
            "Number of skipped buys": self.no_skipped_buys,
            "Number of skipped sells": self.no_skipped_sells,
            "Total number of skipped transactions": self.no_skipped_transactions,
        }
        
        return pf_statistics
