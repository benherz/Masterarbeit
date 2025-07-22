import pandas as pd
from tqdm import tqdm
import numpy as np


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
        

    ##### Function to load input stock prices
    def load_stock_prices(self, stock_prices_df):
        """Load stock prices DataFrame with dates as monthly periods."""
        df = stock_prices_df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.to_period('M')
        self.stock_prices = df

    ##### Function to load input recommendations
    def load_recommendations(self, recommendations_df):
        """Load recommendations DataFrame with dates as monthly periods."""
        df = recommendations_df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.to_period('M')
        self.recommendations = df


    ##### LLM, as well as analyst recommendations do not necessarily align with the monthly stock prices on a day-level.
    # Therefore, a function to grab the nearest stock price for a given recommendation date is implemented.
    def get_nearest_price(self, cik, date):
        # Convert input date to Period (monthly)
        if not isinstance(date, pd.Period):
            date = pd.to_datetime(date).to_period('M')
        """Get the nearest stock price for a given cik on a specific month (period)."""
        stock_on_date = self.stock_prices[self.stock_prices['cik'] == cik].copy()
        stock_on_date = stock_on_date.sort_values(by='date')

        # Simply return price of that month
        if date in stock_on_date['date'].values:
            return stock_on_date[stock_on_date['date'] == date]['price'].values[0]
        else:
            raise ValueError("No price data available.")

    ##### Function to simulate a stock purchase
    # For simplicity, I always just buy or sell 1 stock at a given point in time (for now).
    def buy(self, cik, price, date):
        # Only buy if capital is sufficient
        if price > self.cash:
            #print(f"Skipped BUY on {date}: Not enough capital.")
            return
        # New cash balance is simply old one, minus the price of the stock
        self.cash -= price
        # To keep track of the number of positions, the dictionary is updated
        self.positions[cik] = self.positions.get(cik, 0) + 1 #. get() returns 0 if cik is not in positions, otherwise returns the current value and then we add 1
        # Append the transaction to the list to later on check positions over time
        self.transactions.append(('buy', cik, price, date))

    ##### Function to simulate a stock sale
    def sell(self, cik, price, date):
        if self.positions.get(cik, 0) == 0: # 0 inside the brackets is simply the "default" to return, if cik is not in positions
            return
        # As above, new cash balance is simply old one, plus the price of the stock
        self.cash += price
        self.positions[cik] -= 1
        self.transactions.append(('sell', cik, price, date))
        print(f"Sold {cik} at {price} on {date}")
        


    ##### Function that puts it all together
    def simulate_trading(self):
        """Simulate trading based on recommendations."""
        for _, rec in tqdm(self.recommendations.iterrows(), total=len(self.recommendations), desc="Simulating Trades"):
            cik, action, date = rec['cik'], rec['action'], rec['date']
            try:
                price = self.get_nearest_price(cik, date)  # Use nearest price
                # Value error is "grabbed" from get_nearest_price function
            except ValueError as e:
                print(f"Error fetching price for {cik} on {date}: {e}")
                continue # continue to next recommendation if no price is available, no action will be taken
            # Simple if statement to determine action
            # For now, I ignore "Strong" signals and 
            if action == "strong buy":
                action = 'buy'
            elif action == "strong sell":
                action = 'sell'
            if action == 'buy':
                self.buy(cik, price, date)
                print(f"Bought {cik} at {price} on {date}")
            elif action == 'sell':
                self.sell(cik, price, date)
            # Hold can simply be ignored

######################################################################################################################################################################
    # Function to get the monetary value of the portfolio at a specific date
    def get_portfolio_value(self, date=None):
        # Convert input date to Period (monthly)
        if not isinstance(date, pd.Period):
            date = pd.to_datetime(date).to_period('M')
        positions_df = self.get_positions_at_date(date)
        
        total_value = 0
        for _, row in positions_df.iterrows():
            cik = row['cik']
            qty = row['position']
            value = row['value']
            if cik == 'cash':
                total_value += qty  # cash amount
            else:
                total_value += qty * value
        return np.round(total_value, 4)



    ###### Function to get positions at a specific date
    def get_positions_at_date(self, date):
        
        # Convert input date to Period (monthly)
        if not isinstance(date, pd.Period):
            date = pd.to_datetime(date).to_period('M')

        # Initialize positions with cash
        positions = {'cash': self.initial_capital}  # cash starts at initial capital

        # Loop through transactions and update positions as transactions were made
        for action, cik, price, tx_date in self.transactions:
            # Only consider transactions up to the specified date
            if tx_date <= date:
                if action == 'buy':
                    positions[cik] = positions.get(cik, 0) + 1
                    positions['cash'] -= price
                elif action == 'sell':
                    positions[cik] = positions.get(cik, 0) - 1
                    positions['cash'] += price

        # Filter out zero or negative stock positions but keep cash always
        positions = {k: v for k, v in positions.items() if (k == 'cash' or v > 0)}
        # Empty list to store positions
        positions_list = []
        for cik, qty in positions.items():
            if cik == 'cash':
                val = qty
            else:
                price = self.get_nearest_price(cik, date)
                val = price * qty
            positions_list.append((date, cik, qty, val))
        # Convert to df 
        positions_df = pd.DataFrame(
            positions_list,
            columns=['date', 'cik', 'position', 'value']
        )
        return positions_df.set_index('date')


#### testing function for all positions df
    def get_positions_over_time(self):
        """Returns a DataFrame of positions at each unique transaction date."""
        unique_dates = sorted(set(pd.to_datetime(tx[3]) for tx in self.transactions))
        all_positions = []

        for date in unique_dates:
            positions_df = self.get_positions_at_date(date)
            all_positions.append(positions_df)

        # Combine all the DataFrames vertically
        combined_df = pd.concat(all_positions)
        # Fill missing values with 0 and convert to int
        combined_df = combined_df.fillna(0)
        return combined_df


    ##### Function to calculate monthly returns
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

        return pd.DataFrame(monthly_returns)
