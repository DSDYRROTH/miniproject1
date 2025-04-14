import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from tkcalendar import Calendar


data = {
    'sales': [100, 150, 130, 170, 160, 180, 190, 200, 210, 220],
}
date_range = pd.date_range(start='2025-01-01', periods=10, freq='D')
df = pd.DataFrame(data, index=date_range)


start_of_year = '2025-01-01'
end_of_year = '2025-12-31'
print(f"Available date range: {start_of_year} to {end_of_year}")


def is_valid_date(start_date_str, end_date_str):
    try:
        
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        
        if start_date >= end_date:
            print("Start date must be earlier than the end date.")
            return False
        return True
    except ValueError:
        print("Invalid date format. Please use 'YYYY-MM-DD'.")
        return False


def get_date_range():
    start_date = start_calendar.get_date()
    end_date = end_calendar.get_date()

    
    if is_valid_date(start_date, end_date):
        return start_date, end_date
    else:
        messagebox.showerror("Invalid Date Range", "Start date must be earlier than the end date.")
        return None, None


def submit():
    start_date, end_date = get_date_range()
    if not start_date or not end_date:
        return  

    
    df_filtered = df[start_date:end_date]

    # Check if there are enough data points
    if len(df_filtered) < 3:
        messagebox.showerror("Not Enough Data", f"Not enough data points in the selected date range. You need at least 3 data points. Current data points: {len(df_filtered)}")
        return

    
    model = ARIMA(df_filtered['sales'], order=(1, 1, 1))  
    model_fit = model.fit()

    
    forecast = model_fit.forecast(steps=5)

    print(f"Forecasted sales for the next 5 days: {forecast}")

    
    plt.figure(figsize=(10, 6))
    plt.plot(df_filtered.index, df_filtered['sales'], label='Actual Sales', color='blue')
    plt.plot(pd.date_range(df_filtered.index[-1], periods=6, freq='D')[1:], forecast, label='Forecasted Sales', color='red', linestyle='dashed')
    plt.title("Sales Forecasting with ARIMA")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()


root = tk.Tk()
root.title("Sales Forecasting with ARIMA")


label1 = tk.Label(root, text="Select Start Date", font=("Arial", 12))
label1.pack(pady=10)


start_calendar = Calendar(root, selectmode='day', date_pattern='yyyy-mm-dd')
start_calendar.pack(pady=10)

label2 = tk.Label(root, text="Select End Date", font=("Arial", 12))
label2.pack(pady=10)


end_calendar = Calendar(root, selectmode='day', date_pattern='yyyy-mm-dd')
end_calendar.pack(pady=10)


submit_button = tk.Button(root, text="Submit", command=submit, font=("Arial", 12))
submit_button.pack(pady=20)


root.mainloop()