import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("btc_data.csv")

# Convert time column to datetime
df["time"] = pd.to_datetime(df["time"])

# Filter data from 2020 onward
df_filtered = df[df["time"] >= "2020-01-01"]

# Select the two columns and plot them
plt.figure(figsize=(12, 8))
plt.plot(
    df_filtered["time"], df_filtered["CapMVRVCur"], label="CapMVRVCur", linewidth=1
)
plt.plot(df_filtered["time"], df_filtered["CapMVRVFF"], label="CapMVRVFF", linewidth=1)

plt.title("Bitcoin MVRV Time Series (2020 onward)")
plt.xlabel("Time")
plt.ylabel("MVRV Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Also print some basic info about the columns for the filtered data
print("CapMVRVCur stats (2020 onward):")
print(df_filtered["CapMVRVCur"].describe())
print("\nCapMVRVFF stats (2020 onward):")
print(df_filtered["CapMVRVFF"].describe())
