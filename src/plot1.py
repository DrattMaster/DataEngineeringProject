import matplotlib.pyplot as plt
import numpy as np

# raw data (workers, total cores, list we ignore, time)
data = [3,6,"[2, 2, 2]",435.31854939460754,
3,6,"[2, 2, 2]",541.0400216579437,
3,6,"[2, 2, 2]",690.071722984314,
2,4,"[2, 2]",812.1448354721069,
2,4,"[2, 2]",645.0954341888428,
2,4,"[2, 2]",767.9313116073608,
1,2,[2],1204.7523562908173,
1,2,[2],1364.8249099254608,
1,2,[2],1943.2460751533508,
3,3,"[1, 1, 1]",1282.6301534175873,
3,3,"[1, 1, 1]",974.2141485214233,
3,3,"[1, 1, 1]",773.2866337299347,
2,2,"[1, 1]",1142.8959515094757,
2,2,"[1, 1]",1329.1595809459686,
2,2,"[1, 1]",1186.2068555355072,
1,1,[1],1888.5314846038818,
1,1,[1],2468.380226612091,
1,1,[1],2590.826119184494]

replication_of_experiment = 3  # we ran each setup 3 times

# clean up data -> keep only workers, cores and time
cleaned_data = []
for i in range(0, len(data), 4):
    workers = data[i]
    total_cores = data[i + 1]
    time_val = data[i + 3]
    cleaned_data.append([workers, total_cores, time_val])

# calculate arg time for each setup
avg_points = []
for i in range(0, len(cleaned_data), replication_of_experiment):
    chunk = cleaned_data[i:i + replication_of_experiment]

    workers = chunk[0][0]
    total_cores = chunk[0][1]
    cpw = total_cores // workers  # cores per worker
    avg_time = np.mean([row[2] for row in chunk])  # avg of the 3 runs

    avg_points.append([workers, cpw, avg_time])

# split into 2 groups (1 core vs 2 cores per worker)
x1, y1 = [], []  # 1 core per worker
x2, y2 = [], []  # 2 cores per worker

for workers, cpw, avg_time in avg_points:
    if cpw == 1:
        x1.append(workers)
        y1.append(avg_time)
    elif cpw == 2:
        x2.append(workers)
        y2.append(avg_time)

# sort so lines dont look weird
x1, y1 = zip(*sorted(zip(x1, y1)))
x2, y2 = zip(*sorted(zip(x2, y2)))

x1 = np.array(x1)
y1 = np.array(y1)
x2 = np.array(x2)
y2 = np.array(y2)

# get slope in log space (since we use log axis)
m1, b1 = np.polyfit(x1, np.log10(y1), 1)
m2, b2 = np.polyfit(x2, np.log10(y2), 1)

print(f"slope 1 core/worker (log10): {m1:.4f}")
print(f"slope 2 cores/worker (log10): {m2:.4f}")

# make smooth lines for the trend
x1_line = np.linspace(x1.min(), x1.max(), 100)
y1_line = 10 ** (m1 * x1_line + b1)
x2_line = np.linspace(x2.min(), x2.max(), 100)
y2_line = 10 ** (m2 * x2_line + b2)

# plotting
plt.figure(figsize=(9, 6))
plt.scatter(x1, y1, s=100, label="1 core per worker")
plt.scatter(x2, y2, s=100, label="2 cores per worker")
plt.plot(x1_line, y1_line, linestyle=":", linewidth=2, label=f"trend 1 core (slope={m1:.3f})")
plt.plot(x2_line, y2_line, linestyle=":", linewidth=2, label=f"trend 2 cores (slope={m2:.3f})")

# print values next to the dots
for x, y in zip(x1, y1):
    plt.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

for x, y in zip(x2, y2):
    plt.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)

plt.yscale("log") 
plt.xticks([1, 2, 3])
plt.xlabel("Number of workers")
plt.ylabel("Time (s, log scale)")
plt.title("Average Execution Time by Workers")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()