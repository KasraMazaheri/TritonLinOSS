import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Parse the data
data = defaultdict(lambda: {"L": [], "time": []})

with open("out.txt", "r") as f:
    for line in f:
        line = line.strip()
        if "parallel scan" in line.lower():
            # Extract L, P, and time
            l_match = re.search(r"L=(\d+)", line)
            p_match = re.search(r"P=(\d+)", line)
            time_match = re.search(r"took ([\d.]+) seconds", line)

            if l_match and p_match and time_match:
                L = int(l_match.group(1))
                P = int(p_match.group(1))
                time = float(time_match.group(1))

                # Determine implementation type
                if "JAX" in line:
                    key = f"JAX (P={P})"
                else:
                    key = f"Parallel (P={P})"

                data[key]["L"].append(L)
                data[key]["time"].append(time)

cm = plt.get_cmap("tab10")

p_to_color = {}
for key in data.keys():
    P = int(key.split("=")[1].replace(")", ""))
    if P not in p_to_color:
        p_to_color[P] = cm(len(p_to_color))

# Plot all P values using same color for Parallel scan/JAX if P is the same
for key in sorted(data.keys()):
    P = int(key.split("=")[1].replace(")", ""))
    color = p_to_color[P]
    if "Parallel" in key and "JAX" not in key:
        plt.plot(data[key]["L"], data[key]["time"], marker="o", label=key, color=color)
    elif "JAX" in key:
        plt.plot(data[key]["L"], data[key]["time"], marker="s", label=key, color=color)

plt.xlabel("Sequence Length (L)")
plt.ylabel("Time (seconds)")
plt.title("Parallel Scan Performance")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("timing_comparison.png", dpi=300, bbox_inches="tight")
print("Plot saved to timing_comparison.png")
plt.show()
