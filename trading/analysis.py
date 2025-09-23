import matplotlib.pyplot as plt

times = []
estimates = []
home_scores = []
away_scores = []

with open("output.txt", "r") as f:   # replace with your file path
    for line in f:
        parts = line.strip().split("::")
        if len(parts) >= 4:
            time = float(parts[0])          # first field = time (in seconds)
            estimate = float(parts[-1])     # last field = estimate
            score_part = parts[1].split()   # e.g., "SCORE 0 - 2"

            # extract home and away scores
            try:
                home = int(score_part[1])   # "0"
                away = int(score_part[3])   # "2"
            except (IndexError, ValueError):
                home, away = None, None

            times.append(time)
            estimates.append(estimate)
            home_scores.append(home)
            away_scores.append(away)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(times, estimates, marker="o", label="Estimate")
plt.plot(times, home_scores, marker="s", label="Home Score")
plt.plot(times, away_scores, marker="^", label="Away Score")

plt.xlabel("Game Time (seconds remaining)")
plt.ylabel("Value")
plt.title("Estimate and Scores over Game Time")
plt.gca().invert_xaxis()  # countdown timer: 2880 -> 0
plt.grid(True)
plt.legend()
plt.show()