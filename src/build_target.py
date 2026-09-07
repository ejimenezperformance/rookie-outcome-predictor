import pandas as pd
import numpy as np

D = "/home/claude/data/lahman_1871-2025_csv/"
batting = pd.read_csv(D+"Batting.csv")
allstar = pd.read_csv(D+"AllstarFull.csv")

rookies = pd.read_csv("rookie_seasons.csv")

# Only keep rookies with at least 5 full following seasons of data available (debut_year + 5 <= 2025)
rookies = rookies[rookies.rookie_year <= 2020].copy()
print(f"Rookies with >=5 following seasons of data available: {len(rookies)}")

agg_cols = ["G","AB","BB","HBP","SF","SH"]
season = batting.groupby(["playerID","yearID"])[agg_cols].sum().reset_index()

allstar_players = set(allstar.playerID.unique())

def compute_target(row):
    pid, ry = row.playerID, row.rookie_year
    window = season[(season.playerID==pid) & (season.yearID > ry) & (season.yearID <= ry+5)]
    total_games = window.G.sum()
    made_allstar = pid in allstar_players and not allstar[(allstar.playerID==pid) & (allstar.yearID>ry) & (allstar.yearID<=ry+5)].empty
    if made_allstar:
        return "Star"
    elif total_games >= 300:
        return "Regular"
    else:
        return "Bust"

rookies["outcome"] = rookies.apply(compute_target, axis=1)
print(rookies.outcome.value_counts())
print(rookies.outcome.value_counts(normalize=True).round(3))

rookies.to_csv("rookie_labeled.csv", index=False)
print("Saved rookie_labeled.csv")
