import pandas as pd
import numpy as np

D = "/home/claude/data/lahman_1871-2025_csv/"
batting = pd.read_csv(D+"Batting.csv")
people = pd.read_csv(D+"People.csv")
allstar = pd.read_csv(D+"AllstarFull.csv")
appearances = pd.read_csv(D+"Appearances.csv")

# Aggregate multi-stint seasons (traded mid-year) into one row per player-year
agg_cols = ["G","AB","R","H","2B","3B","HR","RBI","SB","CS","BB","SO","IBB","HBP","SH","SF","GIDP"]
season = batting.groupby(["playerID","yearID"])[agg_cols].sum().reset_index()

# PA approx
season["PA"] = season.AB + season.BB.fillna(0) + season.HBP.fillna(0) + season.SF.fillna(0) + season.SH.fillna(0)

# Identify each player's first qualifying season (rookie season): PA >= 200
season = season.sort_values(["playerID","yearID"])
first_year = season[season.PA >= 200].groupby("playerID")["yearID"].min().rename("rookie_year")

df = first_year.reset_index()
print(f"Players with a qualifying rookie season (PA>=200): {len(df)}")

# Merge rookie-season stats
rookie_stats = season.merge(df, on="playerID")
rookie_stats = rookie_stats[rookie_stats.yearID == rookie_stats.rookie_year].copy()

# Derived rate stats for rookie season
def derive(row):
    ab, h, bb = row.AB, row.H, row.BB
    hbp, sf = row.HBP, row.SF
    doubles, triples, hr = row["2B"], row["3B"], row.HR
    avg = h/ab if ab else 0
    obp_den = ab+bb+hbp+sf
    obp = (h+bb+hbp)/obp_den if obp_den else 0
    singles = h - doubles - triples - hr
    slg = (singles+2*doubles+3*triples+4*hr)/ab if ab else 0
    return pd.Series({"AVG":avg,"OBP":obp,"SLG":slg,"OPS":obp+slg,"ISO":slg-avg,
                       "BB_pct": bb/obp_den if obp_den else 0, "K_pct": row.SO/obp_den if obp_den else 0})

rookie_stats = pd.concat([rookie_stats, rookie_stats.apply(derive, axis=1)], axis=1)

# Age at rookie season
rookie_stats = rookie_stats.merge(people[["playerID","birthYear","nameFirst","nameLast"]], on="playerID")
rookie_stats["age"] = rookie_stats.yearID - rookie_stats.birthYear

# Primary position during rookie year (from Appearances)
pos_cols = ["G_p","G_c","G_1b","G_2b","G_3b","G_ss","G_of","G_dh"]
app_year = appearances.groupby(["playerID","yearID"])[pos_cols].sum().reset_index()
app_year["primary_pos"] = app_year[pos_cols].idxmax(axis=1)
rookie_stats = rookie_stats.merge(app_year[["playerID","yearID","primary_pos"]], on=["playerID","yearID"], how="left")

print(rookie_stats[["playerID","nameFirst","nameLast","rookie_year","age","PA","OPS","primary_pos"]].head(10))
rookie_stats.to_csv("rookie_seasons.csv", index=False)
print("Saved rookie_seasons.csv, shape:", rookie_stats.shape)
