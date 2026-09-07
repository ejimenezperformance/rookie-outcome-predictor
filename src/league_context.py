import pandas as pd

D = "/home/claude/data/lahman_1871-2025_csv/"
batting = pd.read_csv(D+"Batting.csv")

# League-average OPS per year, computed from ALL qualified-ish batters that year (AB>=50 to filter pitchers/cup-of-coffee noise)
season = batting.groupby(["playerID","yearID"])[["AB","H","2B","3B","HR","BB","HBP","SF"]].sum().reset_index()
season = season[season.AB >= 50].copy()

def derive_ops(row):
    ab,h,bb,hbp,sf = row.AB, row.H, row.BB, row.HBP, row.SF
    doubles,triples,hr = row["2B"],row["3B"],row.HR
    obp_den = ab+bb+hbp+sf
    obp = (h+bb+hbp)/obp_den if obp_den else 0
    singles = h-doubles-triples-hr
    slg = (singles+2*doubles+3*triples+4*hr)/ab if ab else 0
    return obp+slg

season["OPS"] = season.apply(derive_ops, axis=1)
league_avg = season.groupby("yearID")["OPS"].mean().rename("league_avg_OPS")
league_avg.to_csv("league_avg_ops.csv")
print(league_avg.tail(10))
