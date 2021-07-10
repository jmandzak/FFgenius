import pandas as pd

# dict for translating full team name to abbr.

abbr = {
    'San Francisco 49ers' : 'SF',
    'Pittsburgh Steelers' : 'PIT',
    'Baltimore Ravens' : 'BAL',
    'Buffalo Bills' : 'BUF',
    'New England Patriots' : 'NE',
    'Los Angeles Rams' : 'LAR',
    'Chicago Bears' : 'CHI',
    'Kansas City Chiefs' : 'KC',
    'Minnesota Vikings' : 'MIN',
    'New Orleans Saints' : 'NO',
    'Los Angeles Chargers' : 'LAC',
    'Denver Broncos' : 'DEN',
    'New York Jets' : 'NYJ',
    'Philadelphia Eagles' : 'PHI',
    'Seattle Seahawks' : 'SEA',
    'Tennessee Titans' : 'TEN',
    'Green Bay Packers' : 'GB',
    'Dallas Cowboys' : 'DAL',
    'Indianapolis Colts' : 'IND',
    'Tampa Bay Buccaneers' : 'TB',
    'Cleveland Browns' : 'CLE',
    'Houston Texans' : 'HOU',
    'Jacksonville Jaguars' : 'JAC',
    'Atlanta Falcons' : 'ATL',
    'Washington Football Team' : 'WAS',
    'Carolina Panthers' : 'CAR',
    'Detroit Lions' : 'DET',
    'Las Vegas Raiders' : 'LV',
    'New York Giants' : 'NYG',
    'Miami Dolphins' : 'MIA',
    'Arizona Cardinals' : 'ARI',
    'Cincinnati Bengals' : 'CIN'
}

# create all the initial dataframes
final_QBs = pd.read_csv("stats/FinalQBs.csv", thousands=',')
final_RBs = pd.read_csv("stats/FinalRBs.csv", thousands=',')
final_WRs = pd.read_csv("stats/FinalWRs.csv", thousands=',')
final_TEs = pd.read_csv("stats/FinalTEs.csv", thousands=',')
final_DEFs = pd.read_csv("stats/FinalDEFs.csv", thousands=',')
final_Ks = pd.read_csv("stats/FinalKs.csv", thousands=',')

QBs = pd.read_csv("stats/QBs.csv", thousands=',')
RBs = pd.read_csv("stats/RBs.csv", thousands=',')
WRs = pd.read_csv("stats/WRs.csv", thousands=',')
TEs = pd.read_csv("stats/TEs.csv", thousands=',')
DEFs = pd.read_csv("stats/DEFs.csv", thousands=',')
Ks = pd.read_csv("stats/Ks.csv", thousands=',')

all_final_dfs_except_defs = [final_QBs, final_RBs, final_WRs, final_TEs, final_Ks]
all_dfs_except_defs = [QBs, RBs, WRs, TEs, Ks]

# fix names of players
for df in all_final_dfs_except_defs:
    for i, row in df.iterrows():
        temp_name = row["name"].split()
        temp_name = " ".join(temp_name[0:2])
        temp_name.replace(".", "")
        df.at[i, "name"] = temp_name

# translate defenses to abbreviations
for i, row in final_DEFs.iterrows():
    abbreviation = abbr[row["name"]]
    final_DEFs.at[i, "name"] = abbreviation

Ks.drop('team', inplace=True, axis=1)
for i, row in Ks.iterrows():
    Ks.at[i, "FGpercent"] = row["FGpercent"].strip('%')

qbDF = final_QBs.merge(QBs, how="inner", on="name")
rbDF = final_RBs.merge(RBs, how="inner", on="name")
wrDF = final_WRs.merge(WRs, how="inner", on="name")
teDF = final_TEs.merge(TEs, how="inner", on="name")
defDF = final_DEFs.merge(DEFs, how="inner", on="name")
kDF = final_Ks.merge(Ks, how="inner", on="name")

out_list = [qbDF, rbDF, wrDF, teDF, defDF, kDF]
out_files = ["stats/filteredQBs.csv", "stats/filteredRBs.csv", "stats/filteredWRs.csv", "stats/filteredTEs.csv", "stats/filteredDEFs.csv", "stats/filteredKs.csv", ]


# this is where all the filtering is done
for df, file in zip(out_list, out_files):
    # edit the number of games filter here
    filtered_df = df[df['Games'] >= 13]

    filtered_df.dropna(inplace=True)

    # redundant
    filtered_df.drop('proTeam', inplace=True, axis=1)
    filtered_df.drop('position', inplace=True, axis=1)

    # use this line to drop the composite
    filtered_df.drop('composite', inplace=True, axis=1)

    # drop Games ( how many games they played in the new season )
    filtered_df.drop('Games', inplace=True, axis=1)

    # use this line to drop rookies
    filtered_df = filtered_df[filtered_df['games'] >= 1]

    # 3 possible dependent variables to use
    # Points, Avg, Rank
    # only leave one commented
    #filtered_df.drop('Avg', inplace=True, axis=1)
    filtered_df.drop('Points', inplace=True, axis=1)
    filtered_df.drop('Rank', inplace=True, axis=1)

    filtered_df.to_csv(file, index=False)
