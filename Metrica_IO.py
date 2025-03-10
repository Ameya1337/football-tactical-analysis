#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:18:49 2020

Module for reading in Metrica sample data.

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)
"""

import pandas as pd
import csv as csv
import numpy as np


def read_match_data(DATADIR, gameid):
    """
    read_match_data(DATADIR,gameid):
    read all Metrica match data (tracking data for home & away teams, and ecvent data)
    """
    tracking_home = tracking_data(DATADIR, gameid, "Home")
    tracking_away = tracking_data(DATADIR, gameid, "Away")
    events = read_event_data(DATADIR, gameid)
    return tracking_home, tracking_away, events


def read_event_data(DATADIR, game_id):
    """
    read_event_data(DATADIR,game_id):
    read Metrica event data  for game_id and return as a DataFrame
    """
    eventfile = "/Sample_Game_%d/Sample_Game_%d_RawEventsData.csv" % (
        game_id,
        game_id,
    )  # filename
    events = pd.read_csv("{}/{}".format(DATADIR, eventfile))  # read data
    return events


def tracking_data(DATADIR, game_id, teamname):
    """
    Read Metrica tracking data for game_id and return as a DataFrame.
    Adjusts for header misalignment in CSV files.
    """
    teamfile = f"/Sample_Game_{game_id}/Sample_Game_{game_id}_RawTrackingData_{teamname}_Team.csv"

    # Open the file and correctly extract headers
    with open(f"{DATADIR}/{teamfile}", "r") as csvfile:
        reader = csv.reader(csvfile)
        teamnamefull = next(reader)[3].lower()
        print(f"Reading team: {teamnamefull}")

        # Get jersey numbers and column headers
        jerseys = [x for x in next(reader) if x != ""]
        columns = next(reader)

        # Fix column names by assigning correct player positions
        for i, j in enumerate(jerseys):
            columns[i * 2 + 3] = f"{teamname}_{j}_x"
            columns[i * 2 + 4] = f"{teamname}_{j}_y"

        columns[-2], columns[-1] = "ball_x", "ball_y"

    # Read CSV, skipping only 3 rows to ensure proper alignment
    tracking = pd.read_csv(f"{DATADIR}/{teamfile}", names=columns, skiprows=3)

    # Ensure "Frame" exists and set it as index
    if "Frame" in tracking.columns:
        tracking.set_index("Frame", inplace=True)
        tracking.index = pd.to_numeric(
            tracking.index, errors="coerce"
        )  # Ensure numeric index
    else:
        print("DEBUG: 'Frame' column not found. Available columns:", tracking.columns)
        raise KeyError("Column 'Frame' is missing from tracking data.")

    # Ensure the Frame column (index) is numeric
    tracking.index = pd.to_numeric(tracking.index, errors="coerce")

    return tracking


def merge_tracking_data(home, away):
    """
    merge home & away tracking data files into single data frame
    """
    return home.drop(columns=["ball_x", "ball_y"]).merge(
        away, left_index=True, right_index=True
    )


def to_metric_coordinates(data, field_dimen=(106.0, 68.0)):
    """
    Convert positions from Metrica units to meters (with origin at centre circle)
    """
    x_columns = [c for c in data.columns if c[-1].lower() == "x"]
    y_columns = [c for c in data.columns if c[-1].lower() == "y"]
    data[x_columns] = (data[x_columns] - 0.5) * field_dimen[0]
    data[y_columns] = -1 * (data[y_columns] - 0.5) * field_dimen[1]
    """ 
    ------------ ***NOTE*** ------------
    Metrica actually define the origin at the *top*-left of the field, not the bottom-left, as discussed in the YouTube video. 
    I've changed the line above to reflect this. It was originally:
    data[y_columns] = ( data[y_columns]-0.5 ) * field_dimen[1]
    ------------ ********** ------------
    """
    return data


def to_single_playing_direction(home, away, events):
    """
    Flip coordinates in second half so that each team always shoots in the same direction through the match.
    """
    for team in [home, away, events]:
        second_half_idx = team.Period.idxmax(2)
        columns = [c for c in team.columns if c[-1].lower() in ["x", "y"]]
        team.loc[second_half_idx:, columns] *= -1
    return home, away, events


def find_playing_direction(team, teamname):
    """
    Find the direction of play for the team (based on where the goalkeepers are at kickoff). +1 is left->right and -1 is right->left
    """
    GK_column_x = teamname + "_" + find_goalkeeper(team) + "_x"
    # +ve is left->right, -ve is right->left
    return -np.sign(team.iloc[0][GK_column_x])


def find_goalkeeper(team):
    """
    Find the goalkeeper in team, identifying him/her as the player closest to goal at kick off
    """
    x_columns = [
        c for c in team.columns if c[-2:].lower() == "_x" and c[:4] in ["Home", "Away"]
    ]
    GK_col = team.iloc[0][x_columns].abs().idxmax(axis=1)
    return GK_col.split("_")[1]
