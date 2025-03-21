#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 14:52:19 2020

Module for measuring player velocities, smoothed using a Savitzky-Golay filter, with Metrica tracking data.

Data can be found at: https://github.com/metrica-sports/sample-data

@author: Laurie Shaw (@EightyFivePoint)

"""

import numpy as np
import pandas as pd
import scipy.signal as signal


def calc_player_velocities(
    team, smoothing=True, filter_="Savitzky-Golay", window=7, polyorder=1, maxspeed=12
):
    """
    Calculate player velocities in x & y directions, and total speed at each timestamp.

    Parameters
    ----------
    team : DataFrame
        Tracking data for home or away team.
    smoothing : bool, optional
        Whether to smooth velocity data (default is True).
    filter_ : str, optional
        Type of filter for smoothing ("Savitzky-Golay" or "moving average").
    window : int, optional
        Smoothing window size in number of frames.
    polyorder : int, optional
        Polynomial order for Savitzky-Golay filter.
    maxspeed : float, optional
        Maximum realistic speed (m/s) to filter out outliers.

    Returns
    -------
    DataFrame
        Tracking data with added velocity columns.
    """

    # Remove any existing velocity data
    team = remove_player_velocities(team)

    # Get the player IDs
    player_ids = np.unique([c[:-2] for c in team.columns if c[:4] in ["Home", "Away"]])

    # Calculate time step (dt) and handle NaN values
    dt = team["Time [s]"].diff().fillna(method="bfill")  # Fill first row

    # Iterate through players
    for player in player_ids:
        vx = team[player + "_x"].diff() / dt
        vy = team[player + "_y"].diff() / dt

        # Handle unrealistic speed values (outliers)
        raw_speed = np.sqrt(vx**2 + vy**2)
        vx[raw_speed > maxspeed] = np.nan
        vy[raw_speed > maxspeed] = np.nan

        # Smoothing
        if smoothing:
            try:
                if filter_ == "Savitzky-Golay" and not vx.isna().all():
                    valid_points = vx.dropna().size
                    adjusted_window = min(
                        window,
                        valid_points - 1 if valid_points % 2 == 0 else valid_points,
                    )
                    if adjusted_window >= 3:
                        vx = signal.savgol_filter(
                            vx.fillna(0),
                            window_length=adjusted_window,
                            polyorder=polyorder,
                        )
                        vy = signal.savgol_filter(
                            vy.fillna(0),
                            window_length=adjusted_window,
                            polyorder=polyorder,
                        )
                    else:
                        raise ValueError(
                            "Insufficient valid data points for Savitzky-Golay"
                        )
                elif filter_ == "moving average":
                    vx = vx.rolling(window=window, min_periods=1).mean()
                    vy = vy.rolling(window=window, min_periods=1).mean()
                elif filter_ == "linear interpolation":
                    vx = vx.interpolate(method="linear")
                    vy = vy.interpolate(method="linear")
            except Exception as e:
                print(
                    f"Fallback to linear interpolation for player {player} due to: {e}"
                )
                vx = vx.interpolate(method="linear")
                vy = vy.interpolate(method="linear")

        # Store velocity data in the DataFrame
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt(vx**2 + vy**2)

    return team

def remove_player_velocities(team):
    """
    Remove existing velocity and acceleration columns from the tracking DataFrame.
    """
    columns_to_remove = [
        c
        for c in team.columns
        if c.split("_")[-1] in ["vx", "vy", "ax", "ay", "speed", "acceleration"]
    ]
    return team.drop(
        columns=columns_to_remove, errors="ignore"
    )  # Ignore if columns are missing
