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

    # Ensure missing positions are handled to avoid errors in velocity calculations
    team.fillna(method="ffill", inplace=True)  # Forward fill missing positions

    # Remove any existing velocity columns
    team = remove_player_velocities(team)

    # Get the player IDs based on column names
    player_ids = np.unique([c[:-2] for c in team.columns if c[:4] in ["Home", "Away"]])

    # Compute time step (dt) and handle NaN values
    dt = team["Time [s]"].diff()
    dt.fillna(method="bfill", inplace=True)  # Fill first row

    # Get the index of the first frame in the second half
    second_half_idx = team[team["Period"] == 2].index.min()
    if pd.isna(second_half_idx):
        second_half_idx = len(team)  # Assume second half starts at the end if not found

    # Compute velocities for each player
    for player in player_ids:
        # Compute velocity using position differences
        vx = team[player + "_x"].diff() / dt
        vy = team[player + "_y"].diff() / dt

        # Handle unrealistic speed values
        raw_speed = np.sqrt(vx**2 + vy**2)
        vx[raw_speed > maxspeed] = np.nan
        vy[raw_speed > maxspeed] = np.nan

        # Apply smoothing if needed
        if smoothing and not vx.isna().all():  # Ensure there's data to smooth
            if filter_ == "Savitzky-Golay":
                vx.iloc[:second_half_idx] = signal.savgol_filter(
                    vx.iloc[:second_half_idx].to_numpy(),
                    window_length=window,
                    polyorder=polyorder,
                )
                vy.iloc[:second_half_idx] = signal.savgol_filter(
                    vy.iloc[:second_half_idx].to_numpy(),
                    window_length=window,
                    polyorder=polyorder,
                )
                vx.iloc[second_half_idx:] = signal.savgol_filter(
                    vx.iloc[second_half_idx:].to_numpy(),
                    window_length=window,
                    polyorder=polyorder,
                )
                vy.iloc[second_half_idx:] = signal.savgol_filter(
                    vy.iloc[second_half_idx:].to_numpy(),
                    window_length=window,
                    polyorder=polyorder,
                )
            elif filter_ == "moving average":
                ma_window = np.ones(window) / window
                vx.iloc[:second_half_idx] = np.convolve(
                    vx.iloc[:second_half_idx], ma_window, mode="same"
                )
                vy.iloc[:second_half_idx] = np.convolve(
                    vy.iloc[:second_half_idx], ma_window, mode="same"
                )
                vx.iloc[second_half_idx:] = np.convolve(
                    vx.iloc[second_half_idx:], ma_window, mode="same"
                )
                vy.iloc[second_half_idx:] = np.convolve(
                    vy.iloc[second_half_idx:], ma_window, mode="same"
                )

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
