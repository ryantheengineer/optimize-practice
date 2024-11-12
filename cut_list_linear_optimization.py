# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:53:25 2024

@author: Ryan.Larson

Takes in a CSV file with columns BoardLength and CutLength, with only one type
of length per row.
"""

import numpy as np
import pandas as pd

# Load data from CSV (adjust your file path as needed)
data = pd.read_csv("board_cut_lengths.csv")

# Set initial parameters
board_length = 182.0
cutter_length = 0.125
cut_lengths = data['CutLength'].dropna().tolist()

# Sort cuts in descending order (first fit decreasing strategy)
cut_lengths.sort(reverse=True)

# Initialize boards with remaining length equal to board length
remaining_lengths = []
cut_groups = [[]]

for i, cut_length in enumerate(cut_lengths):
    if i == 0:
        remaining_lengths.append(board_length)
    
    for j in range(len(remaining_lengths)):
        # Determine if the cut should include the cutter length
        required_length = cut_length
        if cut_length != board_length:  # Only add cutter length if it's an actual cut
            required_length += cutter_length
        
        remainder = remaining_lengths[j] - required_length
        
        if remainder >= 0:
            remaining_lengths[j] = remainder
            cut_groups[j].append(cut_length)
            break  # Exit inner loop as the cut was made successfully
        else:
            # If this is the last board in the list, add a new board
            if j == len(remaining_lengths) - 1:
                remaining_lengths.append(board_length)
                remainder = remaining_lengths[-1] - required_length
                remaining_lengths[-1] = remainder
                cut_groups.append([cut_length])
                break  # Exit inner loop as the cut was made on a new board

print('Cut Groups:')
for cut_group in cut_groups:
    print(cut_group)
print('')

# Optional: Validate that total lengths of cut groups are less than board_length
for i, cut_group in enumerate(cut_groups):
    if sum(cut_group) + cutter_length*(len(cut_group)-1) <= board_length:
        print(f'Board {i}: Valid ({sum(cut_group) + cutter_length*(len(cut_group)-1)})')
    else:
        print(f'Board {i}: Invalid ({sum(cut_group) + cutter_length*(len(cut_group)-1)})')

board_count = len(cut_groups)
sum_remainders = sum(remaining_lengths)
print(f'Final board count: {board_count}')
print(f'Sum of remainders: {sum_remainders}')
