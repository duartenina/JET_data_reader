from __future__ import division, print_function, absolute_import
import os
import argparse
import re


pcies = {
    'continuous': {
        3: [1, 2, 3, 4],
        15: [5, 6, 7, 8],
        4: [9, 10, 11, 12],
        16: [13, 14, 15, 16],
        6: [17, 18, 19, None],
        18: [None, None, None, None]
    },
    'distributed': {
        3: [1, 4, 8, None],
        15: [11, 15, 19, None],
        4: [2, 5, 9, None],
        16: [12, 16, None, 18],
        6: [3, 6, 7, 10],
        18: [13, 14, None, 17]
    }
}

orders = {'c': 'continuous', 'd': 'distributed'}

parser = argparse.ArgumentParser()

parser.add_argument(
    '--example-file', '-f', dest='example_file',
    type=str, required=True,
    help=""
)

parser.add_argument(
    '--shot-number', '-s', dest='shot',
    type=str, required=True,
    help=""
)

parser.add_argument(
    '--ordering', '-o', '--order', dest='order',
    type=str, default='c', choices=('c', 'd'),
    help=""
)

args = parser.parse_args()

file_path = args.example_file
order = orders[args.order]

folder_path = os.path.dirname(file_path)
filename = os.path.basename(file_path)

_, _, files = next(os.walk(folder_path))

pattern = re.sub(r'pcie[\d]{1,2}(.*)_[1-4]\.bin',
                 r'pcie([\d]{1,2})\1_([1-4])\.bin',
                 filename)

for old_file_name in files:
    match_obj = re.match(pattern, old_file_name)
    if match_obj is None:
        continue

    pcie_n, ch_n = [int(group) for group in match_obj.groups()]

    channel_num = pcies[order][pcie_n][ch_n-1]

    if channel_num is None:
        print(old_file_name, ' non existent')
        continue

    new_file_name = 'shot_%s_ch%d.bin' % (args.shot, channel_num)
    print('%-40s -> %-40s' % (old_file_name, new_file_name))

    os.rename(
        os.path.join(folder_path, old_file_name),
        os.path.join(folder_path, new_file_name)
    )
