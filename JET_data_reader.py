#!/usr/bin/env python
# -*- coding: cp1252 -*-

from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib import rc


rc('font', size=17)
rc('legend', fontsize=15)
rc('xtick', labelsize=13)
rc('ytick', labelsize=13)
# rc('text', usetex=True)


def create_parser():
    def float_pair(arg):
        split_pair = [float(e) for e in arg.split(',')]
        if len(split_pair) != 2:
            raise ValueError
        return split_pair

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--shot', '-s', dest='shot',
        required=True, type=str,
        help="Shot number"
    )

    parser.add_argument(
        '--channels-ignore', '-i', dest='channels_ignore',
        type=int, default={0}, nargs='+',
        help="Channels to ignore (default none)"
    )

    parser.add_argument(
        '--time-size', '-b', dest="time_n_bits",
        default='48', type=int, choices=(32, 48),
        help="Size of the timestamp (default 48b)"
    )

    parser.add_argument(
        '--freq-clock', '-f', dest='freq_clock',
        default=200, type=float,
        help="Clock frequency in MHz (default 200 MHz)"
    )

    parser.add_argument(
        '--time-delay', dest='time_delay',
        default=30, type=float,
        help="Time after trigger (default 30s)"
    )
    parser.add_argument(
        '--time-shot', dest='time_shot',
        default=[30, 100], type=float, nargs=2,
        metavar=('MIN_TIME', 'MAX_TIME'),
        help="Time span to analyse (default 30 to 100s)"
    )
    parser.add_argument(
        '--time-step', '-T', dest='time_step',
        default=1e-3, type=float,
        help="Time step for the histogram (default 1e-3 s)"
    )
    parser.add_argument(
        '--time-events', '-t', dest='events_limits',
        default=[], type=float_pair, nargs='+',
        metavar=('MIN1,MAX1', 'MIN2,MAX2'),
        help="Time of events to study (default none)"
    )

    parser.add_argument(
        '--time-background', '-B', dest='time_background',
        default=[1, -1], type=float, nargs=2,
        metavar=('TIME_START', 'TIME_END'),
        help="Time interval to consider as background (default no background)"
    )  # TODO: Grab background from another file

    parser.add_argument(
        '--adc-limits', '-a', dest='adc_limits',
        default=[0, 2 ** 15], type=int, nargs=2, metavar=('MIN_BIN', 'MAX_BIN'),
        help="Limits of the ADC (default 0 to 2^15)"
    )

    parser.add_argument(
        '--energy-windows', '-e', dest='energy_windows',
        default=[[250, 5000]], type=float_pair, nargs='+',
        metavar=('MIN1,MAX1', 'MIN2,MAX2'),
        help="Energy windows to count, in keV (default 250,5000keV)"
    )

    parser.add_argument(
        '--folder', '-F', dest='folder_name',
        default='.', type=str,
        help='Location of the bin files (default current folder)'
    )
    parser.add_argument(
        '--prefix', '-P', dest='prefix',
        default='', type=str,
        help="Prefix to prepend to output files (default none)"
    )
    parser.add_argument(
        '--suffix', '-S', dest='suffix',
        default='', type=str,
        help="Suffix to append to output files (default none)"
    )

    parser.add_argument(
        '--calibration-file', '-c', dest='calibration_file_name',
        default='calibration.csv', type=str,
        help="Calibration file (default 'calibration.csv')"
    )

    parser.add_argument(
        '--verbose', '-v', dest='verbose',
        action='store_true',
        help=""
    )

    parser.add_argument(
        '--plot-graphs', '-p', dest='plot_graphs',
        action='store_true',
        help="Show plots (by default they are not shown)"
    )
    parser.add_argument(
        '--energy-lines', '-E', dest='energy_lines',
        default=[], type=float, nargs='+', metavar='ENERGY',
        help="Values of vertical lines in energy spectrum (default none)"
    )
    parser.add_argument(
        '--energy-plot-limits', dest='energy_plot_limits',
        default=[0, 3000], type=float, nargs=2, metavar=('MIN', 'MAX'),
        help="Limits of the energy spectrum plot (default 0 to 3000 keV)"
    )

    return parser


fix_float_pair = lambda l: [[]] if len(l) == 0 else l if len(l[0]) == 2 else [l]

parser = create_parser()
args = parser.parse_args()

shot_str = args.shot
channels_to_use = sorted(set(range(1, 20)) - set(args.channels_ignore))

time_n_bits = args.time_n_bits
freq_clock = args.freq_clock * 1e6  # convert to MHz
time_delay = args.time_delay
min_time, max_time = args.time_shot
time_step = args.time_step
time_background = args.time_background
events_limits = args.events_limits

adc_limits = args.adc_limits

energy_windows = args.energy_windows
energy_lines = args.energy_lines

verbose = args.verbose
plot_graphs = args.plot_graphs
folder_name = args.folder_name
energy_plot_limits = args.energy_plot_limits

image_folder = os.path.join(folder_name, 'images')
try:
    os.makedirs(image_folder)
except OSError:
    if not os.path.isdir(image_folder):
        raise

out_folder = os.path.join(folder_name, 'out')
try:
    os.makedirs(out_folder)
except OSError:
    if not os.path.isdir(out_folder):
        raise

file_name_w_fix = lambda name: args.prefix + name + args.suffix

calibration_file_name = args.calibration_file_name
calibration_file_path = os.path.join(folder_name, calibration_file_name)

calibrations = np.loadtxt(
    fname=calibration_file_path,
    # comments='#', delimiter='\t',
    skiprows=1
)

with open(calibration_file_path, 'r') as calib_file:
    line = calib_file.readline().strip().split('#')[0]
    adc_n_bins = int(line)

counts_channels = []

# Channels selection
for channel in channels_to_use:
    filename = "shot_%s_ch%d.bin" % (shot_str, channel)
    file_path = os.path.join(folder_name, filename)

    word_64_type = np.dtype(
        [('energy', '<u2'), ('empty', '<u2'), ('time', '<u4')]
    )
    try:
        data = np.memmap(file_path, dtype=word_64_type, mode='r')
    except IOError:
        continue

    print('Reading from "' + filename + '".')

    energies = data['energy']

    if time_n_bits == 32:
        times = data['time']
    elif time_n_bits == 48:
        # Read entire 64b
        word_all = np.memmap(file_path, dtype=np.dtype('<u8'), mode='r')

        # Remove "garbage" data (energy)
        times = (word_all & 0xffffffffffff0000) >> 4 * 4
    else:
        raise ValueError("Unknown size of timestamp: '%d'.", time_n_bits)

    # Find last real measurement (time still increasing)
    num_pulses = np.argmax(times[1:] < times[:-1])

    # Remove garbage data
    times = times[:num_pulses]
    energies = energies[:num_pulses]

    # Convert to correct units
    times = times / freq_clock + time_delay  # time in seconds

    # min_time, max_time = times.min(), times.max()

    # Filter to wanted time span
    time_inds = (times >= min_time) & (times <= max_time)
    times = times[time_inds]
    energies = energies[time_inds]

    if verbose:
        # noinspection PyStringFormat
        print(' %d pulses found. %d pulses used (%.0d%%).' % (
            num_pulses, times.shape[0], times.shape[0]/num_pulses * 100
        ))

    # Find background
    bg_inds = (times >= time_background[0]) & (times <= time_background[1])
    times_bg = times[bg_inds]
    energies_bg = energies[bg_inds]

    # Find events
    times_events = []
    energies_events = []
    for event in events_limits:
        event_inds = (times >= event[0]) & (times <= event[1])
        times_events.append(times[event_inds])
        energies_events.append(energies[event_inds])

    # Calculate Time Histogram

    time_bins = np.arange(times[0], times[-1], time_step)
    time_hist, time_bins_edges = np.histogram(times, bins=time_bins)
    time_bins_centre = (time_bins_edges[1:] + time_bins_edges[:-1])/2

    time_bg_inds = (
        (time_bins_centre >= time_background[0]) &
        (time_bins_centre <= time_background[1])
    )

    # Calculate Energy Spectrum

    energy_hist, _ = np.histogram(
        energies, bins=adc_n_bins, range=adc_limits, density=False
    )

    energy_hist_bg, _ = np.histogram(
        energies_bg, bins=adc_n_bins, range=adc_limits, density=False
    )

    calib_channel = calibrations[channel - 1]

    energy_bins_center = np.arange(1, len(energy_hist) + 1, 1)
    energy_bins_center = energy_bins_center*calib_channel[0] + calib_channel[1]
    energy_bin_size = calib_channel[0]

    # TODO: Allow choice of normalization
    # norm_fun = lambda hist: max(np.max(hist), 1)
    # norm_fun = lambda hist: np.sqrt(np.sum(np.power(hist, 2.)))
    # norm_fun = lambda hist: np.sum(hist)
    norm_fun = lambda hist: np.trapz(hist, energy_bins_center)
    # norm_fun = lambda hist: np.sum(hist * energy_bin_size)

    norm_factor = norm_fun(energy_hist)
    norm_factor_bg = max(norm_fun(energy_hist_bg), 1)

    energy_hist = energy_hist / norm_factor
    energy_hist_bg = energy_hist_bg / norm_factor_bg

    energy_hist_events = []
    for energies_ev in energies_events:
        hist_temp, _ = np.histogram(
            energies_ev, bins=adc_n_bins, range=adc_limits, density=False
        )
        norm_temp = norm_fun(hist_temp)

        # TODO: Improve background subtraction
        out_temp = {
            'total': hist_temp,
            'no_bg': hist_temp - energy_hist_bg * norm_temp
        }

        energy_hist_events.append(out_temp)

    energy_hist *= norm_factor
    energy_hist_bg *= norm_factor_bg

    # ### FIGURE ###

    fig = plt.figure(figsize=(10, 8))

    bbox_anchor_limits = (1.03, 1)

    # ### Times ###
    # plt.subplot(4, 1, 1)
    # titlename = "%s%d" % ('CH', channel)
    # plt.title(titlename)
    #
    # plt.plot(times, '.b-')
    # plt.xlabel('Event #')
    # plt.ylabel('t (s)')

    # ### Events ###
    plt.subplot(3, 1, 1)

    plt.plot(times, energies, '.', ms=0.3, color='C0')
    plt.plot(times_bg, energies_bg, '.', ms=0.2, color='C1')
    for n, (times_ev, energies_ev) in enumerate(
            zip(times_events, energies_events)):
        plt.plot(times_ev, energies_ev, '.', ms=0.3, color='C' + str(n + 2))

    plt.xlim(min_time, max_time)

    plt.plot(np.nan, 'o', ms=5, color='C0', label='Shot')
    plt.plot(np.nan, 'o', ms=5, color='C1', label='Background')
    for n, _ in enumerate(times_events, start=1):
        plt.plot(np.nan, 'o', ms=5, color='C' + str(n + 1),
                 label='Event #%d' % n)
    plt.legend(bbox_to_anchor=bbox_anchor_limits, loc='upper left')

    plt.xlabel('t (s)')
    plt.ylabel('E (a.u.)')
    plt.title('Events')

    # ### TIME HISTOGRAM ###
    plt.subplot(3, 1, 2)

    plt.plot(time_bins_centre, time_hist, color='C0', label='Shot')
    plt.plot(time_bins_centre[time_bg_inds], time_hist[time_bg_inds],
             color='C1', label='Background')

    for n, event in enumerate(events_limits):
        inds = (time_bins_centre >= event[0]) & (time_bins_centre <= event[1])
        plt.plot(time_bins_centre[inds], time_hist[inds],
                 color='C' + str(n + 2), label='Event #' + str(n + 1))

    plt.xlim(min_time, max_time)
    plt.ylim(0, time_hist.max()*1.1)

    plt.legend(bbox_to_anchor=bbox_anchor_limits, loc='upper left')

    plt.xlabel('t (s)')
    plt.ylabel('Counts\n(every %.3g s)' % time_step)
    plt.title('Time Histogram')

    # #### ENERGY SPECTRUM ####

    plt.subplot(3, 1, 3)

    plt.axhline(0, ls='-', color='k', alpha=0.7)

    for line_energy in energy_lines:
        plt.axvline(line_energy, ls='--', color='k', alpha=0.7)
        plt.text(line_energy + 50, energy_hist.max() * 0.85,
                 '%.0f keV' % line_energy, alpha=0.7)

    plt.plot(energy_bins_center, energy_hist, color='C0', label='Entire Shot')
    plt.plot(energy_bins_center, energy_hist_bg, color='C1',
             label='Background')

    for n, energy_hist_ev in enumerate(energy_hist_events):
        plt.plot(energy_bins_center, energy_hist_ev['no_bg'],
                 '-', color='C' + str(n + 2),
                 label='Event #%d\n(background\n removed)' % (n + 1))

    plt.xlim(energy_plot_limits)
    if len(events_limits) > 0:
        plt.ylim(-10, 1.5*max(
            map(lambda f: f['no_bg'].max(), energy_hist_events)
        ))

    plt.legend(bbox_to_anchor=bbox_anchor_limits, loc='upper left')

    plt.xlabel('E (keV)')
    plt.ylabel('Counts')
    plt.title('Energy Spectrum')

    # ###### OUTPUT ######

    out_file_name = file_name_w_fix(filename[:-4])

    # ### Plots ###

    image_path = os.path.join(image_folder, out_file_name + '.png')

    plt.tight_layout(rect=[0, 0, 0.75, 1])

    plt.savefig(image_path, dpi=200)
    if verbose:
        print(' Plots saved to "' + image_path + '".')

    if plot_graphs:
        plt.show()
    else:
        plt.close(fig)

    # ### Events ###

    events_file = os.path.join(
        out_folder,
        file_name_w_fix(filename[:-4] + '_events')
    ) + '.dat'
    np.savetxt(
        events_file,
        np.vstack((times, energies)).T,
        fmt='%18g %18g',
        header='%16s %18s' % ('Time (s)', 'Energies (a.u.)')
    )
    if verbose:
        print(' Events saved to "' + events_file + '".')

    # ### Time Histogram ###

    times_file = os.path.join(
        out_folder,
        file_name_w_fix(filename[:-4] + '_times')
    ) + '.dat'
    np.savetxt(
        times_file,
        np.vstack((time_bins_centre, time_hist)).T,
        fmt='%18g %10d',
        header='Times are the centres of the bins\n'
               '%16s %10s' % ('Time (s)', 'Counts')
    )
    if verbose:
        print(' Time histogram saved to "' + times_file + '".')

    # ### Energy Histogram ###

    energies_file = os.path.join(
        out_folder,
        file_name_w_fix(filename[:-4] + '_energies')
    ) + '.dat'

    energy_table = [energy_bins_center, energy_hist, energy_hist_bg]

    for n_time, energy_hist_ev in enumerate(energy_hist_events):
        energy_table += [energy_hist_ev['total']]

    energy_table = np.vstack(energy_table).T

    format_str = '%18g %10d %10d ' + '%10d ' * len(energy_hist_events)
    format_str = format_str[:-1]

    header_str = 'Energies are the centres of the bins\n'
    header_str += '%16s %10s %10s ' % ('Energy (keV)', 'Shot', 'BG')
    for n, limits in enumerate(events_limits, start=1):
        header_str += '%10s ' % ('Event #%d' % n)
    header_str = header_str[:-1]
    header_str += '\n'

    np.savetxt(energies_file, energy_table, fmt=format_str, header=header_str)
    if verbose:
        print(' Energy spectrum saved to "' + energies_file + '".')

    # ### Energy Windows ###

    windows_counts = np.zeros(
        (len(energy_windows), 2 + 2*len(energy_hist_events))
    )

    for n_energy, energy_limits in enumerate(energy_windows):
        windows_counts[n_energy, 0:2] = energy_limits

        for n_time, energy_hist_ev in enumerate(energy_hist_events):
            # TODO (use edges)
            bins_to_use = (
                (energy_bins_center >= energy_limits[0])
                &
                (energy_bins_center <= energy_limits[1])
            )

            t_ind = 2 + 2*n_time

            windows_counts[n_energy, t_ind] = \
                np.sum(energy_hist_ev['total'][bins_to_use])
            windows_counts[n_energy, t_ind+1] = \
                np.sum(energy_hist_ev['no_bg'][bins_to_use])

    counts_channels.append(
        [channel, windows_counts[:, 2:]]
    )

    windows_file = os.path.join(
        out_folder,
        file_name_w_fix(filename[:-4] + '_counts')
    ) + '.dat'

    format_str = '%18g | %18g | ' + '%10d | %10d | '*len(energy_hist_events)
    format_str = format_str[:-3]

    header_str = 'Format = "%s"\n' % format_str
    header_str += '%16s | %18s | ' % ('Min Energy (keV)', 'Max Energy (keV)')
    for n, limits in enumerate(events_limits, start=1):
        header_str += '%23s | ' % (
                'Event #%d: %.2gs to %.2gs' % (n, limits[0], limits[1])
        )
    header_str = header_str[:-3]
    header_str += '\n'

    header_str += ' ' * 16 + ' | ' + ' ' * 18 + ' | '
    for _ in events_limits:
        header_str += '%10s | %10s | ' % ('Total', 'No BG')
    header_str = header_str[:-3]

    np.savetxt(windows_file, windows_counts, fmt=format_str, header=header_str)
    if verbose:
        print(' Energy windows saved to "' + windows_file + '".')

# Combined Counts File

combined_file_name = os.path.join(
    out_folder, file_name_w_fix('shot_' + shot_str + '_combined_counts')
) + '.dat'

format_str = '%4d | ' + '%10d | %10d | '*len(events_limits)
format_str = format_str[:-3]

header_str = '# Energy Window #%d: %.0f keV to %.0f keV\n'
header_str += '# ch | '
for n, limits in enumerate(events_limits, start=1):
    header_str += '%23s | ' % (
            'Event #%d: %.2gs to %.2gs' % (n, limits[0], limits[1])
    )
header_str = header_str[:-3]
header_str += '\n'

header_str += '#    | '
for _ in events_limits:
    header_str += '%10s | %10s | ' % ('Total', 'No BG')
header_str = header_str[:-3]
header_str += '\n'

with open(combined_file_name, 'w') as combined_file:
    combined_file.write('# Format = "' + format_str + '"\n')
    for n_energy, energy_limits in enumerate(energy_windows):
        combined_file.write(header_str % (
            n_energy+1, energy_limits[0], energy_limits[1]
        ))
        for channel, counts in counts_channels:
            line_values = tuple(np.hstack([channel, counts[n_energy, :]]))
            combined_file.write(format_str % line_values)
            combined_file.write('\n')

        combined_file.write('\n')

print('Counts from all channels saved to "' + combined_file_name + '".')
