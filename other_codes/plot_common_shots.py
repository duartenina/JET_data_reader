import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Patch as colorbox
from matplotlib.lines import Line2D


rc('font', size=28)
rc('legend', fontsize=28)
rc('xtick', labelsize=25)
rc('ytick', labelsize=25)
rc('text', usetex=True)

shot = sys.argv[-1]

if shot == '81852':
    # Energies
    data = np.loadtxt(
        '..\\bin_files\\81852\\out\\shot_81852_combined_counts.dat',
        delimiter='|'
    )
    data = data.reshape((2, 14, 5))

    data_useful_horz = data[0, :7, :]
    data_useful_vert = data[0, 7:, :]

    plt.figure(figsize=(14, 8))

    plt.subplot(1, 2, 1)

    plt.plot(data_useful_horz[:, 0], data_useful_horz[:, 3], '-o',
             color='C0', ms=2)
    plt.plot(data_useful_vert[:, 0], data_useful_vert[:, 3], '-o',
             color='C0', label='With background')

    plt.plot(data_useful_horz[:, 0], data_useful_horz[:, 4], '-o',
             color='C1')
    plt.plot(data_useful_vert[:, 0], data_useful_vert[:, 4], '-o',
             color='C1', label='Without background')

    plt.text((2+8)/2, 250, 'Horizontal\nCameras', ha='center', va='top')
    plt.text((13+19)/2, 250, 'Vertical\nCameras', ha='center', va='top')

    plt.xticks(np.arange(0, 21, 2))
    plt.ylim(0, 1700)

    plt.title('Neutral Beam Injection')
    plt.xlabel('Detector number')
    plt.ylabel('Counts')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(data_useful_horz[:, 0], data_useful_horz[:, 1], '-o',
             color='C0')
    plt.plot(data_useful_vert[:, 0], data_useful_vert[:, 1], '-o',
             color='C0', label='With background')

    plt.plot(data_useful_horz[:, 0], data_useful_horz[:, 2], '-o',
             color='C1')
    plt.plot(data_useful_vert[:, 0], data_useful_vert[:, 2], '-o',
             color='C1', label='Without background')

    plt.text((2+8)/2, 50, 'Horizontal\nCameras', ha='center', va='top')
    plt.text((13+19)/2, 50, 'Vertical\nCameras', ha='center', va='top')

    plt.xticks(np.arange(0, 21, 2))
    plt.ylim(0, 350)

    plt.title('Ion Cyclotron Resonance')
    plt.xlabel('Detector number')
    plt.ylabel('Counts')

    plt.legend()

    plt.tight_layout()
    plt.savefig('..\images\\heatings.png')
    plt.close()

    # Times

    plt.figure(figsize=(14, 7))

    times = np.loadtxt(
        '..\\bin_files\\81852\\out\\shot_81852_ch15_times.dat'
    )
    events = np.loadtxt(
        '..\\bin_files\\81852\\out\\shot_81852_ch15_events.dat'
    )

    limits_all = (
        (65, 70),
        (50, 54),
        (58, 60)
    )

    plt.subplot(1, 2, 1)

    plt.plot(events[:, 0], events[:, 1], '.', ms=1, color='C0')

    for n, limits in enumerate(limits_all, start=1):
        inds = (events[:, 0] >= limits[0]) & (events[:, 0] <= limits[1])
        plt.plot(events[inds, 0], events[inds, 1], '.', ms=1,
                 color='C' + str(n))

    plt.xlim(45, 75)

    plt.xlabel('t (s)')
    plt.ylabel('Energy (a.u.)')
    plt.title('Events')

    plt.subplot(1, 2, 2)

    plt.plot(times[:, 0], times[:, 1], lw=2, color='C0', label='Shot')

    for n, limits in enumerate(limits_all, start=1):
        inds = (times[:, 0] >= limits[0]) & (times[:, 0] <= limits[1])
        plt.plot(times[inds, 0], times[inds, 1], '-', lw=2, color='C' + str(n))

    plt.xlim(45, 75)

    plt.xlabel('t (s)')
    plt.ylabel('Counts\n(every 0.1s)')
    plt.title('Time Histogram')

    plt.subplots_adjust(bottom=0.35, wspace=0.5)

    ax_leg = plt.axes([0.1, 0.05, 0.8, 0.1], frameon=False)

    ax_leg.set_xticks([])
    ax_leg.set_yticks([])

    names = [
        'Shot',
        'Background',
        'Ion Cyclotron Resonance',
        'Neutral Beam Injection'
    ]

    for n, name in enumerate(names):
        ax_leg.plot(np.nan, '-o', ms=15, color='C' + str(n), label=name)

    ax_leg.legend(loc='center', ncol=2, columnspacing=2)

    # plt.tight_layout()
    plt.savefig('..\\images\\shot_2_events.png')
    plt.close()

elif shot in ['Na', 'Cs']:
    plt.figure(figsize=(14, 7))

    data_eg = {}
    for shot in ['Na', 'Cs']:
        filename = '..\\bin_files\\' + shot + '\\out\\shot_' + shot

        if shot == 'Na':
            filename += ''
        else:
            filename += '2'

        filename += '_ch7'

        data_eg[shot] = np.loadtxt(filename + '_energies.dat')

    plt.plot(data_eg['Cs'][:, 0], data_eg['Cs'][:, 1], '-',
             label=r'$^{137}$Cs + $^{133}$Ba + $^{22}$Na')
    plt.plot(data_eg['Na'][:, 0], data_eg['Na'][:, 1], '-',
             label=r'$^{22}$Na')
    plt.plot(data_eg['Cs'][:, 0], data_eg['Cs'][:, 1] - data_eg['Na'][:, 1],
             '-', label=r'$^{137}$Cs + $^{133}$Ba \\($^{22}$Na removed)')

    plt.axhline(0, ls='-', color='k', alpha=0.7)

    lines = [
        # (302, 900, '$^{133}$Ba'),
        (356, 1350, '$^{133}$Ba'),
        (511, 500, '$^{22}$Na'),
        (662, 1100, '$^{137}$Cs'),
        (1475, 200, '$^{22}$Na')
    ]

    for line in lines:
        plt.axvline(line[0], ymin=0, ymax=line[1]/1500,
                    ls='--', color='k', alpha=0.7)
        plt.text(line[0]+30, line[1]+00, line[2] + ' (%d keV)' % line[0],
                 va='top', ha='left',
                 bbox=dict(fc='w', ec='none', alpha=0.7))

    plt.xlim(0, 2500)
    plt.ylim(-20, 1500)

    plt.xlabel('E (keV)')
    plt.ylabel('Counts')
    plt.title('Calibration Shots')

    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('..\\images\\calibration.png')
    plt.close()
elif shot == 'time':
    file_path = '..\\bin_files\\91975\\shot_91975_ch2.bin'

    word_64_type = np.dtype(
        [('energy', '<u2'), ('empty', '<u2'), ('time', '<u4')]
    )
    data = np.memmap(file_path, dtype=word_64_type, mode='r')

    num_pulses = np.argmax(data['time'][1:] < data['time'][:-1])

    plt.figure(figsize=(10, 6))

    xmin = -1*10**3
    xmax = 45*10**3

    plt.plot(data['time'][:xmax], '.', ms=10, color=(0.3, 0.6, 1))
    plt.plot(data['time'][:xmax], '--', lw=1.5, color=(0.7, 0, 0))

    plt.axvspan(xmin, 0, color='k', alpha=0.2)
    plt.axvline(0, color='k')

    plt.axvline(num_pulses, color='k')
    plt.axvspan(num_pulses, xmax, color='k', alpha=0.2)

    plt.xlim(xmin, xmax)
    plt.ylim(0, 1.7*10**8)

    plt.xlabel('Row \#')
    plt.ylabel('Cycle \#')
    plt.title('Timestamp')

    plt.legend(
        handles=[
            Line2D([], [], color=(0.7, 0, 0), marker='o', lw=2,
                   mfc=(0.3, 0.6, 1), ms=15, mec='none'),
            colorbox(facecolor='w', edgecolor='k'),
            colorbox(facecolor='k', alpha=0.2, edgecolor='k')
        ],
        labels=[
            'Timestamp as read',
            'Valid data',
            'Invalid data'
        ],
        loc=(0.1, 0.53)
    )

    plt.tight_layout()
    plt.savefig('..\\images\\time_array.png')
    plt.close()
elif shot == 'bg':
    data = np.loadtxt(
        '..\\bin_files\\81852\\out\\shot_81852_ch15_energies.dat'
    )

    energies = data[:, 0]
    counts = data[:, 1]
    counts_bg = data[:, 2]
    counts_ev = []
    for i in range(2):
        counts_ev.append(data[:, 3+i])

    plt.figure(figsize=(13, 5.5))

    plt.plot(energies, counts_bg, '-', color='k', label='Background')

    for i in range(2):
        plt.plot(energies, counts_ev[i], '-', color='C' + str(i),
                 label='Event \#' + str(i+1))
        plt.plot(energies, counts_ev[i] - counts_bg, '--', color='C' + str(i),
                 label='Event \#' + str(i+1) + '\n(without\nbackground)')

    plt.axhline(0, color='k', alpha=0.5)

    plt.xlim(200, 700)

    plt.xlabel('E (keV)')
    plt.ylabel('Counts')
    plt.title('Background Subtraction')

    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize=28)

    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig('..\\images\\background_subtraction')
    plt.close()
else:
    print('shot not recognized: "' + shot + '"')
