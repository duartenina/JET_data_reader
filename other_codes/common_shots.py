import os
import sys

shot = sys.argv[-1]

options = {
    '81852': "-s 81852 -F ..\\bin_files\\81852 -b 32 -f 2.5 -B 65 70"
             " --time-shot 28 100 -T 0.1 -t 50,54 58,60"
             " -e 2300,7000 500,520 --energy-plot-limits 0 8000"
             " -i 1 9 10 11 12",
             # " -i 1 3 4 5 6 7 8 9 10 11 12 13 14 17 18 19",
    '91975': "-s 91975 -F ..\\bin_files\\91975 -b 32 -f 2.5 -B 30 45"
             " -i 1 9 10 11 -T 0.1 -e 490,520 2300,7000 --time-shot 28 85 -t 47,51",
    'Na': "-s Na -F ..\\bin_files\\Na -b 48 -f 200 -e 500,520 1460,1480"
          " -E 511 1475 --time-shot 0 3600 --time-delay 0 -T 60",
    'Cs': "-s Cs2 -F ..\\bin_files\\Cs -b 48 -f 200 -e 500,520 1460,1480"
          " -E 511 1475 --time-shot 0 3600 --time-delay 0 -T 60"
}

if shot in options:
    print('python read_data.py ' + options[shot])
    os.system('python read_data.py ' + options[shot])
elif shot == 'all':
    for shot in options:
        print('python read_data.py ' + options[shot])
        os.system('python read_data.py ' + options[shot])
else:
    print('shot not recognized: "' + shot + '"')
