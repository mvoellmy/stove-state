# stove-state

## How to use easy_record.py
* Change paths in /cfg/cfg.txt so they both point to the same `polybox/ssds` folder.
* run `mount /mnt/polybox` after every boot
* run `python3 easy_record.py`
* label the objects after the following syntax (tampstamp is added automatically):
https://github.com/mvoellmy/stove-state/wiki/Food-Labeling/


## NOT WORKING Instructions to run easy_record.py on startup
* Modify the path in 'easy_record_exec.desktop' so it points to 'easy_record.py'
* Copy (!) 'easy_record_exec.desktop' (shown as Easy Record Exec in Rasbian) to '/home/pi/.config/autostart/'

The file python script will now execute automatically at boot
