#!/bin/bash

# TODO: needs parameters...
for i in $(find silero_vad_segments/ -type f -empty); do k=$(echo $i | sed -e 's|silero_vad_segments/||g' | sed -e 's/.seg//g'); l=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $(cat wav.scp | grep $k | awk '{print $2}')); echo "$k-0.0-$(printf '%.2f' $l)" $k 0.0 $(printf
 '%.2f' $l) > $i; done