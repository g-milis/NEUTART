#!/bin/bash

# Quickly inspect the duration of each clip in your dataset
# Simply adjust the path pointing to the clips and run the script

videos_path=/path/to/videos/*.mp4


for mp4 in $(ls $videos_path); do
    ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $mp4 >> durations.txt
done

seconds=$(awk '{sum += $1} END {print sum}' durations.txt)
echo -n "Seconds: "
echo $seconds

echo -n "Minutes: "
echo "scale=2 ; $seconds / 60" | bc

rm durations.txt
