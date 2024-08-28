import os
from datetime import datetime, timedelta

def list_missing_hours(directory, start_time, end_time, prefix, file_type):
    # Convert the input strings to datetime objects
    start_dt = datetime.strptime(start_time, '%Y-%m-%d %H%M%S')
    end_dt = datetime.strptime(end_time, '%Y-%m-%d %H%M%S')

    # List all files in the directory
    files = os.listdir(directory)

    # Extract timestamps from file names
    available_times = set()
    for file in files:
        if file.startswith(prefix) and file.endswith(file_type):
            # Extract the timestamp portion from the filename
            timestamp_str = file.split('_')[-1].replace(file_type, '').replace('.gz', '')
            timestamp_dt = datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S')
            available_times.add(timestamp_dt)

    # Generate a list of all expected times within the range
    current_time = start_dt
    missing_times = []
    while current_time <= end_dt:
        if current_time not in available_times:
            missing_times.append(current_time)
        current_time += timedelta(hours=1)

    return missing_times

directory_full = r"Y:\ATD\GIS\MRMS_Data\MRMS Data USA\2023"
directory_clip = r"Y:\ATD\GIS\MRMS_Data\MRMS Data Clipped\2023"
start_time = '2023-05-01 000000'
end_time = '2023-10-21 230000'

# start_time = '2023-01-01 010000'
# end_time = '2023-10-21 230000'
prefix = "MultiSensor_QPE_01H_Pass2_00.00_"
file_type = ".grib2"
prefix_clip = "Clipped_"
file_type_clip = ".tif"
missing_hours_full = list_missing_hours(directory_full, start_time, end_time, prefix, file_type)
missing_hours_clipped = list_missing_hours(directory_clip, start_time, end_time, 
                                           prefix_clip, file_type_clip)

# Print the missing hours present in clipped files but not in full files
missing_hours = [hour for hour in missing_hours_clipped if hour not in missing_hours_full]
print("Missing hours from clipped files:")
for missing in missing_hours:
    print(missing.strftime('%Y-%m-%d %H%M%S'))


if missing_hours_full:
    print("\nMissing hours from MRMS:")
    for missing in missing_hours:
        print(missing.strftime('%Y-%m-%d %H%M%S'))
else:
    print("No missing hours found.")

#save missing hours to a text file
output_file = r"Y:\ATD\GIS\MRMS_Data\Missing_Hours.txt"
with open(output_file, 'w') as f:
    f.write(f"2022 start time: {start_time}\n")
    f.write(f"2022 end time: {end_time}\n")
    f.write("Missing hours from MRMS:\n")
    for missing in missing_hours:
        f.write(missing.strftime('%Y-%m-%d %H%M%S') + '\n')