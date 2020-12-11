import os

data_path = "/work/CAER-S/test"
sub_dir = "Angry"

total_files = sorted([f for dp, dn, fn in os.walk(os.path.expanduser(os.path.join(data_path, sub_dir))) for f in fn])
write_file = open(os.path.join(data_path, '{}.txt'.format(sub_dir)), 'w')
for file_name in total_files:
    write_file.writelines(os.path.join(sub_dir, file_name + '\n'))