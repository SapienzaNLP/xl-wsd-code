import os

def print_mfs_info(non_mfs_predictions):
    mfs_predictions = non_mfs_predictions.replace(".txt", ".mfs.txt")
    outpath = non_mfs_predictions.replace(".txt", ".mfs.info.txt")
    mfs_ids = set()
    with open(non_mfs_predictions) as lines:
        for line in lines:
            fields = line.strip().split(" ")
            if "<unk>" == fields[-1]:
                mfs_ids.add(fields[0])
    with open(mfs_predictions) as lines, open(outpath, "w") as writer:
        for line in lines:
            fields = line.strip().split(" ")
            if fields[0] in mfs_ids:
                writer.write(line.strip() + " MFS\n")
            else:
                writer.write(line)

def print_mfs_info_by_folder(folder):
    for f in os.listdir(folder):
        if f.endswith(".predictions.txt"):
            print_mfs_info(os.path.join(folder, f))