import pycatch22
import os
import csv
import numpy as np

def get_label(filename):

    for i, char in enumerate(filename):
        if char == "-":
            return filename[(i+1):(i+4)]

def processing_file(data_folder, dump_location = None, mode = "fusion"):

    folders = os.listdir(data_folder)
    for folder in folders:

        ID = os.path.join(data_folder, folder)
        files = os.listdir(ID)
        data = []

        if mode == "fusion":
            for file in files:
                with open(os.path.join(ID, file),'r') as f:
                    pickle_in = list(csv.reader(f, delimiter='\t'))
                    seq = np.array(pickle_in)[1:,1:].astype(float)
                    eda =  pycatch22.catch22_all(list(seq[:,0]))["values"]
                    ecg =  pycatch22.catch22_all(list(seq[:,1]))["values"]
                    emg1 =  pycatch22.catch22_all(list(seq[:,2]))["values"]
                    emg2 =  pycatch22.catch22_all(list(seq[:,3]))["values"]
                    emg3 =  pycatch22.catch22_all(list(seq[:,4]))["values"]
                    features = [*eda, *ecg, *emg1, *emg2, *emg3]
                    features.append(get_label(file))
                    data.append(features)

        elif mode == "eda":
            for file in files:
                with open(os.path.join(ID, file),'r') as f:
                    pickle_in = list(csv.reader(f, delimiter='\t'))
                    seq = np.array(pickle_in)[1:,1].astype(float)
                    features =  pycatch22.catch22_all(list(seq))["values"]
                    features.append(get_label(file))
                    data.append(features)

        elif mode == "ecg":
            for file in files:
                with open(os.path.join(ID, file),'r') as f:
                    pickle_in = list(csv.reader(f, delimiter='\t'))
                    seq = np.array(pickle_in)[1:,2].astype(float)
                    features =  pycatch22.catch22_all(list(seq))["values"]
                    features.append(get_label(file))
                    data.append(features)
                    
        elif mode == "emg":
            for file in files:
                with open(os.path.join(ID, file),'r') as f:
                    pickle_in = list(csv.reader(f, delimiter='\t'))
                    seq = np.array(pickle_in)[1:,3:6].astype(float)
                    features1 =  pycatch22.catch22_all(list(seq[:,0]))["values"]
                    features2 =  pycatch22.catch22_all(list(seq[:,1]))["values"]
                    features3 =  pycatch22.catch22_all(list(seq[:,2]))["values"]
                    features = [*features1, *features2, *features3]
                    features.append(get_label(file))
                    data.append(features)
        else:
            raise ValueError("You did not provide a supported feature class")

        if dump_location is None:
            raise ValueError("You need to provide a file dumping location")
        loc = dump_location+"/"+folder+"_"+mode+".csv"
        np.savetxt(loc, np.array(data), delimiter=",", fmt='%s')
        

def main():
    data_path = r"C:\Users\ronny\Documents\GitHub\MetaPain\Data\biosignals_filtered"
    dump_path = "C:/Users/ronny/Documents/GitHub/MetaPain/Data/biosignals_filtered_Processed/emg"
    processing_file(data_folder = data_path, dump_location = dump_path, mode = "emg")


if __name__ == "__main__":

    main()