import scipy.io

# Load the .mat file
mat_contents = scipy.io.loadmat('/Users/youngj/Local/Project/2022_Rolling_Bearing_Fault_Diagnosis/CaseWesternReserveUniversityData-master/48k_Drive_End_OR021@12_3_265.mat')

# Print the contents
for key, value in mat_contents.items():
    print(key)
    print(value)