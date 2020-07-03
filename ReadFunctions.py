#Dependencies
import pandas as pd

#Parent of all Python reader functions

#Literal string conversion for file paths
def raw_string(filepath):
    cleaned=r'{}'.format(filepath)
    return cleaned

#Load csv or xlsx
def read_data(filepath):
    data=pd.read_csv(filepath)
    print("File loaded with {} row and {} columns.".format(*data.shape))
    return data
    #TODO - conditional based on file type

#res=read_data(r"C:\Users\cbath\Documents\Test Files\BMC_Post-Pipeline-ZSO-US_062220.csv")
raw_string("C:\Users\cbath\Documents\Test Files\BMC_Post-Pipeline-ZSO-US_062220.csv")
#TODO - convert to a data array


