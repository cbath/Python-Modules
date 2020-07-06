#Standard Libraries
import pandas as pd
#Regular expressions  (string comparisons)
import re

#Parent of all Python reader functions

#Load csv or xlsx
def read_data(filepath):
    #Try and be smart and handle all data types
    m = re.search('.xlsx',filepath)
    if m:
        dataframe = pd.read_excel(filepath)
    else:
        dataframe = pd.read_csv(filepath)
    print("File loaded with {} row and {} columns.".format(*dataframe.shape))
    print(dataframe.head(5))
    return dataframe
    #TODO - conditional based on file type
    
#Example - always have to unicode escape for windows file paths using r''
x=r"C:\Users\cbath\Documents\Test Files\BMC_Post-Pipeline-ZSO-US_062220.csv"
#x=r"C:\Users\cbath\Documents\Test Files\BMC_Post-Pipeline-ZSO-US_062220.xlsx"
data=read_data(x)

