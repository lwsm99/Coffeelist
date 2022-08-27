# Data Export  script which exports the data into an Excel sheet using pandas

import pandas as pd
import openpyxl


# Export into Excel
def export_data(names):
    file_name = 'Coffeelist.xlsx'

    df = pd.DataFrame(names, columns=['Names', 'x', 'y', 'width', 'height', 'Coffee count'])
    df = df[['Names', 'Coffee count']]

    df.to_excel(file_name)
