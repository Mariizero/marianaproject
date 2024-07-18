import pandas as pd


def dat_to_excel(dat_file_path, excel_file_path, encoding='utf-8'):
    # Read the .dat file with the specified encoding
    try:
        df = pd.read_csv(dat_file_path, delimiter='\s+', encoding=encoding)
    except UnicodeDecodeError as e:
        print(f"Error reading .dat file with encoding {encoding}: {e}")
        return
    except Exception as e:
        print(f"Error reading .dat file: {e}")
        return

    # Write the data to an Excel file
    try:
        df.to_excel(excel_file_path, index=False)
        print(f"Successfully converted {dat_file_path} to {excel_file_path}")
    except Exception as e:
        print(f"Error writing to Excel file: {e}")


# Usage example
dat_file_path = 'rec_1.dat'
excel_file_path = 'data.xlsx'

# Try with different encodings
for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'utf-16']:
    print(f"Trying with encoding: {encoding}")
    dat_to_excel(dat_file_path, excel_file_path, encoding=encoding)
    if dat_to_excel(dat_file_path, excel_file_path, encoding=encoding):
        break
