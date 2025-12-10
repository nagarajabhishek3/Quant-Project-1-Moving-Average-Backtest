# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:01:32 2021

@author: User1
"""
import csv
import pandas as pd
import numpy as np
import os
import sys


def read_csv_to_list(file_name):
    try:
        if not (os.path.isfile(file_name)):
            return None
        
        data = []
        with open(file_name, 'r') as myfile:
            reader = csv.reader(myfile, delimiter=',', quotechar = '"', lineterminator = '\n')
            for row in reader:
                data.append(row)
            myfile.close()
        return data
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception :', ex)
        return None

def write_list_to_csv(file_name, data, append = False):
    """
    data : Must be a 2-D List
    """
    try:
        if append:
            openmode = 'a'
        else:
            openmode = 'w'
        with open(file_name, openmode, newline='') as myfile:
            writer = csv.writer(myfile, delimiter=',', lineterminator = '\n')
            for row in data:
                writer.writerow(row)
            myfile.close()
        return True
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception :', ex)
        return False
    
        
def read_csv_to_array(file_name):
    try:
        if not (os.path.isfile(file_name)):
            return np.empty(0)
        
        data = np.array(read_csv_to_list(file_name))
        return data
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception :', ex)
        return np.empty(0)
    

def write_array_to_csv(file_name, data):
    """
    data : Must be a 2-D Array
    """
    return write_list_to_csv(file_name, data)


def read_dataframe(file_name, separator = ','):
    try:
        if not (os.path.isfile(file_name)):
            return pd.DataFrame()
        
        data = pd.read_csv(file_name, sep = separator)
        return data
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception :', ex)
        return pd.DataFrame()


def write_dataframe(file_name, data, saveindexcol = False):
    """
    data : Must be a pandas dataframe
    """
    try:
        data.to_csv(file_name, index = saveindexcol)
        return True
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception :', ex)
        return False

def read_data_to_string(file_name):
    try:
        if not (os.path.isfile(file_name)):
            return None
        
        with open(file_name, 'r') as myfile:
            data = myfile.read()
            myfile.close()
        return data
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception :', ex)
        return None

def write_string_to_file(file_name, data, append = False):
    try:
        if append:
            openmode = 'a'
        else:
            openmode = 'w'
        with open(file_name, openmode) as myfile:
            myfile.write(data)
            myfile.close()
        
        return True
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception :', ex)
        return False
    

def read_data(file_name):
    try:
        if not (os.path.isfile(file_name)):
            return None
        
        with open(file_name, 'r') as myfile:
            data = myfile.readlines()
            myfile.close()
        return data
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception :', ex)
        return None

def write_data(file_name, data, addnewline = True, append = False):
    try:
        if append:
            openmode = 'a'
        else:
            openmode = 'w'
        data = np.array(data)
        with open(file_name, openmode, newline='') as myfile:
            for row in data:
                if type(row) != list and addnewline:
                    row = row + '\n'
                myfile.writelines(row)
            myfile.close()
        
        return True
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception :', ex)
        return False

def write_binary_data(file_name, data):
    try:
        data = np.array(data)
        with open(file_name, 'wb') as myfile:
            myfile.write(data)
            myfile.close()
        
        return True
    except:
        return False


import os
import pandas as pd

def read_stock_data(stock_name, data_path, date_column='Date', close_column='Close', setindex_date_column=True):
    """
    Reads <stock_name>.csv from data_path and returns a DataFrame.
    Handles Windows paths, case differences, .CSV extensions, and missing columns.
    """
    try:
        # Normalize path for Windows
        data_path = os.path.normpath(data_path)
        file_path = os.path.join(data_path, f"{stock_name}.csv")

        # Case-insensitive file search
        if not os.path.exists(file_path):
            all_files = os.listdir(data_path)
            match = [f for f in all_files if f.lower() == f"{stock_name.lower()}.csv"]
            if match:
                file_path = os.path.join(data_path, match[0])
            else:
                print(f"[WARN] No data file found for '{stock_name}' in {data_path}")
                return pd.DataFrame()

        # Read the CSV
        df = pd.read_csv(file_path)

        # Validate Date and Close columns
        if date_column not in df.columns:
            print(f"[WARN] '{date_column}' column not found in {file_path}. Columns found: {list(df.columns)}")
            return pd.DataFrame()

        if close_column not in df.columns:
            for alt in ['Adj Close', 'adj close', 'close', 'CLOSE']:
                if alt in df.columns:
                    close_column = alt
                    break
            else:
                print(f"[WARN] '{close_column}' column not found in {file_path}. Columns found: {list(df.columns)}")
                return pd.DataFrame()

        # Format and return
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce', infer_datetime_format=True)
        df = df.sort_values(by=date_column)
        if setindex_date_column:
            df = df.set_index(date_column)

        print(f"✅ Loaded {stock_name}: {len(df)} rows from {file_path}")
        return df

    except Exception as ex:
        print(f"read_stock_data exception: {ex}")
        return pd.DataFrame()



def read_stock_OHLCdata(stock_name, data_path, date_column = 'Date', open_column = 'Open', high_column = 'High', low_column = 'Low', close_column = 'Close'):
    # Asssuming :
    # Each stock data is in a separate CSV file
    # Filename is the same as the stockname
    # Adj closing prices are available
    # Prices are already sorted in ascending order
    
    try:
        file_name = data_path + '/' + stock_name + ".csv"
        if not (os.path.isfile(file_name)):
            return pd.DataFrame()
        
        data = pd.read_csv(file_name)
        data = data[[date_column, open_column, high_column, low_column, close_column]]
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.set_index(keys=date_column)
        return data
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception :', ex)
        return pd.DataFrame()

def create_tradelog():
    try:
        df = pd.DataFrame(columns=['TimeStamp','TrdSymbol','Qty','Price'])
        return df
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception: ', ex)
        return pd.DataFrame()

def LogTrade(df, TradeTime, Symbol, Qty, Price):
    try:
        df = df.append({'TimeStamp':TradeTime,'TrdSymbol':Symbol,'Qty': Qty,'Price': Price}, ignore_index=True)
        return df
    except Exception as ex:
        print(sys._getframe().f_code.co_name, 'exception: ', ex)
        return pd.DataFrame()


