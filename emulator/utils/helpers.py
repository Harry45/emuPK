# Author: (Dr to be) Arrykrishna Mootoovaloo
# Collaborators: Prof. Alan Heavens, Prof. Andrew Jaffe, Dr. Florent Leclercq
# Email : a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Important functions to store/load files in a compressed format.
'''

import os
import numpy as np
import dill


def save_excel(df, folder_name, file_name):
    '''
    Given a folder name and file name, we will save a pandas dataframe to a excel file.

    :param: df (pd.DataFrame) - pandas dataframe

    :param: folder name (str) - name of the folder

    :param: file name (str) - name of the file output
    '''

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    df.to_excel(excel_writer=folder_name + '/' + file_name + '.xlsx')


def load_arrays(folder_name, file_name):
    '''
    Given a folder name and file name, we will load
    the array

    :param: folder_name (str) - the name of the folder

    :param: file_name (str) - name of the file

    :return: matrix (np.ndarray) - array
    '''

    matrix = np.load(folder_name + '/' + file_name + '.npz')['arr_0']

    return matrix


def store_arrays(array, folder_name, file_name):
    '''
    Given an array, folder name and file name, we will store the
    array in a compressed format.

    :param: array (np.ndarray) - array which we want to store

    :param: folder_name (str) - the name of the folder

    :param: file_name (str) - name of the file
    '''

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # use compressed format to store data
    np.savez_compressed(folder_name + '/' + file_name + '.npz', array)


def load_pkl_file(folder_name, file_name):
    '''
    Given a folder name and a file name, we will load the Python class.
    For example, a full GP module

    :param: folder_name (str) - the name of the folder

    :param: file_name (str) - name of the file

    :return: module (Python class) : complete module or it can be EMCEE full module
    '''

    with open(folder_name + '/' + file_name + '.pkl', 'rb') as f:
        module = dill.load(f)

    return module


def store_pkl_file(module, folder_name, file_name):
    '''
    Given a trained GP (module), we will save it given
    a folder name and a file name

    :param: module (python class) - for example, the GP module

    :param: folder_name (str) - the name of the folder

    :param: file_name (str) - name of the file
    '''

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # store the module using dill
    with open(folder_name + '/' + file_name + '.pkl', 'wb') as f:
        dill.dump(module, f)
