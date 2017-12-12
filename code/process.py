import numpy as np
import pandas as pd
from sklearn import preprocessing
import config

logger = config.config_logger(__name__,10)

def basic_descriptives(my_df):
    n_row, n_col = my_df.shape
    cols = my_df.columns
    logger.info('# observations: {0}'.format(n_row))
    logger.info('# features: {0}'.format(n_col))
    logger.info('Features: {0}'.format(cols))
    return

def encode_variables(my_df, my_list, my_dict):
    ''' 
    Enconde variables in my_list from the dataframe my_df with the values 
    that appear in dictionary my_dict.
    Output the encoded dataframe.
    '''
    my_df = my_df.copy()
    for var in my_list:
        temp_dict = {var: my_dict}
        my_df.replace(temp_dict, inplace = True)
    return my_df

def get_dicts():
    group1_dict = {'SI': 1, 'NO': 0}
    group2_dict = {'Positivo': 1, 'Negativo': 0, ' ': np.nan}
    group3_dict = {'M': 1, 'F': 0}

    return [group1_dict, group2_dict, group3_dict]

def get_vars():
    group1_vars = ['fiebre', 'escalofrio', 'cefalea', 'mareos', 'tos',
                   'odinofagia', 'nauseas', 'hiporexia', 'dol_lumbar',
                   'disuria', 'mialgias', 'artralgias', 'inyec_conjut',
                   'dol_retroocular', 'erup_cutanea', 'melena', 'epistaxis',
                   'gingivorragia', 'ginecorragia', 'petequias', 'equimosis',
                   'esp_hemoptoico', 'dol_abdominal_int', 'disnea',
                   'vomito_persistente', 'hipotermia', 'lipotimia',
                   'ictericia', 'dism_plaquetas', 'incr_hematoccrito',
                   'somnolencia', 'hipotension_arterial', 'ext_cianoticas',
                   'pulso_debil_rapido', 'dif_PA_20']
    group2_vars = ['dengue_pcr', 'serotipo1', 'serotipo2', 'serotipo3',
                   'serotipo4', 'zika_pcr', 'chik_pcr']
    group3_vars = ['sex']
    return [group1_vars, group2_vars, group3_vars]

def preprocess(my_df):
    my_df = my_df.copy()
    dicts = get_dicts()
    vars = get_vars()
    for var, dict in zip(vars,dicts):
        my_df = encode_variables(my_df, var, dict)
    my_df['age'] = preprocessing.scale(np.array(my_df['age']))
    return my_df
        
def remove_vars(my_df, remove_list):
    '''
    Remove columns in my_df whose name appear in remove_list.
    '''
    my_df = my_df.copy()
    output = my_df.drop(remove_list, axis = 1)
    return output
    
def keep_non_nan(my_df, column):
    '''
    Remove rows from my_df that have NANs or do not appear in column.
    Input column is a pandas Series.
    '''
    my_df = my_df.copy()
    output = my_df.loc[column.index]
    output = output[pd.notnull(column)]
    return output



