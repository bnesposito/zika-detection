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


def select_disease(diseases, code, only_one=False):
    """ Create the dependent variable as a function of the diseases.

    Args:
        diseases (list): collection of three pd.series (Dengue, Zika, Chik).
        code (int): function selection key
            1. Dengue
            2. Zika
            3. Chik
            4. Any
        only_one (bool): if True, input np.nan to patients with a disease
            different than the one selected by code.

    Returns:
        pd.series: dependent variable.

    Raises:
        ValueError if code is not integer
        ValueError if code is not in range
    """
    if not isinstance(code, int):
        raise ValueError('Code must be an integer')

    if code not in range(1, 5):
        raise ValueError('Code must be inside range 1-4')

    y_dengue, y_zika, y_chik = diseases
    output = np.nan
    if code <= 3:
        output = diseases.pop(code-1)
        if only_one:
            for other_disease in diseases:
                output = input_nan(output, other_disease)
    if code == 4:
        output = pd.Series([any(x) for x in zip(y_dengue, y_zika, y_chik)]).astype(int)
    return output


def input_nan(target_series, indicator):
    """ Input np.nan in target_series if indicator is 1.

    Args:
        target_series (pd.series): series that will be inputted with np.nans.
        indicator (pd.series): indicator for the np.nan input
            (np.nan if indicator is 1).

    Returns:
        pd.Series: target_series inputted with np.nans.

    Raises:
        ValueError if the length of target_series and indicator is different.
    """
    if len(target_series) != len(indicator):
        raise ValueError("The target series and the indicator must have the same length")

    output = []
    for target, ind in zip(target_series, indicator):
        if ind == 0:
            output.append(target)
        else:
            output.append(np.nan)
    return pd.Series(output)

