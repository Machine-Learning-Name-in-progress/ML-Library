import pytest
import pandas as pd
from src.ml_library.preprocessing.empty_values_indicate import empty_values_indicate
from src.ml_library.preprocessing.empty_values_remove import empty_values_remove
from src.ml_library.preprocessing.empty_values_replace import empty_values_replace
from src.ml_library.preprocessing.normalization import normalization
from src.ml_library.preprocessing.one_hot import one_hot
from src.ml_library.preprocessing.standardization import standardization

@pytest.fixture
def data():
    data = {
        'Name': ['Ana', 'Bruno', 'Carlos', 'Diana', 'Eduardo'],
        'Age': [25, float('nan'), 30, 22, float('nan')],
        'City': ['São Paulo', 'Rio de Janeiro', 'Salvador', 'Belo Horizonte', float('nan')],
        'Score': [8.5, float('nan'), 9.2, float('nan'), 6.5],
        }

    df = pd.DataFrame(data)
    return df

@pytest.fixture
def numerical_data():
    numerical_data = {
        '1': ['A', 'B', 'C', 'D'],
        '2': [1, 4, 3, 5],
        '3': [-10.6, 14.7, 55.3, -9.1],
        '4': [5.3, 4.7, -2, -4.6],
        '5': [4, -8, 9, -2],
        }

    df = pd.DataFrame(numerical_data)
    return df

@pytest.fixture
def categorical_data():
    categorical_data = {
        '1': ['A', 'B', 'C'],
        '2': ['B', 'B', 'A'],
        '3': ['A', 'C', 'C'],
        }

    df = pd.DataFrame(categorical_data)
    return df

def test_empty_values_indicate(data):
    df = data
    df_ind = empty_values_indicate(df)

    expected_column = pd.Series([0, 1, 0, 1, 1], name="indicate")

    pd.testing.assert_series_equal(expected_column, df_ind['indicate'])
    pd.testing.assert_frame_equal(df_ind[df.columns],df)

    df_ind = empty_values_indicate(df,['Age', 'City'], 2, "has_two_nan")

    expected_column = pd.Series([0, 0, 0, 0, 1], name="has_two_nan")

    pd.testing.assert_series_equal(expected_column, df_ind['has_two_nan'])
    pd.testing.assert_frame_equal(df_ind[df.columns],df)

def test_empty_values_indicate_exceptions(data):
    df = data

    with pytest.raises(ValueError) as excep:
        empty_values_indicate(df, count=2.5)
    assert str(excep.value) == "'count' needs to be an integer greater than 0"

    with pytest.raises(ValueError) as excep:
        empty_values_indicate(df, count=0)
    assert str(excep.value) == "'count' needs to be an integer greater than 0"

    with pytest.raises(ValueError) as excep:
        column = 'Weight'
        empty_values_indicate(df, [column])
    assert str(excep.value) == f"'{column}' column does not exist on dataframe"

    with pytest.raises(ValueError) as excep:
        indicator_name = 'Score'
        empty_values_indicate(df, indicator_name=indicator_name)
    assert str(excep.value) == f"'{indicator_name}' is an existing column on dataframe. Give another name."

    with pytest.raises(ValueError) as excep:
        count = 4
        columns = ['Name', 'Age', 'City']
        empty_values_indicate(df, columns, count)
    assert str(excep.value) == f"'count' ({count}) cannot be greater than the number of analyzed " f"columns ({len(columns)})"

def test_empty_values_remove(data):
    df = data

    df_clean = empty_values_remove(df, ['Name', 'Age', 'City'])

    df_expected = pd.DataFrame({
        'Name': ['Ana', 'Carlos', 'Diana'],
        'Age': [25, 30, 22],
        'City': ['São Paulo', 'Salvador', 'Belo Horizonte'],
        'Score': [8.5, 9.2, float('nan')],
        })

    pd.testing.assert_frame_equal(df_clean, df_expected, check_dtype=False)

    df_clean = empty_values_remove(df, ['Age', 'City'], False)
    
    df_expected = pd.DataFrame({
        'Name': ['Ana', 'Bruno', 'Carlos', 'Diana'],
        'Age': [25, float('nan'), 30, 22],
        'City': ['São Paulo', 'Rio de Janeiro', 'Salvador', 'Belo Horizonte'],
        'Score': [8.5, float('nan'), 9.2, float('nan')],
        })

    pd.testing.assert_frame_equal(df_clean, df_expected, check_dtype=False)

def test_empty_values_remove_exceptions(data):
    df = data

    with pytest.raises(ValueError) as excep:
        column = 'Weight'
        empty_values_remove(df, [column])
    assert str(excep.value) == f"'{column}' column does not exist on dataframe"

def test_empty_values_replace(data):
    df = data

    df_replaced, dict = empty_values_replace(df, 'mean')

    df_expected = pd.DataFrame({
        'Name': ['Ana', 'Bruno', 'Carlos', 'Diana', 'Eduardo'],
        'Age': [25, 77/3, 30, 22, 77/3],
        'City': ['São Paulo', 'Rio de Janeiro', 'Salvador', 'Belo Horizonte', float('nan')],
        'Score': [8.5, 24.2/3, 9.2, 24.2/3, 6.5],
        })

    dict_expected = {'Age':77/3, 'Score': 24.2/3}

    pd.testing.assert_frame_equal(df_replaced, df_expected, check_dtype=False)
    assert dict == dict_expected

    df_replaced, dict = empty_values_replace(df, 'median')
    
    df_expected = pd.DataFrame({
        'Name': ['Ana', 'Bruno', 'Carlos', 'Diana', 'Eduardo'],
        'Age': [25, 25, 30, 22, 25],
        'City': ['São Paulo', 'Rio de Janeiro', 'Salvador', 'Belo Horizonte', float('nan')],
        'Score': [8.5, 8.5, 9.2, 8.5, 6.5],
        })

    dict_expected = {'Age':25, 'Score': 8.5}

    pd.testing.assert_frame_equal(df_replaced, df_expected, check_dtype=False)
    assert dict == dict_expected

    df_replaced, dict = empty_values_replace(df, 'fill', ['Age', 'Score'], fill_value=14)

    df_expected = pd.DataFrame({
        'Name': ['Ana', 'Bruno', 'Carlos', 'Diana', 'Eduardo'],
        'Age': [25, 14, 30, 22, 14],
        'City': ['São Paulo', 'Rio de Janeiro', 'Salvador', 'Belo Horizonte', float('nan')],
        'Score': [8.5, 14, 9.2, 14, 6.5],
        })

    dict_expected = {'Age':14, 'Score': 14}

    pd.testing.assert_frame_equal(df_replaced, df_expected, check_dtype=False)
    assert dict == dict_expected

def test_empty_values_replace_excptions(data):
    df = data

    with pytest.raises(ValueError) as excep:
        method = 3
        empty_values_replace(df, method)
    assert str(excep.value) == f"'{method}' is not a valid method"

    with pytest.raises(ValueError) as excep:
        column = 'Weight'
        empty_values_replace(df, 0, ['Weight'])
    assert str(excep.value) == f"'{column}' column does not exist on dataframe"

    with pytest.raises(ValueError) as excep:
        empty_values_replace(df, 2)
    assert str(excep.value) == "'fill_value' parameter must be provided when using 'fill' method"

def test_normalization(numerical_data):
    df = numerical_data

    df_normalized, dict = normalization(df, ['2','3','4','5'])

    df_expected = pd.DataFrame({
        '1': ['A', 'B', 'C', 'D'],
        '2': [0., 0.75, 0.5, 1.],
        '3': [0., 0.38391502, 1., 0.02276176],
        '4': [1., 0.93939394, 0.26262626, 0.],
        '5': [0.70588235, 0., 1., 0.35294118],
        })

    dict_expected = {'2':{'min':1, 'max':5}, '3':{'min':-10.6, 'max':55.3}, '4':{'min':-4.6, 'max':5.3}, '5':{'min':-8, 'max':9}}

    pd.testing.assert_frame_equal(df_normalized, df_expected)
    assert dict == dict_expected

def test_normalization_exceptions(numerical_data):
    df = numerical_data

    with pytest.raises(ValueError) as excep:
        column = 'Weight'
        normalization(df, ['Weight'])
    assert str(excep.value) == f"'{column}' column does not exist on dataframe"

    with pytest.raises(TypeError) as excep:
        column = '1'
        normalization(df, [column, '2'])
    assert str(excep.value) == f"{column} is not a numerical column and cannot be normalized"

    with pytest.warns(UserWarning, match="2 has no variance, so all values are set on 0"):
        df['2'] = 1
        df_norm, _ = normalization(df)

        assert (df_norm['2'] == 0).all()

def test_one_hot(categorical_data):
    df = categorical_data

    df_one_hot = one_hot(df)

    df_expected = pd.DataFrame({
        '1': ['A', 'B', 'C'],
        '2': ['B', 'B', 'A'],
        '3': ['A', 'C', 'C'],
        '1_A_onehot': [1, 0, 0],
        '1_B_onehot': [0, 1, 0],
        '1_C_onehot': [0, 0, 1],
        '2_B_onehot': [1, 1, 0],
        '2_A_onehot': [0, 0, 1],
        '3_A_onehot': [1, 0, 0],
        '3_C_onehot': [0, 1, 1],
        })

    pd.testing.assert_frame_equal(df_one_hot, df_expected)

    df_one_hot = one_hot(df, ['1', '2'] ,drop=True)
    
    df_expected = pd.DataFrame({
        '3': ['A', 'C', 'C'],
        '1_A_onehot': [1, 0, 0],
        '1_B_onehot': [0, 1, 0],
        '1_C_onehot': [0, 0, 1],
        '2_B_onehot': [1, 1, 0],
        '2_A_onehot': [0, 0, 1],
        })

    pd.testing.assert_frame_equal(df_one_hot, df_expected)

def one_hot_exceptions(categorical_data):
    df = categorical_data

    with pytest.raises(TypeError) as excep:
        df_num = df
        df_num['1'] = 2.5
        one_hot(df, ['1'])
    assert str(excep.value) == "No categorical columns found to be encoded"

    with pytest.raises(TypeError) as excep:
        column = '5'
        one_hot(df, [column])
    assert str(excep.value) == f"'{column}' column does not exist on dataframe"

    with pytest.raises(TypeError) as excep:
        column = '1'
        df_num = df
        df_num[column] = 2.5
        one_hot(df, [column, '2'])
    assert str(excep.value) == f"{column} is not a categorical column, and cannot be encoded."

def test_standardization(numerical_data):
    df = numerical_data

    df_stand, dict = standardization(df, ['2','3','4','5'])

    df_expected = pd.DataFrame({
        '1': ['A', 'B', 'C', 'D'],
        '2': [-1.3174651, 0.43915503, -0.14638501, 1.02469508],
        '3': [-0.75363994, 0.06910399, 1.38939661, -0.70486066],
        '4': [0.90552708, 0.78343354,-0.57994431,-1.10901631],
        '5': [0.44124875,-1.18797741, 1.12009298, -0.37336433],
        })

    dict_expected = {'2':{'mean':3.25, 'std':1.70782513}, '3':{'mean':12.575, 'std':30.7507588}, '4':{'mean':0.85, 'std':4.91426495}, '5':{'mean':0.75, 'std':7.36545993}}

    pd.testing.assert_frame_equal(df_stand, df_expected)
    for column in ['2','3','4','5']:
        assert dict[column] == pytest.approx(dict_expected[column])

def test_standardization_exceptions(numerical_data):
    df = numerical_data

    with pytest.raises(ValueError) as excep:
        column = 'Weight'
        standardization(df, ['Weight'])
    assert str(excep.value) == f"'{column}' column does not exist on dataframe"

    with pytest.raises(TypeError) as excep:
        column = '1'
        standardization(df, [column, '2'])
    assert str(excep.value) == f"{column} is not a numerical column and cannot be standardized"

    with pytest.warns(UserWarning) as excep:
        df['2'] = 1
        df_stand, _ = standardization(df)

    assert str(excep[0].message) == "2 column has (numerically) zero variance, so all values are set to 0"
    assert (df_stand['2'] == 0).all()