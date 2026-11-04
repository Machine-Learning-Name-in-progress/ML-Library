import pytest
import pandas as pd
from empty_values_indicate import empty_values_indicate

def test_empty_values_indicate():
    dados = {
    'Nome': ['Ana', 'Bruno', 'Carlos', 'Diana', 'Eduardo'],
    'Idade': [25, pd.NA, 30, 22, pd.NA],
    'Cidade': ['São Paulo', 'Rio de Janeiro', 'Salvador', 'Belo Horizonte', 'Curitiba'],
    'Pontuação': [8.5, 7.0, 9.2, pd.NA, 6.5],
    }
    
    df = pd.DataFrame(dados)
    df_ind = empty_values_indicate(df)

    expected_column = pd.Series([0, 1, 0, 1, 1], name="indicate")

    pd.testing.assert_series_equal(expected_column, df_ind['indicate'])