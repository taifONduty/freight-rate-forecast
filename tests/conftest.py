import pandas as pd
import pytest

from tests.synthetic import make_loads


@pytest.fixture
def loads() -> pd.DataFrame:
    return make_loads()
