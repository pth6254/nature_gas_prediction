import numpy as np
import pandas as pd
import pytest

from src.ml_experiment import FAMILIES,pipeline_for


@pytest.mark.parametrize("family",FAMILIES)
def test_ml_preprocessing_is_fold_local_and_predictions_are_finite(family):
    x=pd.DataFrame({"a":np.arange(40,dtype=float),"b":[np.nan]*5+list(np.arange(35,dtype=float))})
    y=np.sin(np.arange(40)/8)/10
    model=pipeline_for(family)
    model.fit(x,y)
    mean=model.named_steps["scaler"].mean_.copy()
    result=model.predict(pd.DataFrame({"a":[10000.],"b":[np.nan]}))
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(mean,model.named_steps["scaler"].mean_)
    assert model.named_steps["imputer"].statistics_[0]==19.5
