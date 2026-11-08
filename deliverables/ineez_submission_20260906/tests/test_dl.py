import numpy as np
import pandas as pd
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("neuralforecast")
from src.dl_experiment import MODELS,prepare,split_origins,train_one


def data():
    price=np.exp(np.sin(np.arange(140)/20)*.2+1)
    x=pd.DataFrame({"production_level":np.arange(140,dtype=float)})
    return price,x


def test_inner_validation_is_purged_and_known():
    train,val=split_origins(120)
    assert train[-1]+3<=val[0]
    assert val[-1]+3==120
    assert len(val)==12


def test_future_data_cannot_change_inputs_or_scaling():
    price,x=data()
    original=prepare(price,x,120,["production_level"])
    price[121:]=999
    x.loc[121:]=999
    changed=prepare(price,x,120,["production_level"])
    assert original[4:6]==changed[4:6]
    for idx in [0,1]:
        torch.testing.assert_close(original[idx][1],changed[idx][1])
        for key in ["insample_y","hist_exog"]:
            torch.testing.assert_close(original[idx][0][key],changed[idx][0][key])
    # Validation outcomes must not contribute to normalization or fitting tensors.
    price[original[-1]+1:121]*=2
    different=prepare(price,x,120,["production_level"])
    assert original[4:6]==different[4:6]
    torch.testing.assert_close(original[0][1],different[0][1])


@pytest.mark.parametrize("name",MODELS)
def test_official_networks_generate_three_finite_prices(name):
    torch.set_num_threads(1)
    price,x=data()
    prediction,diag=train_one(prepare(price,x,120,["production_level"]),name,42,2)
    assert prediction.shape==(3,)
    assert np.isfinite(prediction).all() and (prediction>0).all()
    assert diag["validation_last_target"]==120
