import numpy as np
import pandas as pd
import pytest

from ratecast.pipeline import align_to_template


def test_submission_follows_template_order():
    template = pd.DataFrame({"load_id": ["TE-3", "TE-1", "TE-2"], "predicted_rate": np.nan})
    scored = pd.DataFrame({
        "load_id": ["TE-1", "TE-2", "TE-3", "TE-9"],
        "predicted_rate": [100.004, 200.0, 300.0, 1.0],
    })

    submission = align_to_template(template, scored)

    assert list(submission.columns) == ["load_id", "predicted_rate"]
    assert submission["load_id"].tolist() == ["TE-3", "TE-1", "TE-2"]
    assert submission["predicted_rate"].tolist() == [300.0, 100.0, 200.0]
    with pytest.raises(ValueError, match="not scored"):
        align_to_template(template, scored[scored["load_id"] != "TE-2"])
