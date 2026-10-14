import numpy as np

from egopred.openscene import metadata, samples
from egopred.trajectory import FUT, HIST

from .test_metadata import fake_log


def test_log_samples_with_and_without_token_filter():
    df = metadata.compact_log(fake_log("2021.05.12.22.00.38_veh-35_00001_00100", n=20))
    all_s = samples.log_samples(df, None)
    assert len(all_s) == 20 - (HIST - 1) - FUT
    tok = df["token"].iloc[5]
    s = samples.log_samples(df, {tok, df["token"].iloc[0], "not-there"})  # frame 0 has no history
    assert s["sample_id"].tolist() == [tok]
    assert s["hist_token_3"].iloc[0] == tok and s["hist_token_0"].iloc[0] == df["token"].iloc[2]
    np.testing.assert_allclose(samples.future(s)[0, :, 0], 2.5 * np.arange(1, FUT + 1))
    assert s["scenario"].iloc[0] == "cruise" and s["command"].iloc[0] == 1


def test_drive_id_groups_segments_of_one_drive():
    a = samples.drive_id("2021.05.12.22.00.38_veh-35_01008_01518")
    b = samples.drive_id("2021.05.12.22.00.38_veh-35_02000_02400")
    assert a == b == "2021.05.12.22.00.38_veh-35"


def test_val_drives_are_city_stratified_and_deterministic():
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for city, n_drives in [("lv", 60), ("bos", 15), ("pit", 12), ("sg", 10)]:
        for d in range(n_drives):
            rows += [(f"{city}-drive{d}", city)] * int(rng.integers(50, 400))
    df = pd.DataFrame(rows, columns=["drive", "city"])
    val = samples.assign_val_drives(df["drive"], df["city"])
    assert val == samples.assign_val_drives(df["drive"], df["city"])
    share = df.assign(v=df["drive"].isin(val)).groupby("city")["v"].mean()
    assert ((share >= 0.10) & (share < 0.30)).all()  # every city represented, around 10%
