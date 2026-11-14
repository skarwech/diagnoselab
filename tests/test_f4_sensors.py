"""F4 sensor model: reading = truth + bias + sigma * N(0, 1), keyed noise (plan section 8, F4)."""

from diagnoselab.eval.validate_sim import f4


def test_noise_statistics_on_20_seeds() -> None:
    """F4 valid-when: all statistical tests (mean, variance, lag-1 autocorrelation) pass at p > 0.01 on
    20 seeds with N = 1000 reads each; same key -> same value; bias independent of operating point."""
    m = f4()
    assert min(m["p_mean"]) > 0.01 and min(m["p_var"]) > 0.01 and min(m["p_acf"]) > 0.01
    assert m["ks_pooled_p"] > 0.01
    assert m["repeat_ok"] == 20 and m["bias_op_ok"] == 20
