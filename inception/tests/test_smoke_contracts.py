from inception.smoke_runner import validate_result_contract


def test_validate_result_contract_minimal_passes():
    r = validate_result_contract(
        {
            "AnalysisPack": {"Ticker": "_T_", "Last": {"Close": 1.0}},
            "DashboardSummaryPack": {"schema": "DashboardSummaryPack.v1"},
        }
    )
    assert r.ok, f"unexpected errors: {r.errors}"
