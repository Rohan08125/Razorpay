import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from investigate_exception import ExceptionInvestigator


class FakeFinanceData:
    payment_by_id = {
        "pay_1": {"amount": "100.00"},
    }

    def get_settlement(self, settlement_id):
        return {
            "settlement": {
                "merchant_id": "mer_1",
                "net_amount": "95.00",
                "payment_count": "1",
            },
            "items": [
                {
                    "payment_id": "pay_1",
                    "gross_amount": "110.00",
                    "net_amount": "95.00",
                }
            ],
            "bank_transactions": [
                {"credit_amount": "95.00"},
            ],
        }


def test_amount_mismatch_is_classified_from_payment_gross_amount():
    investigator = ExceptionInvestigator.__new__(ExceptionInvestigator)
    investigator.data = FakeFinanceData()

    result = investigator.investigate_settlement("set_1")

    assert result["status"] == "ESCALATE"
    assert result["root_cause"] == "SETTLEMENT_ITEM_AMOUNT_MISMATCH"
    assert any("gross amount" in finding for finding in result["findings"])
