from core.inference import model_status

def test_model_status_has_expected_keys():
    status=model_status(); assert "available" in status and "path" in status and "version" in status
