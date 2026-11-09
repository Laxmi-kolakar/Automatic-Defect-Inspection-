from core.decisions import make_decision

def test_missing_model_requires_review(): assert make_decision("normal",.99,False)=="REVIEW REQUIRED"
def test_low_confidence_requires_review(): assert make_decision("normal",.55,True,.8)=="REVIEW REQUIRED"
def test_normal_confident_passes(): assert make_decision("normal",.95,True,.8)=="PASS"
def test_defective_confident_fails(): assert make_decision("defective",.91,True,.8)=="FAIL"
def test_unknown_label_review(): assert make_decision("casting-pit",.99,True,.8)=="REVIEW REQUIRED"
