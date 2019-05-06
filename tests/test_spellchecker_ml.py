from spellchecker_ml.spellchecker_ml import SpellCheckerML

def test_model_saving():
    try:
        spellchecker = SpellCheckerML()
        spellchecker.train("Hello there, this isn't very much text")
        spellchecker.correction("Hello", "friend")
        assert True
    except:
        assert False

def test_model_correction():
    spellchecker = SpellCheckerML()
    assert "is" == spellchecker.correction("There", "is")
