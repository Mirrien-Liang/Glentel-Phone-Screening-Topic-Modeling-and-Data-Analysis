from spellchecker import SpellChecker
import pandas as pd

# Spellcheck function (Unused due to Poor Reliability)
def spellcheck(text, language="en"):
    """
    Function to spellcheck the text in its detected language.
    """
    # If text is empty or None, return it as is
    if not text or pd.isna(text):
        return text
    
    # If the given language is not supported by the spellchecker, default to English
    try:
        spell = SpellChecker(language=language)
    except ValueError:
        spell = SpellChecker()

    try:
        corrected_text = []
        for word in text.split():
            # Only correct the word if it is misspelled
            if word in spell.unknown([word]):
                corrected_word = spell.correction(word)
                corrected_text.append(corrected_word)
            else:
                corrected_text.append(word)

        return " ".join(corrected_text)
    
    # If empty after corrected, try French
    except TypeError:
        # If already trying French, return the original text
        if language == 'fr':
            return text
        
        try:
            return spellcheck(text, language="fr")
        
        # If still empty after trying French, return the original text
        except TypeError:
            return text


if __name__ == "__main__":
    print(spellcheck("hoping for fulltime work new place want to bbq still with girlfriend healthy happy steady and consistenat job	"))
    print(spellcheck("year in money isnt the good enough area getting more dangerous needd a change	"))
    print(spellcheck("taken u conger for week	"))
    