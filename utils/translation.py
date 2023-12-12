from translate import Translator
import pandas as pd

"""
Translator Module

First, copy (and overwrite) the `mymemory_translated.py` file from the `utils` folder to the `site-packages` folder of your Python installation:
    $PYTHON_PATH$/site-packages/translate/providers/mymemory_translated.py
"""

def translate(translator, text):
    """
    Translate text to the target language, assuming it's not already in that language.
    """
    # If text is empty or None, return it as is
    if not text or pd.isna(text):
        return text

    # Always treat as French and always translate to English
    # If the text is already in English, it will automatically return what it is
    try:
        t_text = translator.translate(text)

    except Exception as e:
        # print(f"Error in translation: {e}")
        return text # Still return the raw text in cases of failure
    
    if "query" in t_text.lower():
        # print(f"Error in translation: The text was too long for translation.")
        return text # Still return the raw text in cases of failure
    
    if "mymemory" in t_text.lower():
        return translate(translator,text)

    return t_text

if __name__ == "__main__":
    proxies = []
    
    # Loop through 1 to 1000
    for i in range(1, 10000):
        proxies.append(f"your-proxy-here")
    

    translator = Translator(
        from_lang="fr", # Hard-coded to French
        to_lang='en',
        proxies=proxies
    )
    
    print(translate(translator, "Bonjour, je suis un texte à traduire. Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire.Bonjour, je suis un texte à traduire."))
    # print(translate(translator, "Hello, I am a text to translate."))
    # print(translate(translator, "Gérant de magasin"))