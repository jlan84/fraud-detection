from spellchecker import SpellChecker
spell = SpellChecker()

def spelling(words):
    # not handling empty strings correctly, assigning 1 correctly
    total = 0
    misspelled = 0
    if len(words) == 0:
        return 0
    for word in words:
        if spell[word] == 0:
            misspelled+=1
        total +=1
    if total == 0:
        return 0
    else:
        prop_misspelled = misspelled/total
        return prop_misspelled


df['spelling'] = df.apply(lambda row: spelling(row.desc_text), axis=1)