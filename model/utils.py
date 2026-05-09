lang_codes = {
    'fr': 'French',
    'en': 'English',
    'nl': 'Dutch',
    'de': 'German',
    'es': 'Spanish',
}

model_paths = {
    't5': 'google/flan-t5', 
    'madlad': 'jbochi/madlad400-3b-mt',
    'gemma': 'google/translategemma-4b-it',
}

valid_pairs = {('en', 'de'),
               ('en', 'es'),
               ('fr', 'en'),
               ('nl', 'en'),
            #    ('all', 'all'),
}

flores_lang_id = {
   'en': 'eng_Latn',
   'fr': 'fra_Latn',
   'de': 'deu_Latn',
   'nl': 'nld_Latn',
   'es': 'spa_Latn',
}