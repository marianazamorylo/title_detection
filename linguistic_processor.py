import re
import pymorphy3
import stanza

morph = pymorphy3.MorphAnalyzer(lang='uk')

try:
    nlp_stanza = stanza.Pipeline(lang='uk', processors='tokenize,mwt,pos,lemma', verbose=False)
except Exception:
    print("Завантаження мовного пакету Stanza для української мови...")
    stanza.download('uk')
    nlp_stanza = stanza.Pipeline(lang='uk', processors='tokenize,mwt,pos,lemma', verbose=False)


def clean_text(text: str) -> str:
    """
    Очищення вхідного рядка від посилань, системних символів та дедуплікація пробілів.
    Регістр символів залишається незмінним.
    """
    if not text:
        return ""
    text = re.sub(r'https?://\S+|www\.\S+|bit\.ly/\S+|t\.me/\S+', '', text)
    text = re.sub(r'[^\w\s\.,!\?\-:"\']', '', text)
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()


def calculate_caps_ratio(text: str) -> float:
    """Розрахунок відсотка літер у верхньому регістрі (CapsLock) відносно всіх літер."""
    letters = re.findall(r'[a-zA-Zа-яА-ЯіІїЇєЄґҐ]', text)
    if not letters:
        return 0.0
    caps_count = sum(1 for char in letters if char.isupper())
    return caps_count / len(letters)


def check_punctuation_patterns(text: str) -> dict:
    """Виявлення аномальних пунктуаційних маркерів нагнітання паніки чи клікбейту."""
    patterns = {'excessive_exclamation': False, 'ellipsis_end': False}
    if re.search(r'(!{2,})|(\?!+)|(!+\?)', text):
        patterns['excessive_exclamation'] = True
    if text.strip().endswith('...'):
        patterns['ellipsis_end'] = True
    return patterns


def tokenize_and_lemmatize_pymorphy(text: str) -> list:
    """Швидка токенізація та лематизація через PyMorphy3 з відсіканням стоп-слів."""
    tokens = re.findall(r'[a-zA-Zа-яА-ЯіІїЇєЄґҐі\-]+', text.lower())
    stop_words = {
        'і', 'та', 'а', 'але', 'чи', 'в', 'у', 'на', 'за', 'до', 'під', 'про', 'як', 'що', 'це', 'не', 'по',
        'з', 'для', 'то', 'все', 'й', 'від', 'проти', 'було', 'вже', 'просто', 'якщо', 'через', 'його'
    }
    lemmas = []
    for token in tokens:
        if token in stop_words or len(token) <= 2:
            continue
        parsed = morph.parse(token)[0]
        lemmas.append(parsed.normal_form)
    return lemmas


def execute_stanza_pipeline(text: str) -> list:
    """Глибока контекстуальна лематизація через нейромережевий конвеєр Stanza."""
    doc = nlp_stanza(text)
    stanza_lemmas = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.pos not in ['PUNCT', 'SYM', 'CCONJ', 'ADP']:
                stanza_lemmas.append(word.lemma.lower() if word.lemma else word.text.lower())
    return stanza_lemmas


def run_full_linguistic_pipeline(raw_text: str) -> dict:
    """Головний метод-інтегратор лінгвістичного процесора верхнього рівня."""
    cleaned = clean_text(raw_text)
    caps_ratio = calculate_caps_ratio(cleaned)
    punct = check_punctuation_patterns(cleaned)
    
    pymorphy_lemmas = tokenize_and_lemmatize_pymorphy(cleaned)
    stanza_lemmas = execute_stanza_pipeline(cleaned)
    
    return {
        'raw_text': raw_text,
        'cleaned_text': cleaned,
        'caps_ratio': caps_ratio,
        'has_excessive_exclamation': punct['excessive_exclamation'],
        'has_ellipsis_end': punct['ellipsis_end'],
        'pymorphy_lemmas': pymorphy_lemmas,
        'stanza_lemmas': stanza_lemmas
    }


if __name__ == "__main__":
    test_headline = "ТЕРМІНОВО!!! Окупанти готують новий НАСТУП на місто, евакуація вже почалась... Читати деталі: https://bit.ly/fake_link"
    

    res = run_full_linguistic_pipeline(test_headline)
    print(f"Вхідний рядок: {res['raw_text']}\n")
    print(f"Очищений рядок: {res['cleaned_text']}")
    print(f"Показник CapsLock: {res['caps_ratio']:.2f}")
    print(f"Клікбейтні знаки (!!!): {res['has_excessive_exclamation']}")
    print(f"Трикрапка в кінці (...): {res['has_ellipsis_end']}")
    print(f"Леми PyMorphy3 (для правил): {res['pymorphy_lemmas']}")
    print(f"Леми Stanza (для BERT): {res['stanza_lemmas']}")