import sqlite3
import re
from linguistic_processor import run_full_linguistic_pipeline

DB_NAME = "clickbait_detector.db"

def get_knowledge_base():
    """Вивантаження правил та маркерів з бази знань SQLite"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT lemma, weight FROM linguistic_markers")
    markers = {row[0]: row[1] for row in cursor.fetchall()}
    
    cursor.execute("SELECT rule_code, threshold_value, weight_penalty FROM production_rules")
    rules = {row[0]: {'threshold': row[1], 'penalty': row[2]} for row in cursor.fetchall()}
    
    conn.close()
    return markers, rules


def evaluate_expert_rules(linguistic_results: dict) -> dict:
    """
    Механізм логічного виведення на основі продукційних правил 
    та математичного апарату Наївного Баєсівського класифікатора.
    """
    markers_db, rules_db = get_knowledge_base()
    
    activated_rules = []
    found_triggers = []
    
    applied_weights = []
    
    caps_rule = rules_db.get('RULE_CAPS')
    if caps_rule and linguistic_results['caps_ratio'] >= caps_rule['threshold']:
        applied_weights.append(caps_rule['penalty'])
        activated_rules.append(f"RULE_CAPS (Ratio: {linguistic_results['caps_ratio']:.2f})")
   
    punct_rule = rules_db.get('RULE_EXCESSIVE_PUNCT')
    if punct_rule and linguistic_results['has_excessive_exclamation']:
        applied_weights.append(punct_rule['penalty'])
        activated_rules.append("RULE_EXCESSIVE_PUNCT (!!!)")
        
    ellipsis_rule = rules_db.get('RULE_ELLIPSIS')
    if ellipsis_rule and linguistic_results['has_ellipsis_end']:
        applied_weights.append(ellipsis_rule['penalty'])
        activated_rules.append("RULE_ELLIPSIS (...)")
        
    for lemma in linguistic_results['pymorphy_lemmas']:
        if lemma in markers_db:
            trigger_weight = markers_db[lemma]
            applied_weights.append(trigger_weight)
            found_triggers.append(f"{lemma} ({trigger_weight})")
            
    if applied_weights:
        odds_product = 1.0
        for w in applied_weights:
            w_clipped = max(min(w, 0.99), 0.01)
            odds_product *= (w_clipped / (1.0 - w_clipped))
        
        final_score = odds_product / (1.0 + odds_product)
    else:
        final_score = 0.0
        
    final_score = round(min(max(final_score, 0.0), 1.0), 4)
    
    is_definitive = final_score >= 0.75
    
    return {
        'expert_score': final_score,
        'activated_rules': activated_rules,
        'found_triggers': found_triggers,
        'stop_inference': is_definitive  
    }

if __name__ == "__main__":
    test_headline = "ТЕРМІНОВО!!! Окупанти готують новий НАСТУП на місто, евакуація вже почалась... Читати деталі: https://bit.ly/fake_link"
    ling_res = run_full_linguistic_pipeline(test_headline)
    expert_res = evaluate_expert_rules(ling_res)
    
    print(f"Активовані правила: {expert_res['activated_rules']}")
    print(f"Знайдені маркер-слова: {expert_res['found_triggers']}")
    print(f"Фінальний експертний бал (Баєс): {expert_res['expert_score']:.2f}")
    print(f"Зупинити виведення: {expert_res['stop_inference']}")