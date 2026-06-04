import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SemanticBERTClassifier:
    def __init__(self):
        self.local_model_path = "./local_bert_model"
        
        required_files = ["config.json", "tokenizer.json", "model.safetensors"]
        files_exist = all(os.path.exists(os.path.join(self.local_model_path, f)) for f in required_files)
        
        if files_exist:
            print("[INFO] Завантаження автентичного семантичного ядра BERT з локальних файлів...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.local_model_path, local_files_only=True)
                self.model.eval()
                print("[INFO] Двонаправлена модель-трансформер BERT успешно ініціалізована!")
            except Exception as e:
                print(f"[ERROR] Помилка ініціалізації архітектури BERT: {e}")
                self.model = None
        else:
            print("[ERROR] Локальні файли BERT не знайдені в папці './local_bert_model'!")
            self.model = None

    def predict_clickbait_probability(self, cleaned_text: str) -> float:
        """
        Глибокий контекстуальний аналіз прихованої семантики маніпуляцій
        на основі двонаправленої архітектури трансформера BERT.
        """
        if self.model is None or self.tokenizer is None:
            return 0.35
            
        try:
            inputs = self.tokenizer(
                cleaned_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).flatten().tolist()
            
            target_idx = 0
            
            # Динамічна перевірка конфігу на випадок інших назв міток
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'id2label'):
                for idx, label in self.model.config.id2label.items():
                    lbl_low = label.lower()
                    if lbl_low in ['toxic', 'manipulation', 'unsafe']:
                        target_idx = int(idx)
                        break
                    elif lbl_low in ['neutral', 'non-toxic', 'safe']:
                        target_idx = 1 if int(idx) == 0 else 0
                        break
            
            manipulation_probability = probabilities[target_idx]
                
            return float(manipulation_probability)
            
        except Exception as e:
            print(f"[WARNING] Помилка інференсу BERT: {e}")
            return 0.25

if __name__ == "__main__":
    bert_model = SemanticBERTClassifier()
    
    if bert_model.model is not None:
        text_1 = "Ці продажні чиновники знову обікрали народ і зрадили ЗСУ, гнати їх треба!"
        prob_1 = bert_model.predict_clickbait_probability(text_1)
        print(f"\nТест 1: '{text_1}'")
        print(f"Ймовірність маніпуляції: {prob_1:.4f} ({prob_1*100:.1f}%)")
        
        text_2 = "Кабінет Міністрів України затвердив новий бюджет на поточний рік."
        prob_2 = bert_model.predict_clickbait_probability(text_2)
        print(f"\nТест 2: '{text_2}'")
        print(f"Ймовірність маніпуляції: {prob_2:.4f} ({prob_2*100:.1f}%)")