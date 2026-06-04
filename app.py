import gradio as ui
from linguistic_processor import run_full_linguistic_pipeline
from expert_system import evaluate_expert_rules
from semantic_core import SemanticBERTClassifier

bert_classifier = SemanticBERTClassifier()

def process_news_pipeline(input_text):
    if not input_text.strip():
        return "Введіть заголовок для аналізу.", "Зелена зона", "0.0%"

    linguistic_results = run_full_linguistic_pipeline(input_text)
    
    expert_results = evaluate_expert_rules(linguistic_results)
    
    expert_score = expert_results['expert_score']
    activated_rules = expert_results['activated_rules']
    found_triggers = expert_results['found_triggers']
    
    final_score = expert_score
    analysis_route = "Експертний контур: достатньо маркерів у базі знань"
    if not expert_results['stop_inference'] and bert_classifier.model is not None:
        cleaned_text = linguistic_results.get('cleaned_text', input_text)
        bert_score = bert_classifier.predict_clickbait_probability(cleaned_text)
        
        if expert_score > 0:
            final_score = (expert_score * 0.3) + (bert_score * 0.7)
            analysis_route = "Гібридний контур: ккспертна система + локальне нейроядро BERT"
        else:
            final_score = bert_score
            analysis_route = "Глибинний семантичний аналіз: локальна модель BERT"

    if final_score >= 0.75:
        color_zone = "🔴 Червона зона (блокування)"
    elif final_score >= 0.40:
        color_zone = "🟡 Жовта зона (попередження)"
    else:
        color_zone = "🟢 Зелена зона (безпечно)"

    report = f"• Очищений text: '{linguistic_results.get('cleaned_text', input_text)}'\n"
    report += f"• Коефіцієнт CapsLock: {linguistic_results.get('caps_ratio', 0)*100:.1f}%\n"
    report += f"• Активовані продукційні правила: {', '.join(activated_rules) if activated_rules else 'Не виявлено'}\n"
    report += f"• Знайдені слова-маркери: {', '.join(found_triggers) if found_triggers else 'Не знайдено'}\n"
    report += f"• Задіяний аналітичний контур: {analysis_route}\n"
    report += f"• Зупинка за лінгвістичними правилами: {'Так (BERT пропущено)' if expert_results['stop_inference'] else 'Ні (потрібен аналіз контексту)'}\n"
    report += f"Фінальний розрахований коефіцієнт загрози: {final_score:.4f}"

    return report, color_zone, f"{final_score * 100:.1f}%"
\
with ui.Blocks(title="Система детекції маніпуляцій", theme=ui.themes.Soft()) as demo:
    ui.Markdown("Інтелектуальна система виявлення медіа-маніпуляцій та клікбейту")
    
    with ui.Row():
        with ui.Column(scale=2):
            input_box = ui.Textbox(
                label="Вхідний новинний заголовок для верифікації", 
                placeholder="Вставте або напишіть текст сюди...",
                lines=3
            )
            submit_btn = ui.Button("Запустити аналіз", variant="primary")
            
            ui.Examples(
                examples=[
                    ["ТЕРМІНОВО!!! Окупанти готують новий НАСТУП на місто, евакуація вже почалась..."],
                    ["Кабінет Міністрів України затвердив новий бюджет на поточний рік."],
                    ["Львівська політехніка оголосила про початок прийому документів для абітурієнтів."]
                ],
                inputs=input_box
            )
            
        with ui.Column(scale=3):
            with ui.Row():
                percentage_out = ui.Label(label="Ймовірність маніпуляції")
                zone_out = ui.Label(label="Вердикт системи")
            
            log_out = ui.Textbox(label="Технічний звіт лінгвістичних модулів та ядра BERT", lines=10)

    submit_btn.click(
        fn=process_news_pipeline,
        inputs=input_box,
        outputs=[log_out, zone_out, percentage_out]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)