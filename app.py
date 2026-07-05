import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model once at startup
MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

EXAMPLES = [
    ["پاکستان میں موسم گرما کے دوران درجہ حرارت میں غیر معمولی اضافہ دیکھا گیا ہے۔ ملک کے مختلف شہروں میں درجہ حرارت 45 ڈگری سینٹی گریڈ سے تجاوز کر گیا ہے۔ موسمیاتی ماہرین کا کہنا ہے کہ یہ صورتحال موسمیاتی تبدیلی کا نتیجہ ہے۔ حکومت نے شہریوں کو احتیاطی تدابیر اختیار کرنے کی ہدایت کی ہے۔"],
    ["پاکستان کرکٹ ٹیم نے آج ایک تاریخی فتح حاصل کی۔ قومی ٹیم نے مضبوط حریف کے خلاف شاندار کھیل کا مظاہرہ کیا۔ بیٹنگ لائن اپ نے بہترین کارکردگی دکھائی اور بولرز نے حریف ٹیم کو کم اسکور پر روک لیا۔ شائقین نے اس فتح کو بہت سراہا۔"],
    ["اسلام آباد میں ایک نئی ٹیکنالوجی کمپنی نے اپنا دفتر کھولا ہے۔ کمپنی مصنوعی ذہانت کے شعبے میں کام کرے گی۔ اس منصوبے سے سینکڑوں نوجوانوں کو روزگار ملے گا۔ حکومت نے اس اقدام کو خوش آئند قرار دیا ہے۔"],
]

def summarize(text, max_length, min_length):
    if not text.strip():
        return "", 0, 0

    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=int(max_length),
            min_length=int(min_length),
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    original_words = len(text.split())
    summary_words = len(summary.split())
    compression = round((1 - summary_words / original_words) * 100) if original_words > 0 else 0

    return summary, original_words, compression


css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&family=Inter:wght@400;500;600&display=swap');

body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'Inter', sans-serif;
}

#title {
    text-align: center;
    padding: 2rem 1rem 1rem;
}

#title h1 {
    font-size: 2rem;
    font-weight: 600;
    color: #e4e4f0;
    margin-bottom: 0.4rem;
}

#title p {
    color: #6b6b8a;
    font-size: 0.9rem;
}

.urdu-box textarea, .urdu-box .output-class {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    font-size: 1.2rem !important;
    direction: rtl !important;
    text-align: right !important;
    line-height: 2.2 !important;
    background: #12121e !important;
    border: 1px solid #2a2a3e !important;
    color: #e4e4f0 !important;
    border-radius: 10px !important;
}

.urdu-box textarea:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.15) !important;
}

button.primary {
    background: #10b981 !important;
    border: none !important;
    color: white !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
}

button.primary:hover {
    background: #059669 !important;
}

#stats-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    padding: 0.5rem 0;
}

footer { display: none !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:

    gr.HTML("""
        <div id="title">
            <h1>📰 Urdu Text Summarizer</h1>
            <p>اردو خبروں اور مضامین کا خلاصہ — Multilingual mT5 trained on XL-Sum</p>
        </div>
    """)

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="اصل متن — Original Text",
                placeholder="یہاں اردو متن پیسٹ کریں...",
                lines=8,
                elem_classes="urdu-box"
            )
            with gr.Row():
                max_len = gr.Slider(
                    label="Max summary length (tokens)",
                    minimum=30,
                    maximum=150,
                    value=80,
                    step=10
                )
                min_len = gr.Slider(
                    label="Min summary length (tokens)",
                    minimum=10,
                    maximum=60,
                    value=20,
                    step=5
                )
            summarize_btn = gr.Button("خلاصہ کریں — Summarize", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="خلاصہ — Summary",
                lines=5,
                interactive=False,
                elem_classes="urdu-box"
            )
            with gr.Row():
                word_count = gr.Number(label="Original word count", interactive=False)
                compression = gr.Number(label="Compression %", interactive=False)

    gr.Examples(
        examples=EXAMPLES,
        inputs=input_text,
        label="مثالیں — Try these Urdu news examples"
    )

    gr.HTML("""
        <div style="text-align:center; padding:1.5rem; color:#3a3a5a; font-size:0.8rem;">
            Built by <a href="https://huggingface.co/H-Layba" style="color:#10b981">H-Layba</a> ·
            Model: mT5 XL-Sum Multilingual · Supports Urdu, English & 42 other languages
        </div>
    """)

    summarize_btn.click(
        fn=summarize,
        inputs=[input_text, max_len, min_len],
        outputs=[output_text, word_count, compression]
    )
    input_text.submit(
        fn=summarize,
        inputs=[input_text, max_len, min_len],
        outputs=[output_text, word_count, compression]
    )

demo.launch()
