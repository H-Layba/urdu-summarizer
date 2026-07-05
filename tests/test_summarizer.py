from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "csebuetnlp/mT5_multilingual_XLSum"

def get_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def summarize(text, tokenizer, model, max_length=80, min_length=20):
    inputs = tokenizer(text.strip(), return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def test_summary_is_shorter_than_input():
    """Summary must be shorter than the original text."""
    tokenizer, model = get_model()
    text = """پاکستان میں موسم گرما کے دوران درجہ حرارت میں غیر معمولی اضافہ دیکھا گیا ہے۔
    ملک کے مختلف شہروں میں درجہ حرارت 45 ڈگری سینٹی گریڈ سے تجاوز کر گیا ہے۔
    موسمیاتی ماہرین کا کہنا ہے کہ یہ صورتحال موسمیاتی تبدیلی کا نتیجہ ہے۔"""
    summary = summarize(text, tokenizer, model)
    assert len(summary) < len(text), "Summary should be shorter than input"

def test_summary_is_not_empty():
    """Summary must not be empty."""
    tokenizer, model = get_model()
    text = "پاکستان کرکٹ ٹیم نے آج ایک تاریخی فتح حاصل کی۔ قومی ٹیم نے مضبوط حریف کے خلاف شاندار کھیل کا مظاہرہ کیا۔"
    summary = summarize(text, tokenizer, model)
    assert len(summary.strip()) > 0, "Summary should not be empty"

def test_summary_contains_urdu():
    """Summary should contain Urdu characters."""
    tokenizer, model = get_model()
    text = "اسلام آباد میں ایک نئی ٹیکنالوجی کمپنی نے اپنا دفتر کھولا ہے۔ کمپنی مصنوعی ذہانت کے شعبے میں کام کرے گی۔ اس منصوبے سے سینکڑوں نوجوانوں کو روزگار ملے گا۔"
    summary = summarize(text, tokenizer, model)
    urdu_chars = any('\u0600' <= c <= '\u06FF' for c in summary)
    assert urdu_chars, "Summary should contain Urdu characters"
