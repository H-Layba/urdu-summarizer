# 📰 Urdu Text Summarizer

![CI](https://github.com/H-Layba/urdu-summarizer/actions/workflows/test.yml/badge.svg)
![Model](https://img.shields.io/badge/HuggingFace-mT5_XLSum-yellow)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Languages](https://img.shields.io/badge/Languages-44-green)

A multilingual text summarizer with full Urdu support — powered by **mT5 trained on XL-Sum**, a dataset of 1 million+ news articles across 44 languages including Urdu. Features a live Gradio demo with adjustable summary length and compression metrics.

---

## 🚀 Live Demo

👉 **[Try it live on HuggingFace Spaces](https://huggingface.co/spaces/H-Layba/urdu-summarizer)**

---

## ✅ Example Output

**Input (Urdu news article):**
> پاکستان میں موسم گرما کے دوران درجہ حرارت میں غیر معمولی اضافہ دیکھا گیا ہے۔ ملک کے مختلف شہروں میں درجہ حرارت 45 ڈگری سینٹی گریڈ سے تجاوز کر گیا ہے۔ موسمیاتی ماہرین کا کہنا ہے کہ یہ صورتحال موسمیاتی تبدیلی کا نتیجہ ہے۔

**Summary:**
> پاکستان میں موسم گرما کے دوران درجہ حرارت میں غیر معمولی اضافہ دیکھا گیا ہے۔

---

## 🏗️ Project Structure

```
urdu-summarizer/
├── app.py                    # Gradio demo (HF Spaces)
├── requirements.txt
├── README.md
├── tests/
│   └── test_summarizer.py    # 3 quality gate tests
└── .github/
    └── workflows/
        └── test.yml          # CI/CD pipeline
```

---

## ⚙️ CI/CD Pipeline

Every push to `main` triggers GitHub Actions:
1. Loads the mT5 model
2. Runs 3 tests — checks summary is shorter than input, non-empty, and contains Urdu characters
3. Fails build if any test fails

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | csebuetnlp/mT5_multilingual_XLSum |
| Training Data | XL-Sum (1M+ multilingual news articles) |
| Languages | 44 including Urdu |
| Demo UI | Gradio |
| Deployment | HuggingFace Spaces |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## 📦 Use the Model

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")

text = "آپ کا اردو متن یہاں لکھیں..."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

outputs = model.generate(inputs["input_ids"], max_length=80, num_beams=4)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)
```

---

## 🗺️ Part of Urdu NLP Suite

- [x] Sentiment Classification → [urdu-sentiment-classifier](https://github.com/H-Layba/urdu-sentiment-classifier)
- [x] Text Summarization ← you are here
- [ ] Question Answering
- [ ] Urdu → English Translation

---

## 👤 Author

**H-Layba** · [HuggingFace](https://huggingface.co/H-Layba) · [GitHub](https://github.com/H-Layba)

---

## 📄 License

Apache 2.0
