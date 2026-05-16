# ══════════════════════════════════════════════════════════════════
# SpamShield AI — Complete Training Notebook (Google Colab)
# ══════════════════════════════════════════════════════════════════
# Instructions:
#   1. Open Google Colab: https://colab.research.google.com
#   2. Go to Runtime → Change runtime type → T4 GPU
#   3. Copy each section below into a separate Colab cell
#   4. Run cells sequentially (Shift+Enter)
# ══════════════════════════════════════════════════════════════════

# %%
# ═══════════════════════════════════════════════
# CELL 1: Clone Repo & Install Dependencies
# ═══════════════════════════════════════════════
# Expected time: ~3-5 minutes

!git clone https://github.com/Aryan63-Prime/Spam-Detection-via-NLP.git
%cd Spam-Detection-via-NLP

# Install core dependencies (skip heavy packages we install separately)
!pip install -q scikit-learn pandas numpy xgboost lightgbm nltk regex
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers datasets accelerate tokenizers
!pip install -q shap lime matplotlib seaborn plotly

# Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
# ═══════════════════════════════════════════════
# CELL 2: Download & Load Dataset
# ═══════════════════════════════════════════════
# Expected time: ~30 seconds

import sys, os
sys.path.insert(0, os.getcwd())

from datasets.download import DatasetDownloader
from datasets.loader import DatasetLoader

# Download UCI SMS Spam Collection (5,574 messages)
downloader = DatasetDownloader()
dataset_path = downloader.download_sms_spam()

# Load and split: 70% train, 10% val, 20% test (stratified)
loader = DatasetLoader()
df = loader.load_sms_spam(dataset_path)
split = loader.create_split(df, test_size=0.2, val_size=0.1)

print("\n📊 Dataset Split Summary:")
for name, stats in split.summary().items():
    print(f"  {name}: {stats}")

print(f"\n📝 Sample messages:")
for i in range(3):
    label = "SPAM" if split.y_train.iloc[i] == 1 else "HAM"
    print(f"  [{label}] {split.X_train.iloc[i][:80]}...")

# %%
# ═══════════════════════════════════════════════
# CELL 3: Preprocess All Text
# ═══════════════════════════════════════════════
# Expected time: ~30-60 seconds
# Runs 12-step NLP pipeline: lowercase → URL removal → HTML →
# emoji → unicode → tokenize → stopwords → lemmatize → slang →
# hinglish → regex → spell correction

from ml.preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline()

print("🔄 Preprocessing training texts...")
train_texts = pipeline.process_batch_to_texts(split.X_train.tolist())
print("🔄 Preprocessing validation texts...")
val_texts = pipeline.process_batch_to_texts(split.X_val.tolist())
print("🔄 Preprocessing test texts...")
test_texts = pipeline.process_batch_to_texts(split.X_test.tolist())

# Show before/after
print("\n📝 Preprocessing Examples:")
for i in range(3):
    print(f"  BEFORE: {split.X_train.iloc[i][:60]}")
    print(f"  AFTER:  {train_texts[i][:60]}")
    print()

# %%
# ═══════════════════════════════════════════════
# CELL 4: TF-IDF Feature Extraction
# ═══════════════════════════════════════════════
# Expected time: ~5 seconds

from ml.features.tfidf import TfidfFeatureExtractor

tfidf = TfidfFeatureExtractor(max_features=50000, ngram_range=(1, 2))

X_train_tfidf = tfidf.fit_transform(train_texts)
X_val_tfidf = tfidf.transform(val_texts)
X_test_tfidf = tfidf.transform(test_texts)

print(f"✅ TF-IDF Feature Matrices:")
print(f"  Train: {X_train_tfidf.shape}")
print(f"  Val:   {X_val_tfidf.shape}")
print(f"  Test:  {X_test_tfidf.shape}")

# Save vectorizer
tfidf.save("tfidf_spam")
print("💾 TF-IDF vectorizer saved.")

# Top features
print("\n🔝 Top 20 TF-IDF features:")
for feat, score in tfidf.get_top_features(20):
    print(f"  {feat}: {score:.4f}")

# %%
# ═══════════════════════════════════════════════
# CELL 5: Train ALL Traditional ML Models
# ═══════════════════════════════════════════════
# Expected time: ~1-3 minutes total
# Models: Naive Bayes, Logistic Regression, SVM,
#          Random Forest, XGBoost, LightGBM

from ml.models.traditional.classifiers import TRADITIONAL_MODEL_REGISTRY, create_traditional_model
from ml.evaluation.metrics import ModelEvaluator
import time

evaluator = ModelEvaluator()
traditional_results = []
traditional_models = []

print("=" * 60)
print("🏋️ TRAINING TRADITIONAL ML MODELS")
print("=" * 60)

for name in TRADITIONAL_MODEL_REGISTRY.keys():
    print(f"\n▶ Training: {name}")
    start = time.perf_counter()

    try:
        model = create_traditional_model(name)
        model.train(X_train_tfidf, split.y_train, X_val=X_val_tfidf, y_val=split.y_val)
        traditional_models.append(model)

        # Evaluate
        result = evaluator.evaluate(model, X_test_tfidf, split.y_test)
        traditional_results.append(result)

        elapsed = time.perf_counter() - start
        print(f"  ✅ {name} — F1: {result.f1:.4f} | Acc: {result.accuracy:.4f} | Time: {elapsed:.2f}s")

        # Save model
        model.save()
        print(f"  💾 Saved to models/{name}/")

    except Exception as e:
        print(f"  ❌ Failed: {e}")

# Print benchmark table
print("\n")
evaluator.print_benchmark_table(traditional_results)

# Save benchmark
from ml.config import MODELS_DIR
evaluator.save_results(traditional_results, MODELS_DIR / "benchmark_traditional.json")

# %%
# ═══════════════════════════════════════════════
# CELL 6: Build Vocabulary for Deep Learning
# ═══════════════════════════════════════════════
# Expected time: ~5 seconds
# Converts text to integer sequences for PyTorch models

import numpy as np
from collections import Counter

def build_vocab(texts, max_vocab=30000):
    """Build word→index vocabulary from training texts."""
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    # Reserve 0=PAD, 1=UNK
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)

    print(f"📖 Vocabulary size: {len(vocab):,} words")
    return vocab

def texts_to_sequences(texts, vocab, max_len=256):
    """Convert texts to padded integer sequences."""
    sequences = []
    for text in texts:
        tokens = text.split()
        seq = [vocab.get(t, 1) for t in tokens[:max_len]]  # 1 = UNK
        # Pad to max_len
        seq = seq + [0] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences, dtype=np.int64)

# Build vocab from training data only (no data leakage)
vocab = build_vocab(train_texts, max_vocab=30000)

# Convert all splits to sequences
X_train_seq = texts_to_sequences(train_texts, vocab, max_len=256)
X_val_seq = texts_to_sequences(val_texts, vocab, max_len=256)
X_test_seq = texts_to_sequences(test_texts, vocab, max_len=256)

print(f"✅ Sequence shapes: train={X_train_seq.shape}, val={X_val_seq.shape}, test={X_test_seq.shape}")

# %%
# ═══════════════════════════════════════════════
# CELL 7: Train ALL Deep Learning Models
# ═══════════════════════════════════════════════
# Expected time: ~5-10 minutes total (on T4 GPU)
# Models: TextCNN, LSTM, BiLSTM, GRU

from ml.models.deep_learning.models import DeepLearningModel

dl_architectures = ["text_cnn", "lstm", "bilstm", "gru"]
dl_results = []
dl_models = []

print("=" * 60)
print("🧠 TRAINING DEEP LEARNING MODELS")
print("=" * 60)

for arch in dl_architectures:
    print(f"\n▶ Training: {arch.upper()}")
    start = time.perf_counter()

    try:
        model = DeepLearningModel(
            architecture=arch,
            vocab_size=len(vocab),
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            batch_size=32,
            epochs=15,
            learning_rate=1e-3,
            patience=3,
        )

        metrics = model.train(
            X_train_seq, split.y_train.values,
            X_val=X_val_seq, y_val=split.y_val.values,
        )
        dl_models.append(model)

        # Evaluate
        y_pred = model.predict(X_test_seq)
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(split.y_test, y_pred)
        f1 = f1_score(split.y_test, y_pred)

        elapsed = time.perf_counter() - start
        print(f"  ✅ {arch.upper()} — F1: {f1:.4f} | Acc: {acc:.4f} | Time: {elapsed:.1f}s")

        dl_results.append({"model": arch, "accuracy": acc, "f1": f1, "time": elapsed})

        # Save model
        model.save()
        print(f"  💾 Saved to models/{arch}/")

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

# Print DL results
print("\n" + "=" * 60)
print("DEEP LEARNING BENCHMARK RESULTS")
print("=" * 60)
print(f"{'Model':<15} | {'Acc':>7} | {'F1':>7} | {'Time':>8}")
print("-" * 45)
for r in sorted(dl_results, key=lambda x: x["f1"], reverse=True):
    print(f"{r['model']:<15} | {r['accuracy']:>7.4f} | {r['f1']:>7.4f} | {r['time']:>7.1f}s")

# %%
# ═══════════════════════════════════════════════
# CELL 8: Train Transformer (DistilBERT)
# ═══════════════════════════════════════════════
# Expected time: ~5-10 minutes on T4 GPU
# Using DistilBERT (best speed/accuracy tradeoff)
# Change model_name to "bert" or "roberta" for higher accuracy

from ml.models.transformers.classifier import TransformerSpamClassifier

print("=" * 60)
print("🤖 TRAINING TRANSFORMER: DistilBERT")
print("=" * 60)

transformer_model = TransformerSpamClassifier(
    model_name="distilbert",      # Options: "bert", "distilbert", "roberta", "deberta"
    max_length=256,
    batch_size=16,
    epochs=5,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
)

# Train on raw texts (transformer has its own tokenizer)
train_result = transformer_model.train(
    X_train=split.X_train.tolist(),
    y_train=split.y_train.tolist(),
    X_val=split.X_val.tolist(),
    y_val=split.y_val.tolist(),
)

# Evaluate
y_pred_transformer = transformer_model.predict(split.X_test.tolist())
from sklearn.metrics import accuracy_score, f1_score, classification_report
acc = accuracy_score(split.y_test, y_pred_transformer)
f1 = f1_score(split.y_test, y_pred_transformer)

print(f"\n✅ DistilBERT Results:")
print(f"  Accuracy: {acc:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"\n{classification_report(split.y_test, y_pred_transformer, target_names=['ham', 'spam'])}")

# Save
transformer_model.save()
print("💾 Transformer saved to models/distilbert/")

# %%
# ═══════════════════════════════════════════════
# CELL 9: (OPTIONAL) Train BERT & RoBERTa
# ═══════════════════════════════════════════════
# Uncomment to train additional transformers (~10-15 min each)

# for model_name in ["bert", "roberta"]:
#     print(f"\n▶ Training: {model_name.upper()}")
#     clf = TransformerSpamClassifier(
#         model_name=model_name,
#         max_length=256,
#         batch_size=16,
#         epochs=3,
#         learning_rate=2e-5,
#     )
#     clf.train(
#         X_train=split.X_train.tolist(),
#         y_train=split.y_train.tolist(),
#         X_val=split.X_val.tolist(),
#         y_val=split.y_val.tolist(),
#     )
#     y_pred = clf.predict(split.X_test.tolist())
#     acc = accuracy_score(split.y_test, y_pred)
#     f1 = f1_score(split.y_test, y_pred)
#     print(f"  ✅ {model_name.upper()} — Acc: {acc:.4f}, F1: {f1:.4f}")
#     clf.save()

# %%
# ═══════════════════════════════════════════════
# CELL 10: Final Summary & Download Models
# ═══════════════════════════════════════════════

import json

print("=" * 60)
print("🏆 FINAL TRAINING SUMMARY")
print("=" * 60)

print("\n📋 Traditional ML:")
for r in traditional_results:
    print(f"  {r.model_name:<25} F1: {r.f1:.4f}  Acc: {r.accuracy:.4f}")

print("\n📋 Deep Learning:")
for r in dl_results:
    print(f"  {r['model']:<25} F1: {r['f1']:.4f}  Acc: {r['accuracy']:.4f}")

print(f"\n📋 Transformer (DistilBERT):")
print(f"  {'distilbert':<25} F1: {f1:.4f}  Acc: {acc:.4f}")

# Zip all trained models for download
print("\n📦 Zipping trained models...")
!cd models && zip -r /content/spamshield_trained_models.zip . -x "*.training*"
print("✅ Download: /content/spamshield_trained_models.zip")

# Also push models back to repo (optional)
print("\n💡 To push trained models back to GitHub, run:")
print('  !git add models/ && git commit -m "feat: add trained models" && git push')

# %%
# ═══════════════════════════════════════════════
# CELL 11: Quick Inference Test
# ═══════════════════════════════════════════════

test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "CONGRATULATIONS! You've won a FREE iPhone! Click here NOW: http://spam.com",
    "Hi mom, I'll be home by 6pm. Love you!",
    "URGENT: Your bank account has been compromised. Verify immediately at http://phish.com",
    "Can you send me the meeting notes from yesterday?",
    "FREE FREE FREE! Win $10000 cash prize! Text WIN to 80085",
]

print("=" * 60)
print("🔍 INFERENCE DEMO")
print("=" * 60)

# Test with best traditional model
best_trad = traditional_models[0] if traditional_models else None
if best_trad:
    print(f"\n📊 Best Traditional Model: {traditional_results[0].model_name}")
    for msg in test_messages:
        processed = pipeline.process_to_text(msg)
        features = tfidf.transform([processed])
        pred = best_trad.predict(features)[0]
        proba = best_trad.predict_proba(features)[0]
        label = "🚨 SPAM" if pred == 1 else "✅ HAM"
        conf = max(proba) * 100
        print(f"  {label} ({conf:.1f}%) → {msg[:60]}")

# Test with transformer
print(f"\n🤖 Transformer (DistilBERT):")
for msg in test_messages:
    pred = transformer_model.predict([msg])[0]
    proba = transformer_model.predict_proba([msg])[0]
    label = "🚨 SPAM" if pred == 1 else "✅ HAM"
    conf = max(proba) * 100
    print(f"  {label} ({conf:.1f}%) → {msg[:60]}")
