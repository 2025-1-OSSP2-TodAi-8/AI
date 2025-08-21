from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained("HyukII/my-emotion-text")
model = AutoModelForSequenceClassification.from_pretrained("HyukII/my-emotion-text").eval()
print(model.config.id2label)   # 라벨 매핑 확인
