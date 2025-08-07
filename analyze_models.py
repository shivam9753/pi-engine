import pickle
import json
import numpy as np

def analyze_theme_models():
    try:
        # Load the training data
        with open('C:/pi-engine/data/manual_training.json', 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        print(f"📊 Training Data Analysis:")
        print(f"Total poems: {len(training_data)}")
        
        # Analyze themes distribution
        all_themes = []
        for poem in training_data:
            if 'themes' in poem:
                all_themes.extend(poem['themes'])
        
        from collections import Counter
        theme_counts = Counter(all_themes)
        print(f"\n🎭 Theme Distribution:")
        for theme, count in theme_counts.most_common():
            print(f"  {theme}: {count} poems")
        
        # Load models
        print(f"\n🤖 Model Analysis:")
        
        # Load theme extractor
        with open('C:/pi-engine/models/theme_extractor.pkl', 'rb') as f:
            theme_extractor = pickle.load(f)
        
        print(f"Trained theme models: {len(theme_extractor)} themes")
        for theme in theme_extractor.keys():
            print(f"  - {theme}")
        
        # Load theme binarizer
        with open('C:/pi-engine/models/theme_binarizer.pkl', 'rb') as f:
            theme_binarizer = pickle.load(f)
        
        print(f"\nBinarizer classes: {list(theme_binarizer.classes_)}")
        
        # Load vectorizer
        with open('C:/pi-engine/models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        print(f"\nVectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"Feature range: {vectorizer.max_features}")
        
        # Test a sample poem
        print(f"\n🔍 Testing with sample poem:")
        sample_text = "I feel sad and lonely in the darkness"
        X = vectorizer.transform([sample_text])
        
        print(f"Sample text: '{sample_text}'")
        print(f"Theme predictions:")
        
        for theme, model in theme_extractor.items():
            try:
                prob = model.predict_proba(X)[0][1]
                print(f"  {theme}: {prob:.3f}")
            except Exception as e:
                print(f"  {theme}: ERROR - {e}")
    
    except Exception as e:
        print(f"❌ Error analyzing models: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_theme_models()