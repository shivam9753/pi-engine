import os
import json
import pickle
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

class ManualPoetryTrainer:
    def __init__(self, data_dir='C:/pi-engine/data'):
        self.data_dir = data_dir
        self.training_file = os.path.join(data_dir, 'manual_training.json')
        self.models_dir = 'C:/pi-engine/models'
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load existing training data
        self.training_data = self.load_training_data()
    
    def load_training_data(self):
        """Load existing training data from JSON file"""
        if os.path.exists(self.training_file):
            with open(self.training_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_training_data(self):
        """Save training data to JSON file"""
        with open(self.training_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, indent=2, ensure_ascii=False)
        print(f"Training data saved to {self.training_file}")
    
    def add_poem(self, text, quality, style, themes, plagiarism=0, notes="", author="", author_bio=""):
        """Add a manually labeled poem to training data"""
        poem_data = {
            'id': len(self.training_data) + 1,
            'text': text.strip(),
            'quality': quality,
            'style': style,
            'themes': themes if isinstance(themes, list) else [themes],
            'plagiarism': plagiarism,
            'notes': notes,
            'author': author,
            'author_bio': author_bio,
            'added_date': datetime.now().isoformat()
        }
        
        self.training_data.append(poem_data)
        self.save_training_data()
        print(f"Added poem #{poem_data['id']} to training data")
        return poem_data['id']
    
    def show_training_stats(self):
        """Display current training data statistics"""
        if not self.training_data:
            print("No training data available")
            return
        
        print(f"\n=== Training Data Statistics ===")
        print(f"Total poems: {len(self.training_data)}")
        
        # Quality distribution
        qualities = [p['quality'] for p in self.training_data]
        print(f"Quality scores: min={min(qualities)}, max={max(qualities)}, avg={np.mean(qualities):.1f}")
        
        # Style distribution
        styles = {}
        for p in self.training_data:
            style = p['style']
            styles[style] = styles.get(style, 0) + 1
        print(f"Styles: {dict(sorted(styles.items()))}")
        
        # Theme distribution
        all_themes = []
        for p in self.training_data:
            all_themes.extend(p['themes'])
        theme_counts = {}
        for theme in all_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        print(f"Themes: {dict(sorted(theme_counts.items()))}")
    
    def interactive_add_poem(self):
        """Interactive poem addition interface"""
        print("\n=== Add New Training Poem ===")
        
        # Get poem text
        print("Enter poem text (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
        text = "\n".join(lines)
        
        if not text.strip():
            print("No text entered, canceling.")
            return
        
        # Get quality score
        while True:
            try:
                quality = float(input("Quality score (1-10): "))
                if 1 <= quality <= 10:
                    break
                print("Please enter a score between 1 and 10")
            except ValueError:
                print("Please enter a valid number")
        
        # Get style
        styles = ['free_verse', 'rhyming', 'haiku', 'sonnet', 'limerick', 'modern', 'classical', 'narrative']
        print(f"Available styles: {', '.join(styles)}")
        while True:
            style = input("Style: ").lower().strip()
            if style in styles:
                break
            print(f"Please choose from: {', '.join(styles)}")
        
        # Get themes
        available_themes = ['love', 'nature', 'death', 'time', 'sadness', 'joy', 'spirituality', 'family', 'politics', 'friendship', 'nostalgia', 'hope']
        print(f"Available themes: {', '.join(available_themes)}")
        print("Enter themes separated by commas:")
        themes_input = input("Themes: ").lower().strip()
        themes = [t.strip() for t in themes_input.split(',') if t.strip()]
        
        # Get plagiarism score
        while True:
            try:
                plagiarism = float(input("Plagiarism risk (0-100, default 0): ") or "0")
                if 0 <= plagiarism <= 100:
                    break
                print("Please enter a score between 0 and 100")
            except ValueError:
                print("Please enter a valid number")
        
        # Get optional notes
        notes = input("Notes (optional): ").strip()
        
        # Confirm addition
        print(f"\n=== Confirm Addition ===")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Quality: {quality}")
        print(f"Style: {style}")
        print(f"Themes: {themes}")
        print(f"Plagiarism: {plagiarism}%")
        print(f"Notes: {notes}")
        
        if input("Add this poem? (y/n): ").lower() == 'y':
            poem_id = self.add_poem(text, quality, style, themes, plagiarism, notes)
            print(f"Poem added successfully with ID: {poem_id}")
        else:
            print("Poem not added.")
    
    def train_models(self):
        """Train ML models from current training data"""
        if len(self.training_data) < 5:
            print(f"Need at least 5 poems to train models (have {len(self.training_data)})")
            return
        
        print(f"Training models with {len(self.training_data)} poems...")
        
        # Prepare data
        texts = [p['text'] for p in self.training_data]
        qualities = [p['quality'] for p in self.training_data]
        styles = [p['style'] for p in self.training_data]
        all_themes = [p['themes'] for p in self.training_data]
        
        # Create text features
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        
        # Train quality model (regression)
        quality_model = RandomForestRegressor(n_estimators=100, random_state=42)
        quality_model.fit(X, qualities)
        
        # Train style model (classification)
        style_model = RandomForestClassifier(n_estimators=100, random_state=42)
        style_model.fit(X, styles)
        
        # Train theme models (multi-label)
        theme_binarizer = MultiLabelBinarizer()
        themes_binary = theme_binarizer.fit_transform(all_themes)
        
        theme_models = {}
        for i, theme in enumerate(theme_binarizer.classes_):
            if sum(themes_binary[:, i]) >= 2:  # Need at least 2 examples
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X, themes_binary[:, i])
                theme_models[theme] = model
        
        # Save all models
        models_to_save = {
            'vectorizer.pkl': vectorizer,
            'quality_model.pkl': quality_model,
            'style_classifier.pkl': style_model,
            'theme_extractor.pkl': theme_models,
            'theme_binarizer.pkl': theme_binarizer
        }
        
        for filename, model in models_to_save.items():
            filepath = os.path.join(self.models_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        print(f"Models trained and saved to {self.models_dir}")
        print(f"- Quality model: {len(qualities)} samples")
        print(f"- Style model: {len(set(styles))} styles")
        print(f"- Theme models: {len(theme_models)} themes")
    
    def view_poems(self):
        """View all training poems"""
        if not self.training_data:
            print("No poems in training data yet")
            return
        
        print(f"\n=== Training Poems ({len(self.training_data)} total) ===")
        for i, poem in enumerate(self.training_data, 1):
            print(f"\n--- Poem #{poem['id']} ---")
            print(f"Text: {poem['text'][:100]}{'...' if len(poem['text']) > 100 else ''}")
            print(f"Quality: {poem['quality']}/10")
            print(f"Style: {poem['style']}")
            print(f"Themes: {', '.join(poem['themes'])}")
            if poem.get('notes'):
                print(f"Notes: {poem['notes']}")
            
            if i % 5 == 0:  # Show 5 at a time
                if input("Press Enter to continue (or 'q' to quit): ") == 'q':
                    break

def main():
    trainer = ManualPoetryTrainer()
    
    while True:
        print("\n=== Manual Poetry Trainer ===")
        print("1. Add new poem")
        print("2. Show training statistics")
        print("3. View all poems")
        print("4. Train models")
        print("5. Exit")
        
        choice = input("Choose option (1-5): ").strip()
        
        if choice == '1':
            trainer.interactive_add_poem()
        elif choice == '2':
            trainer.show_training_stats()
        elif choice == '3':
            trainer.view_poems()
        elif choice == '4':
            trainer.train_models()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == '__main__':
    main()