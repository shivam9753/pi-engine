import sys
import json
import re
import string
import os
import pickle
from collections import Counter
from difflib import SequenceMatcher
import numpy as np

def preprocess_text(text):
    """Basic text preprocessing"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def count_syllables(word):
    """Basic syllable counting"""
    word = word.lower()
    vowels = 'aeiouy'
    syllable_count = 0
    previous_was_vowel = False
    
    for char in word:
        if char in vowels:
            if not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = True
        else:
            previous_was_vowel = False
    
    # Handle silent e
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1
    
    return max(1, syllable_count)

def analyze_quality_ml(text, models):
    """Analyze poetry quality using trained ML model"""
    if not models:
        return analyze_quality_fallback(text)
    
    try:
        # Vectorize text
        X = models['vectorizer'].transform([text])
        
        # Predict quality using trained model
        quality_score = models['quality_model'].predict(X)[0]
        
        # Ensure score is within valid range
        return min(10.0, max(1.0, quality_score))
    
    except Exception as e:
        print(f"Warning: ML quality analysis failed: {e}", file=sys.stderr)
        return analyze_quality_fallback(text)

def analyze_quality_fallback(text):
    """Fallback quality analysis using basic metrics (original method)"""
    lines = text.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    if not lines:
        return 1.0
    
    score = 5.0  # Base score
    
    # Check line count (poems should have multiple lines)
    if len(lines) > 1:
        score += 1.0
    
    # Check vocabulary richness
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) > 0:
        unique_words = len(set(words))
        vocabulary_ratio = unique_words / len(words)
        score += vocabulary_ratio * 2
    
    # Check for poetic devices (alliteration, repetition)
    first_letters = [word[0].lower() for word in words if word]
    letter_counts = Counter(first_letters)
    if any(count >= 3 for count in letter_counts.values()):
        score += 0.5  # Bonus for alliteration
    
    # Check line length consistency
    line_lengths = [len(line.split()) for line in lines]
    if len(line_lengths) > 1:
        avg_length = sum(line_lengths) / len(line_lengths)
        variance = sum((x - avg_length) ** 2 for x in line_lengths) / len(line_lengths)
        if variance < 4:  # Consistent line lengths
            score += 0.5
    
    # Cap score at 10
    return min(10.0, max(1.0, score))

def detect_style_ml(text, models):
    """Detect poetry style using trained ML model"""
    if not models:
        return detect_style_fallback(text)
    
    try:
        # Vectorize text
        X = models['vectorizer'].transform([text])
        
        # Predict style using trained model
        predicted_style = models['style_classifier'].predict(X)[0]
        
        return predicted_style
    
    except Exception as e:
        print(f"Warning: ML style detection failed: {e}", file=sys.stderr)
        return detect_style_fallback(text)

def detect_style_fallback(text):
    """Fallback style detection using rule-based approach (original method)"""
    lines = text.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    if not lines:
        return "unknown"
    
    # Check for haiku (3 lines, roughly 5-7-5 syllables)
    if len(lines) == 3:
        syllable_counts = []
        for line in lines:
            words = re.findall(r'\b\w+\b', line)
            syllables = sum(count_syllables(word) for word in words)
            syllable_counts.append(syllables)
        
        if (4 <= syllable_counts[0] <= 6 and 
            6 <= syllable_counts[1] <= 8 and 
            4 <= syllable_counts[2] <= 6):
            return "haiku"
    
    # Check for sonnet (14 lines)
    if len(lines) == 14:
        return "sonnet"
    
    # Check for free verse vs structured
    line_lengths = [len(line.split()) for line in lines]
    if len(set(line_lengths)) == len(line_lengths):  # All different lengths
        return "free_verse"
    
    # Check for rhyme patterns (basic check)
    last_words = []
    for line in lines[-4:]:  # Check last 4 lines
        words = re.findall(r'\b\w+\b', line.lower())
        if words:
            last_words.append(words[-1][-2:])  # Last 2 characters
    
    if len(set(last_words)) < len(last_words):  # Some rhymes found
        return "rhyming"
    
    return "modern"

def load_trained_models():
    """Load trained ML models from files"""
    models_dir = 'C:/pi-engine/models'
    models = {}
    
    try:
        # Load vectorizer
        with open(os.path.join(models_dir, 'vectorizer.pkl'), 'rb') as f:
            models['vectorizer'] = pickle.load(f)
        
        # Load quality model
        with open(os.path.join(models_dir, 'quality_model.pkl'), 'rb') as f:
            models['quality_model'] = pickle.load(f)
        
        # Load style classifier
        with open(os.path.join(models_dir, 'style_classifier.pkl'), 'rb') as f:
            models['style_classifier'] = pickle.load(f)
        
        # Load theme extractor
        with open(os.path.join(models_dir, 'theme_extractor.pkl'), 'rb') as f:
            models['theme_extractor'] = pickle.load(f)
        
        # Load theme binarizer
        with open(os.path.join(models_dir, 'theme_binarizer.pkl'), 'rb') as f:
            models['theme_binarizer'] = pickle.load(f)
        
        return models
    except Exception as e:
        print(f"Warning: Could not load trained models: {e}", file=sys.stderr)
        return None

def extract_themes_ml(text, models):
    """Extract themes using trained ML models"""
    if not models:
        return extract_themes_fallback(text)
    
    try:
        # Vectorize text
        X = models['vectorizer'].transform([text])
        
        # Predict themes using trained models
        predicted_themes = []
        for theme, model in models['theme_extractor'].items():
            probability = model.predict_proba(X)[0][1]  # Probability of theme being present
            print(f"Theme '{theme}': probability {probability:.3f}", file=sys.stderr)
            if probability > 0.4:  # More reasonable threshold to avoid too many themes
                predicted_themes.append(theme)
        
        print(f"ML predicted themes: {predicted_themes}", file=sys.stderr)
        return predicted_themes if predicted_themes else ['general']
    
    except Exception as e:
        print(f"Warning: ML theme extraction failed: {e}", file=sys.stderr)
        return ['general']  # Only return general if ML completely fails

def extract_themes_fallback(text):
    """Fallback theme extraction using keywords (enhanced method)"""
    text_lower = text.lower()
    themes = []
    
    # Enhanced theme keywords based on your training data
    theme_keywords = {
        'love': ['love', 'heart', 'romance', 'kiss', 'beloved', 'passion', 'adore', 'affection', 'desire', 'devotion'],
        'nature': ['tree', 'flower', 'sky', 'ocean', 'mountain', 'bird', 'wind', 'sun', 'moon', 'earth', 'rain', 'storm'],
        'death': ['death', 'grave', 'funeral', 'mourn', 'loss', 'goodbye', 'farewell', 'dying', 'dead', 'killed'],
        'time': ['time', 'moment', 'forever', 'eternal', 'past', 'future', 'memory', 'yesterday', 'tomorrow', 'years'],
        'sadness': ['sad', 'cry', 'tear', 'sorrow', 'pain', 'hurt', 'lonely', 'broken', 'weep', 'grief', 'melancholy'],
        'depression': ['depression', 'darkness', 'void', 'empty', 'numb', 'hollow', 'despair', 'hopeless'],
        'joy': ['happy', 'joy', 'laugh', 'smile', 'celebration', 'delight', 'bliss', 'cheerful'],
        'spirituality': ['god', 'prayer', 'soul', 'heaven', 'divine', 'sacred', 'faith', 'holy', 'blessed'],
        'family': ['mother', 'father', 'child', 'family', 'home', 'parent', 'daughter', 'son', 'brother', 'sister'],
        'war and conflict': ['war', 'battle', 'fight', 'violence', 'blood', 'weapon', 'soldier', 'conflict'],
        'trauma': ['trauma', 'wounded', 'scarred', 'damaged', 'hurt', 'abuse', 'nightmare'],
        'friendship': ['friend', 'friendship', 'companion', 'buddy', 'pal', 'together'],
        'loneliness': ['alone', 'lonely', 'solitude', 'isolated', 'empty', 'abandoned'],
        'hope': ['hope', 'dream', 'wish', 'aspire', 'optimism', 'faith', 'believe'],
        'identity': ['identity', 'self', 'who am i', 'myself', 'being', 'existence'],
        'vulnerability': ['vulnerable', 'fragile', 'exposed', 'helpless', 'weak'],
        'toxic relationships': ['toxic', 'manipulation', 'abuse', 'control', 'betrayal']
    }
    
    for theme, keywords in theme_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            themes.append(theme)
    
    print(f"Fallback themes found: {themes}", file=sys.stderr)
    return themes if themes else ['general']

def check_plagiarism(text):
    """Basic plagiarism check (placeholder)"""
    # This is a simplified version - in reality, you'd check against a database
    
    # Check for very common phrases that might indicate copied content
    common_phrases = [
        "roses are red, violets are blue",
        "shall i compare thee to a summer's day",
        "two roads diverged in a yellow wood",
        "because i could not stop for death"
    ]
    
    text_lower = text.lower()
    for phrase in common_phrases:
        similarity = SequenceMatcher(None, phrase, text_lower).ratio()
        if similarity > 0.6:
            return min(95.0, similarity * 100)
    
    # Check for very short submissions (might be copied snippets)
    words = text.split()
    if len(words) < 10:
        return 25.0
    
    # Check for unusual repetition patterns
    lines = text.strip().split('\n')
    if len(lines) > 2:
        line_similarities = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                similarity = SequenceMatcher(None, lines[i], lines[j]).ratio()
                line_similarities.append(similarity)
        
        if line_similarities and max(line_similarities) > 0.8:
            return 40.0
    
    return 5.0  # Low base plagiarism score

def generate_poem_description(text, themes, style, quality, author_bio=None):
    """Generate a meaningful description of what the poem explores"""
    if not themes or themes == ['general']:
        return "A poem exploring universal human experiences and emotions."
    
    # Create description based on themes and quality
    theme_descriptions = {
        'love': 'matters of the heart, romantic connections, and emotional bonds',
        'family': 'familial relationships, heritage, and generational connections',
        'memory': 'reflections on the past, nostalgia, and remembered experiences',
        'time': 'the passage of time, temporal experiences, and life\'s transitions',
        'grief': 'loss, mourning, and the processing of sorrow',
        'sadness': 'melancholy, emotional pain, and moments of despair',
        'joy': 'celebration, happiness, and uplifting human experiences',
        'loneliness': 'isolation, solitude, and the search for connection',
        'hope': 'optimism, dreams, and the resilience of the human spirit',
        'identity': 'self-discovery, personal growth, and questions of belonging',
        'spirituality': 'faith, divine connections, and sacred experiences',
        'nature': 'the natural world, environmental imagery, and our relationship with earth',
        'death': 'mortality, endings, and the contemplation of life\'s fragility',
        'relationships': 'human connections, social bonds, and interpersonal dynamics',
        'parenthood': 'the parent-child relationship, nurturing, and generational wisdom',
        'masculinity': 'male identity, gender roles, and masculine experiences',
        'womanhood': 'female identity, gender experiences, and feminine perspectives',
        'resistance': 'struggle against oppression, social justice, and resilience',
        'fear': 'anxiety, uncertainty, and confronting the unknown',
        'aging': 'the passage of years, growing older, and life\'s seasons'
    }
    
    # Get descriptions for the poem's themes
    theme_text = []
    for theme in themes[:3]:  # Use top 3 themes
        if theme.lower() in theme_descriptions:
            theme_text.append(theme_descriptions[theme.lower()])
    
    if not theme_text:
        return "A poem exploring complex human experiences and emotions."
    
    # Construct description based on number of themes
    if len(theme_text) == 1:
        description = f"This poem explores {theme_text[0]}"
    elif len(theme_text) == 2:
        description = f"This poem explores {theme_text[0]} and {theme_text[1]}"
    else:
        description = f"This poem explores {theme_text[0]}, {theme_text[1]}, and {theme_text[2]}"
    
    # Add quality context
    if quality >= 8.5:
        description += ", demonstrating exceptional literary merit with rich imagery and emotional authenticity."
    elif quality >= 7.5:
        description += ", showing strong literary qualities with effective use of language and well-developed themes."
    else:
        description += ", using accessible language that connects with universal human experiences."
    
    # Add style context if notable
    if style in ['haiku', 'sonnet']:
        description += f" The work follows the {style} form."
    elif style == 'free verse':
        description += " The work is written in free verse."
    
    return description

def calculate_confidence(quality, style, themes, plagiarism):
    """Calculate confidence in the analysis"""
    confidence = 70.0  # Base confidence
    
    # Higher confidence for clear patterns
    if style in ['haiku', 'sonnet']:
        confidence += 15.0
    
    if len(themes) > 0:
        confidence += 10.0
    
    if plagiarism < 10:
        confidence += 5.0
    
    return min(95.0, confidence)

def analyze_poem(text, author_bio=None):
    """Main analysis function using trained ML models"""
    if not text or not text.strip():
        return {
            'quality_score': 1.0,
            'detected_style': 'unknown',
            'themes': [],
            'plagiarism_score': 0.0,
            'confidence': 50.0,
            'error': 'Empty text provided'
        }
    
    try:
        # Load trained models (cached after first load)
        if not hasattr(analyze_poem, '_models_cache'):
            analyze_poem._models_cache = load_trained_models()
        
        models = analyze_poem._models_cache
        
        # Preprocess text
        clean_text = preprocess_text(text)
        
        # Enhance text with author bio context if available
        enhanced_text = clean_text
        if author_bio:
            enhanced_text = f"{clean_text}\n\nAuthor Context: {author_bio}"
        
        # Run analysis using ML models when available
        quality = analyze_quality_ml(clean_text, models)
        style = detect_style_ml(clean_text, models)
        themes = extract_themes_ml(enhanced_text, models)  # Use enhanced text for theme detection
        plagiarism = check_plagiarism(clean_text)
        confidence = calculate_confidence(quality, style, themes, plagiarism)
        
        # Generate meaningful description
        description = generate_poem_description(clean_text, themes, style, quality, author_bio)
        
        # Adjust confidence based on author bio availability
        if author_bio:
            confidence += 5.0  # Higher confidence when we have author context
        
        return {
            'quality_score': round(quality, 1),
            'detected_style': style,
            'themes': themes,
            'plagiarism_score': round(plagiarism, 1),
            'confidence': round(confidence, 1),
            'description': description,
            'using_ml_models': models is not None
        }
        
    except Exception as e:
        return {
            'quality_score': 1.0,
            'detected_style': 'unknown',
            'themes': [],
            'plagiarism_score': 0.0,
            'confidence': 20.0,
            'error': str(e)
        }

if __name__ == '__main__':
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        data = json.loads(input_data)
        
        # Analyze the poem with author bio if provided
        result = analyze_poem(data.get('text', ''), data.get('author_bio', None))
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        # Error handling
        error_result = {
            'quality_score': 1.0,
            'detected_style': 'unknown',
            'themes': [],
            'plagiarism_score': 0.0,
            'confidence': 10.0,
            'error': f'Analysis failed: {str(e)}'
        }
        print(json.dumps(error_result))