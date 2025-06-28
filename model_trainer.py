import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier  # Changed from MultinomialNB
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import os
import json

class LaptopQueryClassifier:
    
    def __init__(self):
        # The model and vectorizer can either be trained or loaded from disk
        # - If you want to train a new model: call train_model()
        # - If you want to use an existing trained model: call load_model()
        # - The train_model() method automatically saves the model to disk after training
        # - This way you don't have to retrain the model each time you want to use it
        self.model = None
        self.vectorizer = None
        self.categories = ['student', 'business', 'gaming', 'multimedia', 'portable', 'budget', '2-in-1', 'workstation']
        
        try:
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('stopwords')
            
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def load_training_data(self, filename='55555555555.json'):
        """Load training data from JSON file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"Successfully loaded {len(data)} samples from {filename}")
                return data
            else:
                raise FileNotFoundError(f"Data file '{filename}' not found.")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def preprocess_text(self, text):
        """Preprocess text by lowercasing, removing non-alphabetic chars, and lemmatizing"""
        text = text.lower()
        
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def prepare_data(self, data_file='55555555555.json', test_size=0.2, random_state=42):
        """Load data and split it into training and test sets"""
        try:
            # Load all data from the file
            all_data = self.load_training_data(data_file)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            if 'query' not in df.columns:
                raise ValueError("Data must have 'query' column")
                
            if 'category' not in df.columns:
                raise ValueError("Data must have 'category' column")
            
            # Preprocess all queries
            df['processed_query'] = df['query'].apply(self.preprocess_text)
            
            # Split the data into training and test sets with stratification
            X = df['processed_query']
            y = df['category']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f"Data split into {len(X_train)} training samples and {len(X_test)} test samples")
            
            # Report class distribution to verify stratification
            train_dist = y_train.value_counts(normalize=True)
            test_dist = y_test.value_counts(normalize=True)
            # print("Class distribution in training set:")
            # for category, percentage in train_dist.items():
            #     print(f"  {category}: {percentage:.2%}")
            # print("Class distribution in test set:")
            # for category, percentage in test_dist.items():
            #     print(f"  {category}: {percentage:.2%}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Failed to prepare data: {e}")
            print("Unable to continue without proper data.")
            raise
    
    def train_model(self, data_file='55555555555.json', test_size=0.2, random_state=42):
        """Train the model using data from data_file with internal train-test split"""
        X_train, X_test, y_train, y_test = self.prepare_data(data_file, test_size, random_state)
        
        print(f"Training model with {len(X_train)} samples, testing with {len(X_test)} samples")
        
        # Create and fit the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Use both unigrams and bigrams for better context
            min_df=2             # Ignore terms that appear in fewer than 2 documents
        )
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        # Use Random Forest for better performance
        self.model = RandomForestClassifier(
            n_estimators=100,    # Number of trees
            max_depth=None,      # Maximum depth of trees (None means unlimited)
            min_samples_split=2,
            class_weight='balanced',  # Handle class imbalance
            random_state=random_state,
            n_jobs=-1            # Use all available cores
        )
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate on training data
        y_train_pred = self.model.predict(X_train_tfidf)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Evaluate on test data
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_test_pred = self.model.predict(X_test_tfidf)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        }
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Generate and print detailed classification report
        try:
            report = classification_report(
                y_test, 
                y_test_pred, 
                output_dict=True,
                zero_division=0 
            )
            
            # print("\nDetailed Classification Report:")
            # print(classification_report(y_test, y_test_pred, zero_division=0))
            
            print("\nOverall Model Performance:")
            if 'weighted avg' in report:
                results['precision'] = report['weighted avg']['precision']
                results['recall'] = report['weighted avg']['recall']
                results['f1_score'] = report['weighted avg']['f1-score']
                
                print(f"Precision: {results['precision']:.4f}")
                print(f"Recall: {results['recall']:.4f}")
                print(f"F1-score: {results['f1_score']:.4f}")
            else:
                print("Warning: Could not calculate weighted average metrics.")
                
        except Exception as e:
            print(f"Error generating classification report: {e}")
            print("Continuing with model training...")
        
        # Save the trained model
        self.save_model()
        
        return results
    
    def save_model(self, model_path='laptop_classifier_model.pkl', vectorizer_path='laptop_tfidf_vectorizer.pkl'):
        """Save the trained model and vectorizer to disk"""
        if self.model is not None and self.vectorizer is not None:
            joblib.dump(self.model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            print(f"Model saved to {model_path}")
            print(f"Vectorizer saved to {vectorizer_path}")
        else:
            print("Error: Model or vectorizer not initialized. Train the model first.")
    
    def load_model(self, model_path='c:\\Users\\great\\Desktop\\main mig\\server\\model\\laptop_classifier_model.pkl', 
               vectorizer_path='c:\\Users\\great\\Desktop\\main mig\\server\\model\\laptop_tfidf_vectorizer.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print(f"Model loaded from {model_path}")
            print(f"Vectorizer loaded from {vectorizer_path}")
            return True
        except FileNotFoundError:
            print(f"Error: Model files not found. Please train the model first.")
            return False
    
    def predict(self, queries):
        """Predict the category of user queries"""
        if self.model is None or self.vectorizer is None:
            success = self.load_model()
            if not success:
                print("Error: Model or vectorizer not available.")
                return None
        
        # Handle single query case
        if isinstance(queries, str):
            queries = [queries]
        
        try:
            processed_queries = [self.preprocess_text(query) for query in queries]
            
            query_vectors = self.vectorizer.transform(processed_queries)
            
            predictions = self.model.predict(query_vectors)
            
            probabilities = self.model.predict_proba(query_vectors)
            
            results = []
            for i, query in enumerate(queries):
                category = predictions[i]
                
                # Find the highest probability
                category_idx = self.model.classes_.tolist().index(category)
                probability = probabilities[i][category_idx]
                
                results.append({
                    'query': query,
                    'predicted_category': category,
                    'confidence': probability
                })
            
            if len(results) == 1:
                return results[0]['predicted_category']
            
            return [result['predicted_category'] for result in results]
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            if len(queries) == 1:
                return "general"
            return ["general"] * len(queries)
    
    def analyze_query(self, query):
        """Analyze a single query with detailed information"""
        if self.model is None or self.vectorizer is None:
            success = self.load_model()
            if not success:
                print("Error: Model or vectorizer not available.")
                return None
        
        processed_query = self.preprocess_text(query)
        
        query_vector = self.vectorizer.transform([processed_query])
        
        prediction = self.model.predict(query_vector)[0]
        probabilities = self.model.predict_proba(query_vector)[0]
        
        result = {
            'query': query,
            'processed_query': processed_query,
            'predicted_category': prediction,
            'categories': {}
        }
        
        for i, category in enumerate(self.model.classes_):
            result['categories'][category] = probabilities[i]
        
        if ("programming" in query.lower() or "development" in query.lower() or "coding" in query.lower()) and \
           prediction != "business":
            business_idx = list(self.model.classes_).index("business") if "business" in self.model.classes_ else -1
            if business_idx >= 0 and probabilities[business_idx] > 0.2:
                prediction = "business"
                result['predicted_category'] = prediction
                probabilities = list(probabilities)
                probabilities[business_idx] = max(probabilities)
        
        return result


if __name__ == "__main__":
    classifier = LaptopQueryClassifier()
    
    data_file = '55555555555.json'
    
    print("Training laptop query classifier...")
    # Using internal train-test split instead of separate test file
    accuracy_metrics = classifier.train_model(data_file, test_size=0.2, random_state=42)
    
    print(f"\nModel training complete.")
    print(f"Final training accuracy: {accuracy_metrics['train_accuracy']:.4f}")
    print(f"Final test accuracy: {accuracy_metrics['test_accuracy']:.4f}")
    print(f"Model and vectorizer saved to the current directory.")
    
    # Test with some sample queries
    sample_queries = [
        "I need a laptop for college classes",
        "Looking for a gaming laptop that can run Fortnite",
        "Need a business laptop for presentations",
        "What's a good laptop for video editing and graphic design",
        "Need a lightweight laptop for traveling"
    ]
    
    # print("\nTesting with sample queries:")
    # for query in sample_queries:
    #     result = classifier.analyze_query(query)
    #     print(f"Query: {query}")
    #     print(f"Predicted category: {result['predicted_category']}")
    #     print(f"Top 3 categories with probabilities:")
        
    #     # Sort categories by probability and show top 3
    #     sorted_categories = sorted(
    #         result['categories'].items(), 
    #         key=lambda x: x[1], 
    #         reverse=True
    #     )[:3]
        
    #     for category, prob in sorted_categories:
    #         print(f"  {category}: {prob:.4f}")
    #     print()