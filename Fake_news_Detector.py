from flask import Flask, render_template, request
import joblib #to load pre-trained machine learning models and vectorizers.
import re #string searching and manipulation based on patterns.
import numpy as np #numerical computations 
from PIL import Image # opening, manipulating, and saving image files
import os # operating system-dependent functionality
import nltk #working with human language data
from nltk.corpus import stopwords #common words 
from nltk.stem import WordNetLemmatizer #reduces words to their base or root form
from sklearn.feature_extraction.text import TfidfVectorizer #converts a collection of raw documents into a matrix of TF-IDF features
from sklearn.linear_model import LogisticRegression #binary classification problems
from sklearn.tree import DecisionTreeClassifier #based on feature values
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier #trying to correct errors made by the previous ones
import string 
import logging #tracking events 
import torch #deep learning
from transformers import AutoTokenizer, AutoModelForSequenceClassification #library of hugging face provides pre-trained models for natural language processing tasks.
from fractions import Fraction #represents rational numbers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from textblob import TextBlob #represents rational numbers
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  #sentiment analysis tool designed for social media texts.
from urllib.parse import urlparse #finding differences between two images.
from PIL import Image, ImageChops #such as finding differences between two images.
import cv2 #edge detection and image transformations.


app = Flask(__name__)
app.config['SECRET_KEY'] = " "
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError as e:
    logging.error(f"NLTK download error: {e}.  Attempting to download...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    logging.info("NLTK downloads completed.")


vader_analyzer = SentimentIntensityAnalyzer() 

def preprocess_text(text):
    """
    Improved text preprocessing: removes URLs, punctuation, and stopwords; lemmatizes words.
    """
    if not isinstance(text, str):
        logging.warning(f"Non-string input received: {type(text)}. Returning empty string.")
        return ""

    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    processed_text = ' '.join(words)
    return processed_text


def analyze_sentiment(text): #pos, neu, neg, and compound.
    """
    Performs sentiment analysis using VADER. Returns a compound score.
    """
    scores = vader_analyzer.polarity_scores(text)
    return scores['compound']


def check_topic_agnostic_features(text):
    """
    Checks for topic-agnostic features like excessive capitalization, exaggerated punctuation, and emotive language.
    """
    num_exclamation = text.count("!")
    num_question = text.count("?")


    excessive_punctuation = (num_exclamation + num_question) > (len(text.split()) * 0.05)


    emotive_words = ["shocking", "outrageous", "unbelievable", "amazing", "terrible", "horrible", "disgusting", "mind-blowing", "astounding", "sensational", "controversial"] #added more words
    emotive_language = any(word in text.lower() for word in emotive_words)


    capitalized_words = [word for word in text.split() if word.isupper() and len(word) > 1]
    excessive_capitalization = len(capitalized_words) > (len(text.split()) * 0.15)

    return {
        "excessive_punctuation": excessive_punctuation,
        "emotive_language": emotive_language,
        "excessive_capitalization": excessive_capitalization,
    }


def analyze_image(image_dataset):
    """Analyzes an image and returns "Real Image" or "Fake Image"."""
    try:
        logging.debug(f"Analyzing image: {image_dataset}")
        img = Image.open(image_dataset)
        width, height = img.size
        file_size = os.path.getsize(image_dataset) / 1024
        logging.debug(f"Image Dimensions: {width}x{height}, File size: {file_size} KB")

        features = {}

        features['file_size'] = file_size

        try:
            aspect_ratio = Fraction(width, height)
            features['aspect_ratio'] = aspect_ratio.numerator / aspect_ratio.denominator
        except ZeroDivisionError:
            logging.error("Division by zero error with width or height is zero", exc_info=True)
            return "Error"

        try:
            gray = img.convert('L')
            edges = cv2.Canny(np.array(gray), 50, 150)
            features['edge_density'] = np.count_nonzero(edges) / float(width * height)
        except Exception as e:
            logging.error(f"Error during edge detection: {e}", exc_info=True)
            features['edge_density'] = 0

        try:
            palette = img.convert('P', palette=Image.ADAPTIVE, colors=256).getcolors()
            features['num_colors'] = len(palette)
        except Exception as e:
            logging.error(f"Error during color palette analysis: {e}", exc_info=True)
            features['num_colors'] = 0

        try:
            features['noise'] = np.std(np.array(gray))
        except Exception as e:
            logging.error(f"Error during noise analysis: {e}", exc_info=True)
            features['noise'] = 0

        features['ela_max_diff'] = 0
        try:
            ela_quality = 85
            ela_filename = "ela_temp.jpg"
            img.save(ela_filename, 'JPEG', quality=ela_quality)
            ela_img = Image.open(ela_filename)
            diff = ImageChops.difference(img, ela_img)
            extrema = diff.getextrema()
            features['ela_max_diff'] = max([pix[1] for pix in extrema])
            os.remove(ela_filename)
        except Exception as e:
            logging.error(f"Error during ELA: {e}", exc_info=True)

        try:
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features['blur'] = np.var(laplacian)
        except Exception as e:
            logging.error(f"Error during blur detection: {e}", exc_info=True)
            features['blur'] = 0

        # New Features:
        try:
            brightness = np.mean(np.array(gray))
            features['brightness'] = brightness
        except Exception as e:
            logging.error(f"Error during brightness analysis: {e}", exc_info=True)
            features['brightness'] = 0

        try:
            contrast = np.std(np.array(gray))
            features['contrast'] = contrast
        except Exception as e:
            logging.error(f"Error during contrast analysis: {e}", exc_info=True)
            features['contrast'] = 0

        logging.debug(f"Extracted Features: {features}")

        # Adjusted Thresholds
        file_size_threshold = 10  # KB
        aspect_ratio_threshold_low = 0.5
        aspect_ratio_threshold_high = 2.0
        edge_density_threshold = 0.1
        num_colors_threshold = 50
        noise_threshold = 15
        ela_max_diff_threshold = 70
        blur_threshold = 80
        brightness_threshold = 50
        contrast_threshold = 10

        if features['file_size'] < file_size_threshold:
            logging.info("Fake Image: File size is very small.")
            return "Fake Image"

        if features['aspect_ratio'] < aspect_ratio_threshold_low or features['aspect_ratio'] > aspect_ratio_threshold_high:
            logging.info("Fake Image: Unusual aspect ratio.")
            return "Fake Image"

        if features['ela_max_diff'] > ela_max_diff_threshold:
            logging.info("Fake Image: High ELA score detected.")
            return "Fake Image"

        if features['blur'] < blur_threshold and features['edge_density'] > edge_density_threshold:
            logging.info("Fake Image: Low blur and high edge density suggests screenshot.")
            return "Fake Image"

        if features['num_colors'] < num_colors_threshold and features['noise'] < noise_threshold:
            logging.info("Fake Image: Limited color palette and low noise suggests screenshot.")
            return "Fake Image"

        if features['brightness'] < brightness_threshold:
            logging.info("Fake Image: Low brightness detected.")
            return "Fake Image"

        if features['contrast'] < contrast_threshold:
            logging.info("Fake Image: Low contrast detected.")
            return "Fake Image"

        logging.info("Real Image: No strong signs of manipulation detected.")
        return "Real Image"

    except FileNotFoundError:
        logging.error("Image file not found.", exc_info=True)
        return "Error"
    except (IOError, OSError) as e:
        logging.error(f"Cannot open or read image file: {e}", exc_info=True)
        return "Error"
    except Exception as e:
        logging.exception(f"Error analyzing image: {e}")
        return "Error"

try:
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    LR_model = joblib.load('LR_model.joblib')
    DT_model = joblib.load('DT_model.joblib')
    GB_model = joblib.load('GB_model.joblib')
    RF_model = joblib.load('RF_model.joblib')

    model_name = "bert-base-uncased"
    bert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    logging.info(f"Models and vectorizer loaded successfully, including BERT model '{model_name}' from Hugging Face Hub.")

except Exception as e:
    logging.error(f"Error loading models: {e}", exc_info=True)
    exit()

def predict(news, threshold=0.28, bert_weight=0.3, sentiment_weight=0.02, topic_weight=0.05):
    """
    Predicts whether a news article is real or fake, combining traditional models, BERT, sentiment, and topic features.
    """
    news_processed = preprocess_text(news)

    if not news_processed.strip() or len(news_processed) < 10:
        logging.warning("Input text is empty or too short after preprocessing.")
        return "Error: Input text is empty or too short after preprocessing."

    try:
        new_xv_test = vectorizer.transform([news_processed])
        prob_LR_real = LR_model.predict_proba(new_xv_test)[0][1]
        prob_DT_real = DT_model.predict_proba(new_xv_test)[0][1]
        prob_GB_real = GB_model.predict_proba(new_xv_test)[0][1]
        prob_RF_real = RF_model.predict_proba(new_xv_test)[0][1]
        average_probability_traditional = (prob_LR_real + prob_DT_real + prob_GB_real + prob_RF_real) / 4

        # BERT Prediction
        inputs = bert_tokenizer(news_processed, return_tensors="pt", truncation=True, padding=True)
        outputs = bert_model(**inputs)
        bert_probability = torch.softmax(outputs.logits, dim=1)[0][1].item()

        # Sentiment Analysis (VADER)
        sentiment_score = analyze_sentiment(news_processed)
        logging.info(f"Sentiment Score (VADER): {sentiment_score:.4f}")
        # Using absolute value of sentiment score to identify neutral content
        abs_sentiment_score = abs(sentiment_score)

        # Check for topic-agnostic features
        topic_agnostic_features = check_topic_agnostic_features(news)
        logging.info(f"Topic-Agnostic Features: {topic_agnostic_features}")

        # Calculate a topic-agnostic score
        topic_agnostic_score = 0
        if topic_agnostic_features['excessive_punctuation']:
            topic_agnostic_score -= 0.03  # Reduced penalty
        if topic_agnostic_features['emotive_language']:
            topic_agnostic_score -= 0.05  # Reduced penalty
        if topic_agnostic_features['excessive_capitalization']:
            topic_agnostic_score -= 0.01  # Minimal penalty
        topic_agnostic_score = max(-0.1, topic_agnostic_score)  # Ensure score is not too low

        # Extract URLs and create a feature
        url_present = 1 if re.search(r'https?://\S+|www\.\S+', news) else 0
        url_weight = 0.02 if url_present else 0  # Small bonus for URL presence

        # Weighted Averaging
        average_probability = (
            (1 - bert_weight - sentiment_weight - topic_weight) * average_probability_traditional
            + bert_weight * bert_probability
            + sentiment_weight * abs_sentiment_score  # Using absolute sentiment score
            + topic_weight * topic_agnostic_score
            + url_weight # Adding a small bonus for URL
        )

        # Clip the probability to be between 0 and 1
        average_probability = max(0, min(1, average_probability))

        # Introduce a minimum probability for "True News"
        if average_probability_traditional > 0.6:
            final_prediction_label = "True News"
        elif average_probability > threshold:
            final_prediction_label = "True News"
        else:
            final_prediction_label = "Fake News"

        logging.info(f"Prediction: {final_prediction_label}, Avg Prob: {average_probability:.4f}, BERT Prob: {bert_probability:.4f}, Sentiment: {sentiment_score:.4f}, Topic: {topic_agnostic_score:.4f}, URL: {url_present}")

        return final_prediction_label

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return "Error during prediction."


@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    image_analysis = None
    threshold = 0.28
    bert_weight = 0.3
    sentiment_weight = 0.02
    topic_weight = 0.05

    if request.method == 'POST':
        news_text = request.form.get('news_text', '')
        try:
            threshold = float(request.form.get('threshold', 0.5))
            threshold = max(0.01, min(threshold, 0.99))
        except ValueError:
            logging.warning("Invalid threshold provided, using default 0.5")
            threshold = 0.28

        try:
            bert_weight = float(request.form.get('bert_weight', 0.5))
            bert_weight = max(0.0, min(bert_weight, 1.0))
        except ValueError:
            logging.warning("Invalid BERT weight provided, using default 0.5")
            bert_weight = 0.3

        try:
            sentiment_weight = float(request.form.get('sentiment_weight', 0.1))
            sentiment_weight = max(0.0, min(sentiment_weight, 1.0))
        except ValueError:
            logging.warning("Invalid Sentiment Weight provided, using default 0.1")
            sentiment_weight = 0.02

        try:
            topic_weight = float(request.form.get('topic_weight', 0.1))
            topic_weight = max(0.0, min(topic_weight, 1.0))
        except ValueError:
            logging.warning("Invalid Topic Weight provided, using default 0.1")
            topic_weight = 0.05

        image_file = request.files.get('image_file')
        image_path = None

        if news_text.strip():
            prediction_text = predict(news_text, threshold, bert_weight, sentiment_weight, topic_weight)
            result = prediction_text
            logging.info(f"Text prediction result: {result}")
        else:
            logging.info("No text input provided.")

        if image_file and image_file.filename != '':
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            try:
                image_file.save(image_path)
                image_analysis = analyze_image(image_path)
                logging.info(f"Image analysis result: {image_analysis}")
            except Exception as e:
                image_analysis = f"Error processing image: {e}"
                logging.exception(f"Image processing error: {e}")
        else:
            logging.info("No image file uploaded.")
    return render_template('index.html', result=result, image_analysis=image_analysis)

if __name__ == '__main__':
    app.run(debug=True)
