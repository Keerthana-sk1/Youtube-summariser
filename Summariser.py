!pip install nltk
!pip install gensim
!pip install -U openai-whisper
!pip install ffmpeg
!pip install pytube
import nltk
import pytube as pt
import whisper
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm
import string
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Download YouTube audio and transcribe
input_link=input("enter the link:")
yt = pt.YouTube(input_link)
stream = yt.streams.filter(only_audio=True)[0]
stream.download(filename="audio_english.mp3")
model = whisper.load_model("base")
result = model.transcribe("audio_english.mp3")
transcribed_text = result["text"]
print(transcribed_text)

# Preprocess
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    return tokens

# Tokenization
processed_sentences = [preprocess(sentence) for sentence in sent_tokenize(transcribed_text)]

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in processed_sentences]

# Coherence scores for different numbers of topics
coherence_scores = []
for num_topics in tqdm(range(2, 10)):
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    coherence_model = CoherenceModel(model=lda_model, texts=processed_sentences, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    coherence_scores.append(coherence_score)

# Optimal number of topics
optimal_num_topics = coherence_scores.index(max(coherence_scores)) + 2

# Train LDA model with optimal number of topics
lda_model = models.LdaModel(corpus, num_topics=optimal_num_topics, id2word=dictionary, passes=10)

# Assign topics to sentences
topic_assignments = []
for sentence in processed_sentences:
    bow = dictionary.doc2bow(sentence)
    topic_assignment = max(lda_model[bow], key=lambda x: x[1])[0]  # Assign sentence to topic with highest probability
    topic_assignments.append(topic_assignment)

# Group sentences by topics
grouped_sentences = [[] for _ in range(optimal_num_topics)]
for i, sentence in enumerate(sent_tokenize(transcribed_text)):
    grouped_sentences[topic_assignments[i]].append(sentence)

# Convert topics to dictionary
topic_dict = {}
for i, sentences in enumerate(grouped_sentences):
    if sentences:
        topic_key = sentences[0]  # First line as key
        topic_values = sentences[1:]  # Rest as values
        topic_dict[topic_key] = topic_values

# Print all keys
print("\n\nAll topics:")
for key in topic_dict.keys():
    print(key)

# Take input key 
user_input_key = input("Enter the topic key to display its values: ")

# Show values 
if user_input_key in topic_dict:
    print(f"Values for topic '{user_input_key}':")
    for value in topic_dict[user_input_key]:
        print(value)
else:

    print("Topic key not found.")

# Remove downloaded audio file
os.remove("audio_english.mp3")

