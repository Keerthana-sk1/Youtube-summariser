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
from gensim import corpora, models, similarities, matutils
from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm

import string
from gensim import corpora, models
import os
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
yt = pt.YouTube("https://youtu.be/4RixMPF4xis?si=mZhoxdUtRBiRqaMU")
stream = yt.streams.filter(only_audio=True)[0]
stream.download(filename="audio_english.mp3")
model = whisper.load_model("base")
result = model.transcribe("audio_english.mp3")
transcribed_text = result["text"]
print(transcribed_text)
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    return tokens
processed_sentences = [preprocess(sentence) for sentence in sent_tokenize(transcribed_text)]
processed_sentences = [preprocess(sentence) for sentence in sent_tokenize(transcribed_text)]
processed_sentences = [preprocess(sentence) for sentence in sent_tokenize(transcribed_text)]
dictionary = corpora.Dictionary(processed_sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in processed_sentences]
coherence_scores = []
for num_topics in tqdm(range(2, 10)):
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary)
    coherence_model = CoherenceModel(model=lda_model, texts=processed_sentences, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    coherence_scores.append(coherence_score)
optimal_num_topics = coherence_scores.index(max(coherence_scores)) + 2
lda_model = models.LdaModel(corpus, num_topics=optimal_num_topics, id2word=dictionary, passes=10)
topic_assignments = []
for sentence in processed_sentences:
    bow = dictionary.doc2bow(sentence)
    topic_assignment = max(lda_model[bow], key=lambda x: x[1])[0]  # Assign sentence to topic with highest probability
    topic_assignments.append(topic_assignment)
grouped_sentences = [[] for _ in range(optimal_num_topics)]
for i, sentence in enumerate(sent_tokenize(transcribed_text)):
    grouped_sentences[topic_assignments[i]].append(sentence) for i, sentences in enumerate(grouped_sentences):
    if sentences:
        print(f"Topic {i}:")
        for sentence in sentences:
            print(sentence)
        print()
os.remove("audio_english.mp3")

