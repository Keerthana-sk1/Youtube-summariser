!pip install nltk
!pip install gensim
!pip install -U openai-whisper
!pip install pytube
!pip install translate
!pip install googletrans==4.0.0-rc1
import os
import ipywidgets as widgets
from tqdm.notebook import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.models import CoherenceModel
import pytube as pt
import whisper
from googletrans import Translator
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
def on_transcribe_button_clicked(b):
    transcribed_text, topic_dict = transcribe_and_classify_topics(input_link.value)
    transcript_output.value = transcribed_text
    topics_output.value = '\n'.join([f'{key}\n{", ".join(values)}\n' for key, values in topic_dict.items()]
def transcribe_and_classify_topics(input_link):
    yt = pt.YouTube(input_link)
    stream = yt.streams.filter(only_audio=True)[0]
    stream.download(filename="audio_english.mp3")
    model = whisper.load_model("base")
    result = model.transcribe("audio_english.mp3")
    transcribed_text = result["text"]

    def preprocess(text):
        tokens = word_tokenize(text.lower())
        tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
        tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
        return tokens

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
        topic_assignment = max(lda_model[bow], key=lambda x: x[1])[0]
        topic_assignments.append(topic_assignment)

    grouped_sentences = [[] for _ in range(optimal_num_topics)]
    for i, sentence in enumerate(sent_tokenize(transcribed_text)):
        grouped_sentences[topic_assignments[i]].append(sentence)

    topic_descriptions = []
    for sentences in grouped_sentences:
        if sentences:
            topic_descriptions.append(sentences[0])

    topic_dict = {}
    for i, sentences in enumerate(grouped_sentences):
        if sentences:
            topic_key = f"Topic {i}: {topic_descriptions[i]}"
            topic_values = sentences[1:]
            topic_dict[topic_key] = topic_values

    os.remove("audio_english.mp3")
    return transcribed_text, topic_dict
def get_word_meaning(word):
    synsets = wordnet.synsets(word)
    if synsets:
        return synsets[0].definition()
    else:
        return "Meaning not found."
def on_meaning_button_clicked(b):
    word = word_input.value.strip()
    if word:
        meaning = get_word_meaning(word)
        meaning_output.value = meaning
    else:
        meaning_output.value = "Please enter a word."
transcribe_button = widgets.Button(description="Transcribe and Classify Topics")
transcribe_button.on_click(on_transcribe_button_clicked)

meaning_button = widgets.Button(description="Get Meaning")
meaning_button.on_click(on_meaning_button_clicked)

input_link = widgets.Text(placeholder='Enter YouTube video link', layout={'width': '50%'})
word_input = widgets.Text(placeholder='Enter a word for dictionary meaning', layout={'width': '50%'})
transcript_output = widgets.Textarea(placeholder='Transcript will appear here', layout={'height': '200px', 'width': '100%'})
topics_output = widgets.Textarea(placeholder='Topics will appear here', layout={'height': '200px', 'width': '100%'})
meaning_output = widgets.Textarea(placeholder='Meaning will appear here', layout={'height': '50px', 'width': '50%'})

display(input_link)
display(transcribe_button)
display(transcript_output)
display(topics_output)
display(word_input)
display(meaning_button)
display(meaning_output)
