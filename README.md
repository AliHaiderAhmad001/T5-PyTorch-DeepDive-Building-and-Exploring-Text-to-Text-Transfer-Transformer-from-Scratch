# T5-PyTorch-DeepDive-Building-and-Exploring-Text-to-Text-Transfer-Transformer-from-Scratch
Implementation for the Paper [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](https://arxiv.org/abs/1910.10683) Alongside a Comprehensive Demonstration.

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Ubuntu-orange.svg)](https://www.ubuntu.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
![Transformers](https://img.shields.io/badge/transformers-4.36-yellow.svg)
[![Python](https://img.shields.io/badge/Python-3-blue.svg)](https://www.python.org/)


## Goals and Scope

- **Educational Purpose**: This project is intended as a learning resource. It aims to offer a clear understanding of the T5 architecture through a hands-on approach by implementing the model from scratch using PyTorch.
- **Model Understanding**: Users can delve into each part of the model's architecture, including the encoder, decoder, and the self-attention mechanism, to understand how they contribute to the model's performance.
- **Project Customization**: While the primary focus is educational, users are encouraged to modify and adapt the provided components to fit the requirements of their projects.

## Limitations

- **Dataset Handling**: The current implementation of the dataset loading and processing is optimized for simplicity and small-scale data. Users intending to train models on larger datasets may need to modify the dataset handling components for efficiency.
- **Pre-training Only**: This codebase is set up for model pre-training. To use this implementation for fine-tuning on specific tasks, further modifications may be required.

## Getting Started

### Prerequisites

Ensure you have Python 3.7+ and PyTorch installed. Additionally, this project uses the Transformers library for tokenization.

### Installation

Clone the repository and install the required Python packages:

```
git clone https://github.com/AliHaiderAhmad001/T5-PyTorch-DeepDive-Building-and-Exploring-Text-to-Text-Transfer-Transformer-from-Scratch.git
cd T5-PyTorch-DeepDive-Building-and-Exploring-Text-to-Text-Transfer-Transformer-from-Scratch
pip install -r requirements.txt
```

### Usage

To train the model with the default settings, simply run:

```
python run.py
```

You can customize the training by passing additional arguments. For example, to change the number of training epochs and batch size, you can run:

```
python run.py --epochs 3 --batch_size 32
```

## Quick review of the paper

The aim of the paper was to explore the most popular current transfer learning techniques in natural language processing and present a unified framework that converts all text-to-text language problems into a text-to-text format. The paper compares, according to a specific approach, pre-training objectives, architectures, unlabeled data sets, transfer approaches, and more. By combining insights from this exploration they were able to deliver a T5 model that achieved SOTA results on several GELU and Super-GELU benchmarks such as text summarization, question answering, text classification, and more. Of course, the paper is very large, about 50 pages, and full of information.

The paper proposed (almost) the first model that combines most language tasks under a "text-to-text" format (in conjunction with the third version of the GPT-3 model, which also provided a unified framework). The input text is what is given to the model, and the output text is the corresponding result. This also means that the pre-training and fine-tuning process is consistent no matter the task.

### Text-to-text approach

One of the big benefits of framing language tasks into this format is that it allows us to achieve "consistency" between the pre-training objective and the fine-tuning process. What does it mean? This text-to-text framework allows us to apply the same model, objective, training procedures, and decoding process directly to every task we consider. That is, instead of designing specific structures or deleting and adding output layers that serve specific downstream task, T5 treats all tasks as the task of converting input text to output text. For example, for translation, the input text is preceded by “English to German translation:” followed by the sentence to be translated, and the model generates the translated sentence. That is, the input is as follows: `translate English to German: That is good.`

The model is trained to give us: `Das ist gut.`

- For text classification, the input text is the text to be classified, and the model generates a classification word (for example, “positive” or “negative” for sentiment analysis). Although the model here applies a probability distribution to all the vocabulary of the language, it will not deviate from the aforementioned classes, because this "monster" called a "neural network", which is capable of understanding and learning very complex patterns and relationships, why would it not be able to understand this without explicitly defining the main classes?

### Model architecture

At its core, the T5 model is based on the Transformer architecture, which has become a standard architecture for most modern NLP models. The T5 model consists of an encoder and a decoder.

Of course many architectures have been tried (only those related to transformers). According to the paper, a key distinct factor for the different architectures is the “mask” used by the different self-attention mechanisms in the model. Remember that the self-attention process in a transformer involves taking a sequence of tokens as input and outputting a new sequence of the same length. Each token in the output sequence is a weighted average of the tokens in the input sequence. An “attention mask” is then used "mask" to filter certain weights from the calculations, in order to create a restriction on the input tokens that are included in the calculation of the tokens in the output sequence at a specific time step. That is, for example, the token number `i` of the output string is calculated according to the weighted average of all the input tokens `j<i` only (i.e. restricting vision or attention). This type of restriction is called “causal masking” (this technique is used during training so that the model cannot “see into the future” while producing its output).

The first model architecture they consider is an encoder-decoder transformer. The encoder uses a "fully-visible" attention mask. This type allows the self-attention mechanism to pay attention to all the tokens in the input seq when each token in the output seq is produced. On the other flip, the decoder uses an attention mask called "causal masking". The decoder is used to produce the output sequence autoregressively. This means that at each output time step, a token is sampled from the probability distribution of the model output (after applying Softmax and obtaining Logits probability) and fed back into the model to produce a prediction for the next output time step, and so on. As such, the decoder from the transformer (without the encoder) can be used as an LM language model. This forms the structure of the second model studied.

The structure of the second model is a transformer decoder, meaning without an encoder, as I mentioned. Of course, many previous works have used the architecture with a “language modeling objective” as GPT (Radford et al., 2018). Language models are typically used for compression or generating sequences, but can also be used within text-to-text by simply mapping inputs with targets. For example, we consider the case of translation from English to German, let us assume a training sample: `That is good.` The corresponding translation is `Das ist gut.` Simply train the model on input from the shape (with a causal mask): `translate English to German: That is good. target: This is good.` If we want to get the model's prediction (for example, in the inference phase), the model will be fed with the prefix:
`translate English to German: That is good. target:`

It is the responsibility of the model to predict the rest of the sequence in autoregressively manner. In this way, the model can predict the sequence of outputs given the inputs, meeting the needs of text-to-text tasks.

Of course, one of the main drawbacks of such a structure lies in the limited view of the model due to which the causal mask `j<i`. Because of causal masking, the prefix representation that the model uses to create each text unit is limited to information from the previous tokens only. That is, it cannot take advantage of the full context of the input sequence, including any tokens that follow the current text unit. This means that the model's understanding and representation of the input context is "unnecessarily limited" because it does not take advantage of the full context that may be available or necessary for more accurate or context-appropriate generation.

Of course, this problem can be avoided simply by changing the masking style used. Instead of using a causal mask, we use another masking method called “prefix LM”, which is a combination of the two types of masking mentioned above, so that the prefix part is “fully-visible” and the remaining part is "causal".

So, the three basic architectures that were studied are: Prefix-LM, Language model and Encoder-decoder model.

These three models were trained and tested with the two pre-training objectives “Denoising and LM” and the result was best with Encoder-decoder architecture.

### Pre-training and fine-tuning

The T5 model undergoes a two-stage training process (like most other LM, Transfer models): pretraining and fine-tuning.

- **Pre-training:** Here the model is trained on a huge set of text data using self-supervised learning tasks (or “pre-training objectives”). Through it, the model acquires general knowledge and understanding of the language that allows it to apply it to downstream tasks (such as translation or classification). The most common pre-training objectives are:

1- denoising objective(MLM or BERT-style, Replace corrupted spans and Drop corrupted tokens,MASS ..etc). 

2- language modeling(Prefix language modeling and Casual language modeling)

3. deshuffling.

Of course, in the paper, all of these types were tested and Replace corrupted spans (or I.i.d. noise, replacement spans) were adopted because it achieved the best performance according to the comparison approach they developed.

- **Fine-tuning:** Here the model (pre-trained) is trained more and more specifically on a smaller dataset tailored to a specific task. The model is adapted such that it becomes specialized at the task and thus performs higher.

Of course, the model-Encoder-decoder architecture is trained according to several pre-training approaches (or objectives), some of which are modified to fit this model architecture, and in other cases, training objectives that combine concepts from several approaches are used:


1.Prefix language modeling:
Input: Thank you for inviting
Target: me to your party last week .

2. BERT-style
Example:
Input: Thank you <M> <M> me to your party apple week .
Target: (original text)

3. Deshuffling
Example:
Input: party me for your to . last fun you inviting week Thank
Target: (original text)

4. MASS-style 
Example:
Input: Thank you <M> <M> me to your party <M> week
Target: (original text)

5. I.i.d. noise, replace spans 
Example:
Input: Thank you <X> me to your party <Y> week
Target: <X> for inviting <Y> last <Z>

6. I.i.d. noise, drop tokens
Example: 
Input: Thank you me to your party week .
Target: for inviting last

7. Random spans
Example: 
Input: Thank you <X> to <Y> week .
Target: <X> for inviting me <Y> your party last <Z>

Of course, types 4, 5, and 6 are modifications of the BERT-style. According to experiments, their performance was similar, but they outperform types 7, 3, and 2.

## Contributing

This project is open to contributions and corrections. If you find any inconsistencies with the original T5 paper or have suggestions for improving the implementation, please feel free to open an issue or submit a pull request.

## Acknowledgments

This project is inspired by the original T5 paper and the PyTorch community. Special thanks to the authors of the T5 model for their groundbreaking work in NLP.

## License

This project is licensed under the MIT License.


