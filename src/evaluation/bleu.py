from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from tqdm import tqdm

TICKS_PER_BEAT = 24
TICKS_PER_MEASURE = 4 * TICKS_PER_BEAT
TICKS_PER_SENTENCE = 8 * TICKS_PER_MEASURE

class BleuScore:

    def __init__(self, sequence_len):
        self.seq_len = sequence_len

    def evaluate_bleu_score(self, predictions, targets, ticks=False, corpus=True):
        """
        Given an array of predicted ticks and its ground truth, compute the BLEU score across 8 measure "sentences".
        :param predictions: an N x 38 numpy matrix, where N is the number of predicted ticks to evaluate
        :param targets: an N x 38 numpy matrix, where N is the number of target ticks to be evaluated against
        :return: the BLEU score across the corpus of predicted ticks
        """
        if ticks:
            ref_sentences = self._ticks_to_sentences(targets)
            cand_sentences = self._ticks_to_sentences(predictions)
        else:
            ref_sentences = [[str(x) for x in seq] for seq in predictions]
            cand_sentences = [[str(x) for x in seq] for seq in targets]

        if corpus:
            bleu_score = corpus_bleu([[l] for l in ref_sentences], cand_sentences)
        else:
            bleu_score = 0.0
            num_sentences = 0

            for i in tqdm(range(len(ref_sentences))):
                sentence_bleu_score = sentence_bleu(ref_sentences[i], cand_sentences[i])
                print(sentence_bleu_score)
                bleu_score += sentence_bleu_score
                num_sentences += 1

            bleu_score /= num_sentences

        return bleu_score

    def _ticks_to_sentences(self, ticks):
        """
        Given an array of ticks, converts vector values to strings, returning a list of 8 measure "sentence" concatenations.
        :param ticks: an np array of ticks to convert to sentences
        :return: a list of sentences
        """

        sentences = []

        for seq in ticks:
            sentence = []
            for i in range(seq.shape[0]):
                word = ''.join([str(x) for x in seq[i, :]])
                sentence.append(word)
            sentences.append(sentence)

        # tick_ctr = 0
        # num_ticks = ticks.shape[0]
        # while tick_ctr < num_ticks:
        #     if num_ticks - tick_ctr >= self.seq_len:
        #         sentence_array = ticks[tick_ctr:tick_ctr + self.seq_len, :]
        #     else:
        #         sentence_array = ticks[tick_ctr:, :]
        #
        #     sentence = []
        #     for i in range(sentence_array.shape[0]):
        #         word = ''.join([str(x) for x in sentence_array[i, :]])
        #         sentence.append(word)
        #
        #     sentences.append(sentence)
        #
        #     tick_ctr += sentence_array.shape[0]

        return sentences
