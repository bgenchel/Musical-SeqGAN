from nltk.translate.bleu_score import corpus_bleu

TICKS_PER_BEAT = 24
TICKS_PER_MEASURE = 4 * TICKS_PER_BEAT
TICKS_PER_SENTENCE = 8 * TICKS_PER_MEASURE


def evaluate_belu_score(predictions, targets):
    """
    Given an array of predicted ticks and its ground truth, compute the BLEU score across 8 measure "sentences".
    :param predictions: an N x 38 numpy matrix, where N is the number of predicted ticks to evaluate
    :param targets: an N x 38 numpy matrix, where N is the number of target ticks to be evaluated against
    :return: the BLEU score across the corpus of predicted ticks
    """
    ref_sentences = ticks_to_sentences(targets)
    cand_sentences = ticks_to_sentences(predictions)
    bleu_score = corpus_bleu([[l] for l in ref_sentences], cand_sentences)
    return bleu_score


def ticks_to_sentences(ticks):
    """
    Given an array of ticks, converts vector values to strings, returning a list of 8 measure "sentence" concatenations.
    :param ticks: an np array of ticks to convert to sentences
    :return: a list of sentences
    """

    sentences = []

    tick_ctr = 0
    num_ticks = ticks.shape[0]
    while tick_ctr < num_ticks:
        if num_ticks - tick_ctr >= TICKS_PER_SENTENCE:
            sentence_array = ticks[tick_ctr:tick_ctr + TICKS_PER_SENTENCE, :]
        else:
            sentence_array = ticks[tick_ctr:, :]

        sentence = []
        for i in range(sentence_array.shape[0]):
            word = ''.join([str(x) for x in sentence_array[i, :]])
            sentence.append(word)

        sentences.append(sentence)

        tick_ctr += sentence_array.shape[0]

    return sentences
