/**
 * Evaluator class to run multiple evaluation metrics for a tasks such as exact text matching.
 */
export class Evaluator {
    constructor() {
    }

    /**
     * Run multiple metrics for a prediction against the ground truth and return the results.
     *
     * @param pred - Predicted string
     * @param truth - Ground truth string
     * @param latencyMs - Latency measured during inference
     * @returns {{exact: number, totalTokens: number, tokensPerSecond: number}}
     */
    evaluate(pred, truth, latencyMs) {
        const total_words = this._countWords(pred);
        return {
            exactMatch: this._exactTextMatch(pred.answer, truth),
            totalWords: total_words,
            wordsPerSecond: this._wordsPerSecond(total_words, latencyMs)
        };
    }

    /**
     * Check the prediction for exact text match against the ground truth
     *
     * @param pred - Predicted string
     * @param truth- Ground truth string
     * @returns {number}
     * @private
     */
    _exactTextMatch(pred, truth) {
        return this._normalize(pred) === this._normalize(truth) ? 1 : 0;
    }


    /**
     * Normalize a string to avoid false negatives due to spaces or capitalization
     * Convert input to a string in case it is not already
     *
     * @param s - Input string
     * @returns {string}
     * @private
     */
    _normalize(s) {
        return String(s || '').trim().toLowerCase();
    }

    /**
     * Count the number of tokens (words) in a string
     *
     * @param s - Input string
     * @returns {number}
     */
    _countWords(s) {
        return String(s || '').trim().split(/\s+/).filter(Boolean).length;
    }

    /**
     * Calculate tokens per second given token count and latency in ms
     * @param wordCount - Number of tokens
     * @param latencyMs - Latency in milliseconds
     * @returns {number}
     */
    _wordsPerSecond(wordCount, latencyMs) {
        return latencyMs > 0 ? wordCount / (latencyMs / 1000) : 0;
    }

}