
/**
 * DatasetLoader loads a dataset and prepare it for the processing.
 */
export class DatasetLoader {
    constructor(datasetName = 'boolq_validation') {
        this.running = false;
        this._dataset = null;
        this._datasetName = datasetName

        this.loadDataset(this._datasetName);
    }

    /**
     * Load the dataset from CSV file based on the given name
     * If a comma appears inside a quote (context) it is not interpreted as a delimiter
     *
     * @param name - Name of the csv dataset to load without file extension
     * @private
     */
    loadDataset(name) {
        const path = `./data/${name}.csv`;

        return fetch(path)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Dataset file not found: ${path}`);
                }
                return response.text();
            })
            .then(data => {
                const lines = data.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
                // drop header
                lines.shift();

                this._dataset = lines
                    .filter(l => l.trim().length > 0)
                    .map(line => {
                        let id, answer, full_prompt;

                        // load different datasets based on the dataset name
                        switch (name) {
                            case 'boolq_validation':
                                ({id, full_prompt, answer} = this._loadBoolQLine(line));
                                break;
                            case 'spam_ham_dataset':
                                ({id, full_prompt, answer} = this._loadSpamHamLine(line));
                                break;
                            case 'imdb_dataset':
                                ({id, full_prompt, answer} = this._loadIMDBLine(line));
                                break;
                            case 'ag_news_test':
                                ({id, full_prompt, answer} = this._loadAGNewsLine(line));
                                break;
                            case 'lorem_ipsum_dataset':
                                ({id, full_prompt, answer} = this._loadLoremIpsumLine(line));
                                break;
                            default:
                                throw new Error(`DatasetLoader: Unsupported dataset name '${name}'`);
                        }

                        return {id: id, prompt: full_prompt, groundTruth: answer};
                    });

                console.log(`✅ Dataset '${name}' loaded with ${this._dataset.length} items.`);
                console.log(this._dataset.slice(0, 2)); // log first 2 items for verification
                return this._dataset;
            })
            .catch(error => {
                console.error(error);
            });
    }


    /**
     * Load a single line from the BoolQ dataset and prepare the prompt
     *
     * @param line - A single line from the BoolQ CSV dataset
     * @returns {{full_prompt: string, answer: *, id: *}}
     * @private
     */
    _loadBoolQLine(line) {
        // parse line into fields handling quoted commas
        const [id, question, answer, context] = this._parseCSVLine(line);

        // set the prompt
        const full_prompt = `Question: ${question}
                                        Context: ${context}
                                        Instructions: Answer with ONLY the word "true" or "false". Do not provide any explanation or additional text.
                                        Answer:`;

        return {id, full_prompt, answer}
    }


    /**
     * Load a single line from the SpamHam dataset and prepare the prompt
     *
     * @param line - A single line from the SpamHam CSV dataset
     * @returns {{full_prompt: string, answer: (string), id: *}}
     * @private
     */
    _loadSpamHamLine(line) {
        let [id, text, is_spam] = this._parseCSVLine(line);

        // convert answer (is a string due to csv parsing) to string boolean
        is_spam = (is_spam === '1') ? 'true' : 'false';

        // set the prompt
        const full_prompt = `Task: Determine whether the following message is spam or not.
                                        Instructions: Answer with ONLY the word "true" or "false". Do not provide any explanation or additional text.
                                        Message: ${text}
                                        Answer:`;

        return {id, full_prompt, answer: is_spam}
    }


    /**
     * Load a single line from the IMDB dataset and prepare the prompt
     *
     * @param line - A single line from the IMDB CSV dataset
     * @returns {{full_prompt: string, answer: *, id: *}}
     * @private
     */
    _loadIMDBLine(line) {
        let [id, review, sentiment] = this._parseCSVLine(line);

        // set the prompt
        const full_prompt = `Task: Determine whether the sentiment of the following review is positive or negative.
                                        Instructions: Answer with ONLY the word "positive" or "negative". Do not provide any explanation or additional text.
                                        Review: ${review}
                                        Sentiment:`;

        return {id, full_prompt, answer: sentiment}
    }


    /**
     * Load a single line from the AG News dataset and prepare the prompt
     *
     * @param line - A single line from the AG News CSV dataset
     * @returns {{full_prompt: string, answer: *, id: *}}
     * @private
     */
    _loadAGNewsLine(line) {
        let [id, class_index, title, description] = this._parseCSVLine(line);

        // set the prompt
        const full_prompt = `Task: Determine whether the following news article belong to world, sports, business or Sci/Tech category.
                                        Categories: World (1), Sports (2), Business (3), Sci/Tech (4).
                                        Instructions: Answer with ONLY the id (1,2,3 or 4) of the class. Do not provide any explanation or additional text.
                                        News Title: ${title}
                                        News Description: ${description}
                                        `;

        return {id, full_prompt, answer: class_index}
    }


    /**
     * Load a single line from the Lorem Ipsum dataset and prepare the prompt
     *
     * @param line - A single line from the lorem ipsum CSV dataset
     * @returns {{full_prompt: string, answer: *, id: *}}
     * @private
     */
    _loadLoremIpsumLine(line) {
        let [id, text, char_count, word_count, answer] = this._parseCSVLine(line);

        // set the prompt
        const full_prompt = `Task: Determine the last word of the provided text and return it.
                                        Instructions: Answer with ONLY the last word. Do not provide any explanation or additional text.
                                        Text: ${text}
                                        `;

        return {id, full_prompt, answer}
    }



    /**
     * Parse a single CSV line into fields, handling quoted fields with commas
     *
     * @param line - A single line from a CSV file
     * @private
     */
    _parseCSVLine(line) {

        // inline CSV parse with quotes support
        const fields = [];
        let cur = '';
        let inQuotes = false;

        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (inQuotes) { // if we are in a quote we just look for the quote ending
                if (ch === '"') {
                    // escaped quote ""
                    if (i + 1 < line.length && line[i + 1] === '"') {
                        cur += '"';
                        i++;
                    } else {
                        inQuotes = false;
                    }
                } else {
                    cur += ch;
                }
            } else {   // only if we are not in a quote we count the comma as e delimiter
                if (ch === ',') {
                    fields.push(cur);
                    cur = '';
                } else if (ch === '"') {
                    inQuotes = true;
                } else {
                    cur += ch;
                }
            }
        }
        fields.push(cur);
        return fields;
    }
}

