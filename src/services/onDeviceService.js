// OnDeviceService: uses Xenova's transformers.js to run a small causal LM in browser
// Uses ES module import for Xenova's transformers.js
import {pipeline} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.0';


/**
 * On device llm inference service using transformers.js
 * TODO Implement this class!
 */
export class OnDeviceService {
    constructor({modelName = '', quantization = 'fp32'} = {}) {
        this.modelName = modelName;
        this.modelQuantization = quantization;
        this._ready = false;
        this._model = null;
    }


    /**
     * Load the model into memory to be ready for inference.
     * Download the model if not already cached. Cache the model for future use.
     *
     * @param progressCb
     * @returns {Promise<void>}
     */
    async load(progressCb) {
        console.log(`⬇️ Download Model '${this.modelName}'...`);
        // Provide a default progress callback if none is given
        const defaultProgressCb = (progress) => {
            if (progress && typeof progress === 'object') {
                if (progress.status) {
                    console.log(`[Model Loading] ${progress.status}`);
                }
                if (progress.loaded && progress.total) {
                    const percent = ((progress.loaded / progress.total) * 100).toFixed(1);
                    console.log(`[Model Loading] ${percent}% (${progress.loaded}/${progress.total} bytes)`);
                }
            } else {
                console.log(`[Model Loading] Progress:`, progress);
            }
        };

        this._model = await pipeline('text-generation', this.modelName, {
            progress_callback: progressCb || defaultProgressCb,
            device: 'webgpu', // run on WebGPU if available
            dtype: this.modelQuantization, // set model quantization
        });
        console.log(`✅ Model '${this.modelName}' loaded and ready.`);
        this._ready = true;
    }


    /**
     * Returns if the model is loaded and ready for inference
     * @returns {boolean}
     */
    isReady() {
        return this._ready;
    }


    /**
     * Perform inference on the on-device model
     * TODO Implement inference
     *
     * @param prompt - The input prompt string
     * @param maxNewTokens - Maximum number of new tokens to generate
     * @returns {Promise<string>}
     */
    async infer(prompt, {maxNewTokens = 50} = {}) {
        if (!this._ready || !this._model) {
            console.log("model not ready:", this._ready, this._model);
            throw new Error('Model not loaded. Call load() first.');
        }
        console.log("🔄 Running inference on-device for prompt:\n", prompt);

        const messages = [
            { role: "user", content: prompt },
        ];

        const output = await this._model(messages, {
            max_new_tokens: maxNewTokens,
            temperature: 0.2,
        });

        console.log("✅ Completed inference on-device for prompt:\n", prompt);

        // take last generated text which corresponds to the model's answer
        const generated_output = output[0]?.generated_text;
        const text = generated_output[generated_output.length - 1]?.content.trim() || '';

        // todo calculate input and output tokens
        return {answer: text, stats: {input_tokens: undefined, output_tokens: undefined}};
    }

    /**
     * Update configuration with new values
     *
     * @param modelName - The name of the model to use
     */
    updateConfig({modelName, quantization} = {}) {
        if (modelName) this.modelName = modelName;
        if (quantization) this.modelQuantization = quantization;
    }


    /**
     * Retrieve the name of the currently loaded model.
     *
     * @returns {string} - The name of the model as a string.
     */
    getModelName(){
        return this.modelName;
    }
}