import {measureAsync, sleep} from './utils.js';


/**
 * RequestManager routes inference requests to on-device or cloud services based on a routing strategy and configurations.
 * The manager does orchestrate the inference requests, collects statistics, evaluates the results and returns the final statistic.
 *
 * We provide different routing strategies:
 * - always_cloud: all requests go to cloud
 * - always_device: all requests go to device
 * - probabilistic: each request goes to cloud with a defined probability
 * - roundrobin: requests alternate between cloud and device
 * - hero: routes to the server with the shortest expected queue time
 */
export class RequestManager {
    constructor({
                    deviceService,
                    cloudService,
                    evaluator,
                    logger = null,
                    routeStrategy = 'roundrobin',
                    cloudProb = 0.5,
                    devicePerfModel = {slope: 0, intercept: 0},
                    cloudPerfModel = {slope: 0, intercept: 0}
                } = {}) {

        /**
         * On-device inference service
         */
        this.device = deviceService;

        /**
         * Cloud inference service
         */
        this.cloud = cloudService;

        /**
         * Evaluator instance for evaluating inference results
         */
        this.evaluator = evaluator;

        /**
         * Optional logger callback function
         * @type {null}
         */
        this.logger = logger;

        /**
         * Routing strategy (always_cloud, always_device, probabilistic, roundrobin)
         * @type {string}
         */
        this.routeStrategy = routeStrategy;

        /**
         * Probability of routing to cloud when using 'probabilistic' strategy
         * @type {number}
         */
        this.cloudProb = cloudProb;

        /**
         * Performance model for the device {slope, intercept}
         * @type {{slope: number, intercept: number}}
         */
        this.devicePerfModel = devicePerfModel;

        /**
         * Performance model for the cloud {slope, intercept}
         * @type {{slope: number, intercept: number}}
         */
        this.cloudPerfModel = cloudPerfModel;

        // Initialize Thompson samplers using offline fits as priors
        const deviceMu0 = [this.devicePerfModel.intercept || 0, this.devicePerfModel.slope || 0];
        const cloudMu0 = [this.cloudPerfModel.intercept || 0, this.cloudPerfModel.slope || 0];

        // Hyperparameters: priorLambda and sigma2 (tune as needed)
        const priorLambda = 1e-3;      // prior regularization (fast adaptation: 1e-6 … 1e-3, balanced: 1e-2 … 1.0, slow adaptation (anchor to offline fit): 10 … 1e3)
        const observationSigma2 = 1e4; // noise variance (ms^2) => sd ~100ms (low-noise (sd ≈ 20–50 ms): 400 … 2500, medium-noise (sd ≈ 50–100 ms): 2500 … 10000, high-noise (sd ≈ 100–200 ms): 10000 … 40000)

        this.tsDevice = new LinearThompsonSampler(deviceMu0, priorLambda, observationSigma2);
        this.tsCloud = new LinearThompsonSampler(cloudMu0, priorLambda, observationSigma2);

        /**
         * Internal round robin counter (even = cloud, odd = device)
         * @type {number}
         * @private
         */
        this._rrCounter = 0;

        /**
         * Statistics about routing and evaluations of this job run
         * @type {{cloud: number, evaluations: *[], count: number, device: number, totalLatencyMs: number}}
         */
        this.stats = {count: 0, cloud: 0, device: 0, totalLatencyMs: 0, results: []};

        /**
         * Cloud job queue
         * @type {*[]}
         */
        this.cloud_queue = [];

        /**
         * Device job queue
         * @type {*[]}
         */
        this.device_queue = [];

        // start processing jobs from the queues
        this.runOnDeviceJobsFromQueue();
        this.runCloudJobsFromQueue();
    }

    /**
     * Push a job to the appropriate queue based on routing strategy.
     *
     * @param job - The job to be processed
     */
    pushJob(job) {
        // get routing strategy and inference service
        const route = this._choose(job);
        console.log(`Device Queue Length: ${this.device_queue.length}, \nCloud Queue Length: ${this.cloud_queue.length}`);

        if (route === 'cloud') {
            this.cloud_queue.push(job);
        } else {
            this.device_queue.push(job);
        }
    }


    /**
     * Update routing configuration
     *
     * @param routeStrategy - New routing strategy
     * @param cloudProb - New cloud probability for 'probabilistic' strategy
     * @param devicePerfModel
     * @param cloudPerfModel
     */
    updateRouting({routeStrategy, cloudProb, devicePerfModel, cloudPerfModel}) {
        if (routeStrategy) this.routeStrategy = routeStrategy;
        if (cloudProb !== undefined) this.cloudProb = cloudProb;
        if (devicePerfModel) this.devicePerfModel = devicePerfModel;
        if (cloudPerfModel) this.cloudPerfModel = cloudPerfModel;
    }


    /**
     * Handle device jobs by routing it to the appropriate service, as long as there are jobs in the queue.
     *
     * @returns {Promise<void>}
     */
    async runOnDeviceJobsFromQueue() {
        while (true) {
            if (this.device_queue.length > 0) {
                const job = this._getNextJobFromQueue(this.device_queue, 'fifo');
                const service = this.device;
                const route = 'device';

                // run the job and await until compteted
                await this._runJob(job, route, service);
            }

            // sleep for 10ms to not run into memory leak
            await sleep(10);
        }
    }

    /**
     * Handle cloud jobs by routing it to the appropriate service, as long as there are jobs in the queue.
     *
     * @returns {Promise<void>}
     */
    async runCloudJobsFromQueue() {
        while (true) {
            if (this.cloud_queue.length > 0) {
                const job = this._getNextJobFromQueue(this.cloud_queue, 'fifo');
                const service = this.cloud;
                const route = 'cloud';

                // run the job and await until it completes
                await this._runJob(job, route, service);
            }

            // sleep for 10ms to not run into memory leak
            await sleep(10);
        }
    }


    /**
     * Run the given job on the specified service and record statistics.
     *
     * @param job - The job object containing prompt and ground truth
     * @param route - The selected route ('cloud' or 'device')
     * @param service - The inference service to use
     * @returns {Promise<void>}
     * @private
     */
    async _runJob(job, route, service) {
        let full_prompt = job.prompt; // ensure string input

        // this is a little workaround to disable the thinking mode in qwen models
        if (service.getModelName().toLowerCase().includes("qwen3".toLowerCase())) {
            full_prompt = full_prompt; // + "/no_think";
            console.log("ℹ️ \"/no_think\" was added to the prompt to avoid thinking")
        }

        let response, latencyMs, cleanedResponse; // response is object with .answer and .stats
        try {
            // Mark inference start
            job.timestamps.inferenceStart = Date.now();

            const {res, ms} = await measureAsync(() => service.infer(full_prompt));
            response = res;
            latencyMs = ms;

            // Mark inference end
            job.timestamps.inferenceEnd = Date.now();
        } catch (err) {
            response = `__error__:${err.message}`;
            latencyMs = -1;
            job.timestamps.inferenceEnd = Date.now();
        }

        // Calculate timing metrics
        const queueingTime = job.timestamps.inferenceStart - job.timestamps.jobStart;
        const inferenceTime = job.timestamps.inferenceEnd - job.timestamps.inferenceStart;
        const totalLatency = job.timestamps.inferenceEnd - job.timestamps.jobStart;


        // clean response
        cleanedResponse = this._cleanResponse(response);

        // evaluate result and store results
        const evalRes = this.evaluator.evaluate(cleanedResponse, job.groundTruth, latencyMs);
        this._record(route, latencyMs, evalRes, job, cleanedResponse, {queueingTime, inferenceTime, totalLatency});

        if (this.logger) {
            try {
                this.logger({job, route, latency: latencyMs, evalRes, response: cleanedResponse.answer, queueingTime, inferenceTime, totalLatency});
            } catch (error) {
                console.error("Logger encountered an error:", error);
            }
        }

        // update Thompson sampler with observed inference time (ms)
        try {
            if (latencyMs > 0) {
                const x = [1, job.prompt.length];
                const y = inferenceTime; // ms
                if (route === 'device') {
                    this.tsDevice.update(x, y);
                } else {
                    this.tsCloud.update(x, y);
                }
                // update public linear models (used by _decideHERO UI)
                this._updateLinearModelsForHERO();
            }
        } catch (err) {
            console.warn("TS update failed:", err);
        }

        // logging on console
        console.log(cleanedResponse)
        console.log("🎯 Models Answer: " + response.answer +
            "; \nCleaned Answer: " + cleanedResponse.answer +
            '; \nGround Truth: ' + job.groundTruth +
            "; \nInference Time: " + inferenceTime.toFixed(2) + "ms" +
            "; \nQueueing Time: " + queueingTime.toFixed(2) + "ms" +
            "; \nTotal Latency: " + totalLatency.toFixed(2) + "ms");
    }

    _getNextJobFromQueue(queue, policy) {
        // currently only FIFO is implemented
        return queue.shift();
    }

    /**
     * Choose a route based on the configured strategy.
     *
     * @returns {string} 'cloud' or 'device'
     * @private
     */
    _choose(job) {
        switch (this.routeStrategy) {
            case 'always_cloud':
                return 'cloud';
            case 'always_device':
                return 'device';
            case 'probabilistic':
                return Math.random() < this.cloudProb ? 'cloud' : 'device';
            case 'roundrobin':
                this._rrCounter++;
                return this._rrCounter % 2 === 0 ? 'cloud' : 'device';
            case 'hero':
                return this._decideHERO(job);
            default:
                return 'device';
        }
    }

    /**
     * Decide route based on our HERO policy.
     *
     * @param job
     * @returns {string}
     * @private
     */
    _decideHERO(job) {
        const now = Date.now();
        const input_size = job.prompt.length;

        // Thompson sample thetas (intercept, slope) in ms
        const thetaDevice = this.tsDevice.sampleTheta();
        const thetaCloud = this.tsCloud.sampleTheta();

        // predict inference time for the new job on both servers
        const device_predicted_inference_time = thetaDevice[0] + thetaDevice[1] * input_size;
        const cloud_predicted_inference_time = thetaCloud[0] + thetaCloud[1] * input_size;


        // Get the last job in the queue for both servers, to estimate when they will be free
        const lastDeviceJob = this.device_queue.length > 0 ? this.device_queue[this.device_queue.length - 1] : null;
        const lastCloudJob = this.cloud_queue.length > 0 ? this.cloud_queue[this.cloud_queue.length - 1] : null;

        // Calculate when each server will be free based on the last job in the queue
        const device_free_at = Math.max(now, lastDeviceJob?.hero_predictions?.device?.expectedFinishTime || 0);
        const cloud_free_at = Math.max(now, lastCloudJob?.hero_predictions?.cloud?.expectedFinishTime || 0);

        // Calculate expected finish time for the new job on both servers
        const device_expected_finish_time = device_free_at + device_predicted_inference_time;
        const cloud_expected_finish_time = cloud_free_at + cloud_predicted_inference_time;

        // calculate expected total time for both servers (predicted inference time + time until server is free)
        const device_expected_total_time = device_expected_finish_time - now;
        const cloud_expected_total_time = cloud_expected_finish_time - now;

        // store the predicted values in the job for logging and analysis
        job.hero_predictions = {
            device: {
                predictedInferenceTime: device_predicted_inference_time,
                predictedTotalTime: device_expected_total_time,
                expectedFinishTime: device_expected_finish_time,
                numberOfJobsInQueue: this.device_queue.length || 0
            },
            cloud: {
                predictedInferenceTime: cloud_predicted_inference_time,
                predictedTotalTime: cloud_expected_total_time,
                expectedFinishTime: cloud_expected_finish_time,
                numberOfJobsInQueue: this.cloud_queue.length || 0
            }
        };


        // Choose the server with the earlier expected finish time
        if (device_expected_finish_time <= cloud_expected_finish_time) {
            return 'device';
        } else {
            return 'cloud';
        }
    }

    /**
     * Update public linear model values from sampler posteriors (used by UI & logging)
     */
    _updateLinearModelsForHERO() {
        const devMean = this.tsDevice.posteriorMean();
        const cloudMean = this.tsCloud.posteriorMean();

        // posteriorMean returns [intercept, slope]
        this.devicePerfModel = { intercept: devMean[0], slope: devMean[1] };
        this.cloudPerfModel = { intercept: cloudMean[0], slope: cloudMean[1] };

        // dispatch update event for the frontend
        if (typeof window !== 'undefined') {
            window.dispatchEvent(new CustomEvent('perfModelsUpdated', {detail: { device: this.devicePerfModel, cloud: this.cloudPerfModel }}));
        }
    }

    /**
     * Record statistics for the given job evaluation.
     * Increases counters for total requests and cloud/device usage.
     * Updates the total latency.
     *
     * @param route - The route taken ('cloud' or 'device')
     * @param latency - Latency in milliseconds
     * @param evalRes - Evaluation result object
     * @param job - The job object
     * @param text - The inference result text
     * @param timingMetrics - Object containing queueingTime, inferenceTime, and totalLatency
     * @private
     */
    _record(route, latency, evalRes, job, text, timingMetrics) {
        this.stats.count++;
        if (route === 'cloud') this.stats.cloud++; else this.stats.device++;
        if (latency > 0) this.stats.totalLatencyMs += latency;
        this.stats.results.push({
            job: job,
            route,
            latency,
            evalRes,
            text,
            queueingTime: timingMetrics.queueingTime,
            inferenceTime: timingMetrics.inferenceTime,
            totalLatency: timingMetrics.totalLatency,
            timestamps: job.timestamps
        });
    }

    /**
     * Remove reasoning/thinking tokens and sections from a model's response.
     * Supports various formats (XML tags, special tokens, markdown, etc.) used by reasoning models.
     * Returns a cleaned response object with only the final answer for evaluation.
     *
     * @param response - The uncleaned response object (may include reasoning/thinking sections)
     * @return {object|string} - Cleaned response object or original error string
     * @private
     */
    _cleanResponse(response) {
        // If response is an error string, return as-is
        if (typeof response === 'string' && response.startsWith('__error__')) {
            return response;
        }

        // Clone the response object to avoid mutating the original
        const cleanedResponse = { ...response };
        
        if (!cleanedResponse.answer || typeof cleanedResponse.answer !== 'string') {
            return cleanedResponse;
        }

        let cleanedAnswer = cleanedResponse.answer;

        // Define patterns for thinking tokens (common formats)
        const thinkingPatterns = [
            // XML-style tags
            /<think>[\s\S]*?<\/think>/gi,
            /<thinking>[\s\S]*?<\/thinking>/gi,
            /<reasoning>[\s\S]*?<\/reasoning>/gi,
            /<thought>[\s\S]*?<\/thought>/gi,
            
            // Special tokens
            /<\|startofthinking\|>[\s\S]*?<\|endofthinking\|>/gi,
            /<\|reasoning_start\|>[\s\S]*?<\|reasoning_end\|>/gi,
            
            // Markdown-style
            /\[THINKING\][\s\S]*?\[\/THINKING\]/gi,
            /\[REASONING\][\s\S]*?\[\/REASONING\]/gi,
            /\[THOUGHT\][\s\S]*?\[\/THOUGHT\]/gi,
            
            // Other common patterns
            /\*\*Thinking:\*\*[\s\S]*?(?=\*\*Answer:\*\*|$)/gi,
            /\*\*Reasoning:\*\*[\s\S]*?(?=\*\*Answer:\*\*|$)/gi,
        ];

        // Apply all patterns to remove thinking sections
        for (const pattern of thinkingPatterns) {
            cleanedAnswer = cleanedAnswer.replace(pattern, '');
        }

        // Clean up extra whitespace
        cleanedAnswer = cleanedAnswer.trim();
        
        // If we removed everything, keep original (safety check)
        if (cleanedAnswer.length === 0 && cleanedResponse.answer.length > 0) {
            console.warn('⚠️ Thinking token removal resulted in empty answer. Keeping original.');
            return cleanedResponse;
        }

        cleanedResponse.answer = cleanedAnswer;
        return cleanedResponse;
    }
}




/**
 * 2-D Bayesian linear regressor for Thompson Sampling.
 * Models y = x^T theta + noise, x = [1, n]
 * Maintains A = Lambda + (1/sigma2) X^T X  (2x2), b = Lambda*mu0 + (1/sigma2) X^T y (2x1)
 * Posterior: theta | data ~ N(mu = A^{-1} b, cov = A^{-1})
 */
class LinearThompsonSampler {
    constructor(mu0 = [0, 0], priorLambda = 1e-3, sigma2 = 1e4) {
        // mu0: prior mean [intercept, slope]
        // priorLambda: scalar multiplied with identity (regularization)
        // sigma2: observation noise variance (ms^2)
        this.mu0 = [...mu0];
        this.priorLambda = priorLambda;
        this.sigma2 = sigma2;

        // A (2x2) initialized to priorLambda * I
        this.A = [
            [priorLambda, 0],
            [0, priorLambda]
        ];

        // b (2x1) initialized to priorLambda * mu0
        this.b = [
            priorLambda * this.mu0[0],
            priorLambda * this.mu0[1]
        ];
    }

    /**
     * Sample theta ~ N(mu, cov)
     * Uses Cholesky decomposition for sampling from multivariate normal.
     *
     * @returns {*[]}
     */
    sampleTheta() {
        const cov = invert2x2(this.A); // cov = A^{-1}
        const mu = matVecMul(cov, this.b);

        // Cholesky of cov (2x2) for sampling
        const L = cholesky2x2(cov);
        const z0 = randn(), z1 = randn();
        // theta = mu + L * z
        const theta0 = mu[0] + L[0][0] * z0;
        const theta1 = mu[1] + L[1][0] * z0 + L[1][1] * z1;
        return [theta0, theta1];
    }

    /**
     * Update with one observation (x: [1, n], y: observed_time_ms)
     *
     * @param x
     * @param y
     */
    update(x, y) {
        // A += (1/sigma2) * x x^T
        const factor = 1.0 / this.sigma2;
        this.A[0][0] += factor * x[0] * x[0];
        this.A[0][1] += factor * x[0] * x[1];
        this.A[1][0] += factor * x[1] * x[0];
        this.A[1][1] += factor * x[1] * x[1];

        // b += (1/sigma2) * x * y
        this.b[0] += factor * x[0] * y;
        this.b[1] += factor * x[1] * y;
    }

    /**
     * Return posterior mean [intercept, slope]
     *
     * @returns {*}
     */
    posteriorMean() {
        const cov = invert2x2(this.A);
        return matVecMul(cov, this.b);
    }
}

/* ===== Helper linear algebra (small, 2x2 implementations) ===== */

function matVecMul(mat, vec) {
    return [
        mat[0][0] * vec[0] + mat[0][1] * vec[1],
        mat[1][0] * vec[0] + mat[1][1] * vec[1]
    ];
}

function invert2x2(m) {
    // returns inverse of 2x2 matrix m
    const a = m[0][0], b = m[0][1], c = m[1][0], d = m[1][1];
    const det = a * d - b * c;
    const eps = 1e-12;
    const detSafe = Math.abs(det) < eps ? (det >= 0 ? eps : -eps) : det;
    const invDet = 1.0 / detSafe;
    return [
        [ d * invDet, -b * invDet ],
        [ -c * invDet, a * invDet ]
    ];
}

function cholesky2x2(m) {
    // m must be symmetric positive definite
    const a = m[0][0];
    const b = m[0][1]; // equals m[1][0]
    const c = m[1][1];
    const l00 = Math.sqrt(Math.max(a, 1e-12));
    const l10 = b / l00;
    const l11 = Math.sqrt(Math.max(c - l10 * l10, 1e-12));
    return [
        [l00, 0],
        [l10, l11]
    ];
}

function randn() {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}